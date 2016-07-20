# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops import rnn_cell
from translate import utils
from translate import decoders

from tensorflow.python.ops import variable_scope


class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self, src_ext, trg_ext, buckets, learning_rate, global_step, embeddings,
               src_vocab_size, trg_vocab_size, size, layers, max_gradient_norm, batch_size,
               num_samples=512, reuse=None, dropout_rate=0.0, embedding_size=None,
               bidir=False, freeze_variables=None, attention_filters=0,
               attention_filter_length=0, use_lstm=False, pooling_ratios=None,
               model_weights=None, binary_input=None,
               attention_window_size=0, **kwargs):
    """Create the model.

    Args:
      src_vocab_size: size of the sources vocabularies.
      trg_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      binary_input: list of encoder_names that directly read features instead
        of token ids.
    """
    size = size[0]   # for now, only one size
    self.buckets = buckets
    self.batch_size = batch_size
    self.encoder_count = len(src_ext)
    self.model_weights = model_weights
    self.binary_input = binary_input or []

    self.learning_rate = learning_rate
    self.global_step = global_step
    self.trg_vocab_size = trg_vocab_size

    assert len(src_vocab_size) == self.encoder_count

    # if we use sampled softmax, we need an output projection
    output_projection = None
    softmax_loss_function = None
    # sampled softmax only makes sense if we sample less than vocabulary size.
    if 0 < num_samples < self.trg_vocab_size:
      with tf.device("/cpu:0"):
        with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=reuse):
          w = tf.get_variable("proj_w", [size, self.trg_vocab_size])
          w_t = tf.transpose(w)
          b = tf.get_variable("proj_b", [self.trg_vocab_size])
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        with tf.device("/cpu:0"):
          labels = tf.reshape(labels, [-1, 1])
          return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, self.trg_vocab_size)
      softmax_loss_function = sampled_loss

    # create the internal multi-layer cell for our RNN
    if use_lstm:
      single_cell = rnn_cell.BasicLSTMCell(size)
    else:
      single_cell = rnn_cell.GRUCell(size)
    cell = single_cell

    # for now, we only apply dropout to the RNN cell inputs
    # TODO: try dropout at the output of the units, inside the attention mechanism, after the projections
    if dropout_rate > 0:
      self.dropout = tf.Variable(1 - dropout_rate, trainable=False, name='dropout_keep_prob')
      self.dropout_off = self.dropout.assign(1.0)
      self.dropout_on = self.dropout.assign(1 - dropout_rate)
      cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=self.dropout)   # Zaremba applies dropout on the input
    else:
      self.dropout = None

    # encoder takes a single cell (it builds the MultiRNNCell by itself)
    encoder_cell = decoder_cell = cell
    if layers > 1:
      decoder_cell = rnn_cell.MultiRNNCell([decoder_cell] * layers)

    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    self.encoder_input_length = []

    self.extensions = list(src_ext) + [trg_ext]
    self.encoder_names = list(src_ext)
    self.decoder_name = trg_ext
    self.embedding_size = embedding_size

    # last bucket is the largest one
    src_bucket_size, trg_bucket_size = buckets[-1]
    for encoder_name, embedding_size_ in zip(self.encoder_names, self.embedding_size):
      encoder_inputs_ = []
      for i in xrange(src_bucket_size):
        placeholder_name = "encoder_{}_{}".format(encoder_name, i)
        if encoder_name in self.binary_input:
          placeholder = tf.placeholder(tf.float32, shape=[None, embedding_size_], name=placeholder_name)
        else:
          placeholder = tf.placeholder(tf.int32, shape=[None], name=placeholder_name)

        encoder_inputs_.append(placeholder)

      self.encoder_inputs.append(encoder_inputs_)
      self.encoder_input_length.append(
        tf.placeholder(tf.int32, shape=[None], name="encoder_{}_length".format(encoder_name))
      )

    for i in xrange(trg_bucket_size + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder_{}_{}".format(self.decoder_name, i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight_{}_{}".format(self.decoder_name, i)))

    # our targets are decoder inputs shifted by one
    targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

    parameters = dict(
      encoder_names=self.encoder_names, decoder_name=self.decoder_name,
      num_encoder_symbols=src_vocab_size, num_decoder_symbols=self.trg_vocab_size,
      embedding_size=self.embedding_size, embeddings=embeddings, layers=layers,
      output_projection=output_projection, bidir=bidir, initial_state_attention=True,
      attention_filters=attention_filters, attention_filter_length=attention_filter_length,
      pooling_ratios=pooling_ratios, attention_window_size=attention_window_size
    )

    # self.attention_states, self.encoder_state = decoders.multi_encoder(
    #   self.encoder_inputs, encoder_input_length=self.encoder_input_length, **parameters
    # )

    self.attention_states, self.encoder_state = decoders.encoder_with_buckets(
      self.encoder_inputs, buckets, reuse=reuse, encoder_input_length=self.encoder_input_length,
      cell=encoder_cell, **parameters
    )

    self.outputs, self.decoder_states, self.attention_weights = decoders.decoder_with_buckets(
      self.attention_states, self.encoder_state, self.decoder_inputs, buckets,
      cell=decoder_cell, reuse=reuse, feed_previous=False, **parameters
    )
    # useful only for greedy decoding (beam size = 1)
    self.greedy_outputs, _, _ = decoders.decoder_with_buckets(
      self.attention_states, self.encoder_state, self.decoder_inputs, buckets,
      cell=decoder_cell, reuse=True, feed_previous=True, **parameters
    )

    self.losses = decoders.loss_with_buckets(self.outputs, targets, self.target_weights,
                                             buckets, softmax_loss_function, reuse)

    # gradients and SGD update operation for training the model
    if freeze_variables is None:
      freeze_variables = []

    variable_names = set([var.name for var in tf.all_variables()])
    assert all(name in variable_names for name in freeze_variables), \
      'you cannot freeze a variable that doesn\'t exist'

    # compute gradient only for variables that are not frozen
    params = [var for var in tf.trainable_variables() if var.name not in freeze_variables]

    self.gradient_norms = []
    self.updates = []
    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
    # opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)  # TODO (doesn't seem to work as well)

    for bucket_loss in self.losses:
      gradients = tf.gradients(bucket_loss, params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
      self.gradient_norms.append(norm)
      self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

    if output_projection is not None:
      w, b = output_projection
      self.greedy_outputs = [
        [tf.matmul(output, w) + b for output in bucket_outputs]
        for bucket_outputs in self.greedy_outputs]
      self.outputs = [
        [tf.matmul(output, w) + b for output in bucket_outputs]
        for bucket_outputs in self.outputs]

    # the beam search decoder only needs the first output, but normalized
    self.beam_search_outputs = [tf.nn.softmax(bucket_outputs[0]) for bucket_outputs in self.outputs]

  def step(self, session, data, bucket_id, forward_only=False):
    if self.dropout is not None:
      session.run(self.dropout_on)

    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs, target_weights, encoder_input_length = self.get_batch(data,
                                                                                          bucket_id)

    # check if the sizes match
    if len(encoder_inputs[0]) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs[0]), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # input feed: encoder inputs, decoder inputs, target_weights
    input_feed = {}

    for i in xrange(self.encoder_count):
      input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
      for l in xrange(encoder_size):
        input_feed[self.encoder_inputs[i][l].name] = encoder_inputs[i][l]

    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # since our targets are decoder inputs shifted by one, we need one more
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros_like(input_feed.values()[0])

    output_feed = [self.losses[bucket_id]]
    if not forward_only:
      output_feed.append(self.updates[bucket_id])

    res = session.run(output_feed, input_feed)
    return res[0]  # losses

  def greedy_decoding(self, session, token_ids):
    if self.dropout is not None:
      session.run(self.dropout_off)

    bucket_id = min(b for b in xrange(len(self.buckets)) if all(self.buckets[b][0] > len(ids_) for ids_ in token_ids))
    encoder_size, decoder_size = self.buckets[bucket_id]
    data = [token_ids + [[]]]
    encoder_inputs, decoder_inputs, target_weights, encoder_input_length = self.get_batch({bucket_id: data},
                                                                                          bucket_id, batch_size=1)

    input_feed = {}
    for i in xrange(self.encoder_count):
      input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
      for l in xrange(encoder_size):
        input_feed[self.encoder_inputs[i][l].name] = encoder_inputs[i][l]

    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # since our targets are decoder inputs shifted by one, we need one more
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros_like(input_feed.values()[0])

    output_feed = self.greedy_outputs[bucket_id][:decoder_size]

    outputs = session.run(output_feed, input_feed)
    return [int(np.argmax(logit, axis=1)) for logit in outputs]  # greedy decoder

  def beam_search_decoding(self, session, token_ids, beam_size, normalize=True, ngrams=None, 
                           weights=None, reverse_vocab=None):
    if not isinstance(session, list):
      session = [session]

    if self.dropout is not None:
      for session_ in session:
        session_.run(self.dropout_off)

    bucket_id = min(b for b in xrange(len(self.buckets)) if all(self.buckets[b][0] > len(ids_) for ids_ in token_ids))
    encoder_size, _ = self.buckets[bucket_id]

    data = [token_ids + [[]]]
    encoder_inputs, decoder_inputs, target_weights, encoder_input_length = self.get_batch({bucket_id: data}, bucket_id,
                                                                                          batch_size=1)
    input_feed = {}
    for i in xrange(self.encoder_count):
      input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
      for l in xrange(encoder_size):
        input_feed[self.encoder_inputs[i][l].name] = encoder_inputs[i][l]

    output_feed = [self.encoder_state[bucket_id]] + self.attention_states[bucket_id]
    res = [session_.run(output_feed, input_feed) for session_ in session]
    state, attention_states = zip(*[(res_[0], res_[1:]) for res_ in res])

    max_len = self.buckets[bucket_id][1]
    attention_weights = [[np.zeros([1, encoder_size]) for _ in self.encoder_names] for _ in session]
    decoder_input = decoder_inputs[0]  # GO symbol

    finished_hypotheses = []
    finished_scores = []

    hypotheses = [[]]
    scores = np.zeros([1], dtype=np.float32)

    for _ in range(max_len):
      # each session/model has its own input and output
      input_feed = [{
          self.encoder_state[bucket_id]: state_,
          self.decoder_inputs[0]: decoder_input  # in beam-search decoder, we only feed the first input
        } for state_ in state]

      for input_feed_, attention_states_, attention_weights_ in zip(input_feed, attention_states, attention_weights):
        for i in range(self.encoder_count):
          input_feed_[self.attention_states[bucket_id][i]] = attention_states_[i]
          input_feed_[self.attention_weights[bucket_id][0][i]] = attention_weights_[i]

      output_feed = [self.beam_search_outputs[bucket_id],
                     self.decoder_states[bucket_id][0]] + self.attention_weights[bucket_id][1]

      res = [session_.run(output_feed, input_feed_) for session_, input_feed_ in zip(session, input_feed)]
      decoder_output, decoder_state, attention_weights = zip(*[(res_[0], res_[1], res_[2:]) for res_ in res])

      # hypotheses, list of tokens ids of shape (beam_size, previous_len)
      # decoder_output, shape=(beam_size, trg_vocab_size)
      # decoder_state, shape=(beam_size, cell.state_size)
      # attention_weights, shape=(beam_size, max_len)

      if ngrams is not None:
        lm_score = []
        lm_order = len(ngrams)

        for hypothesis in hypotheses:
          hypothesis = [utils.BOS_ID] + hypothesis   # not sure about this (should we put <s> at the beginning?)
          history = hypothesis[1 - lm_order:]
          score_ = []

          for token_id in range(self.trg_vocab_size):
            # if token is not in unigrams, this means that either there is something
            # wrong with the ngrams (e.g. trained on wrong file),
            # or trg_vocab_size is larger than actual vocabulary
            if (token_id,) not in ngrams[0]:
              prob = float('-inf')
            elif token_id == utils.BOS_ID:
              prob = float('-inf')
            else:
              prob = utils.estimate_lm_probability(history + [token_id], ngrams)
            score_.append(prob)

          lm_score.append(score_)
        lm_score = np.array(lm_score, dtype=np.float32)
      else:
        lm_score = np.zeros((1, self.trg_vocab_size))

      # default LM weight: 0.4
      weights = self.model_weights
      if weights is None and ngrams is not None:
        weights = [0.6 / len(session)] * len(session) + [0.4]

      scores_ = scores[:, None] - np.average([np.log(decoder_output_) for decoder_output_ in decoder_output] +
                                             [lm_score],
                                             axis=0, weights=weights)
      scores_ = scores_.flatten()
      flat_ids = np.argsort(scores_)[:beam_size]

      token_ids_ = flat_ids % self.trg_vocab_size
      hyp_ids = flat_ids // self.trg_vocab_size

      new_hypotheses = []
      new_scores = []
      new_state = [[] for _ in session]
      new_input = []

      for flat_id, hyp_id, token_id in zip(flat_ids, hyp_ids, token_ids_):
        hypothesis = hypotheses[hyp_id] + [token_id]
        score = scores_[flat_id]

        # for debugging purposes
        if reverse_vocab:
          hyp_str = ' '.join(reverse_vocab[id_] if 0 < id_ < len(reverse_vocab) else utils._UNK for id_ in hypothesis)

        if token_id == utils.EOS_ID:
          # early stop: hypothesis is finished, it is thus unnecessary to keep expanding it
          beam_size -= 1  # number of possible hypotheses is reduced by one
          finished_hypotheses.append(hypothesis)
          finished_scores.append(score)
        else:
          new_hypotheses.append(hypothesis)
          for i, decoder_state_ in enumerate(decoder_state):
            new_state[i].append(decoder_state_[hyp_id])
          new_scores.append(score)
          new_input.append(token_id)

      hypotheses = new_hypotheses
      state = [np.array(new_state_) for new_state_ in new_state]
      scores = np.array(new_scores)
      decoder_input = np.array(new_input, dtype=np.int32)

      if beam_size <= 0:
        break

    hypotheses += finished_hypotheses
    scores = np.concatenate([scores, finished_scores])

    if normalize:  # normalize score by length (to encourage longer sentences)
      scores /= map(len, hypotheses)

    # sort best-list by score
    sorted_idx = np.argsort(scores)
    hypotheses = np.array(hypotheses)[sorted_idx].tolist()
    scores = scores[sorted_idx].tolist()
    return hypotheses, scores

  def get_batch(self, data, bucket_id, batch_size=None):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of tuples of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.
      batch_size: if specified, use this batch size instead of the training
        batch size (useful for decoding).

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    decoder_inputs = []
    batch_size = batch_size or self.batch_size

    encoder_inputs = [[] for _ in range(self.encoder_count)]
    encoder_input_length = [[] for _ in range(self.encoder_count)]
    batch_encoder_inputs = [[] for _ in range(self.encoder_count)]

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(batch_size):
      # Get a random tuple of sentences in this bucket.
      sentences = random.choice(data[bucket_id])
      src_sentences = sentences[0:-1]
      trg_sentence = sentences[-1] + [utils.EOS_ID]

      for i, src_sentence in enumerate(src_sentences):
        if isinstance(src_sentence[0], np.ndarray):
          pad = np.zeros([self.embedding_size], dtype=np.float32)
        else:
          pad = utils.PAD_ID

        encoder_pad = [pad] * (encoder_size - len(src_sentence))
        # reverse THEN pad (better for early stopping...)
        # reversed_sentence = list(reversed(src_sentence)) + encoder_pad
        reversed_sentence = list(reversed(src_sentence + encoder_pad))

        encoder_inputs[i].append(reversed_sentence)
        encoder_input_length[i].append(len(src_sentence))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(trg_sentence) - 1
      decoder_inputs.append([utils.BOS_ID] + trg_sentence +
                            [utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_decoder_inputs, batch_weights = [], []

    encoder_input_length = [np.array(input_length_, dtype=np.int32) for input_length_ in encoder_input_length]
    for i in range(self.encoder_count):
      for length_idx in xrange(encoder_size):
        # import pdb; pdb.set_trace()
        batch_encoder_inputs[i].append(
          np.array([encoder_inputs[i][batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(batch_size, dtype=np.float32)
      for batch_idx in xrange(batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights, encoder_input_length

  def assign_data_set(self, train_set):
    self.train_set = train_set

    train_bucket_sizes = map(len, self.train_set)
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    self.train_bucket_scales = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                                for i in xrange(len(train_bucket_sizes))]
