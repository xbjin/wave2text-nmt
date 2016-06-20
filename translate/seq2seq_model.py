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
               src_vocab_size, trg_vocab_size, size, num_layers, max_gradient_norm, batch_size, use_lstm=True,
               num_samples=512, reuse=None, dropout_rate=0.0, embedding_size=None, **kwargs):
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
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
    """
    self.buckets = buckets
    self.batch_size = batch_size
    self.encoder_count = len(src_ext)

    self.learning_rate = learning_rate
    self.global_step = global_step

    trg_ext = trg_ext[0]   # FIXME: for now we assume we have only one decoder
    self.trg_vocab_size = trg_vocab_size[0]

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
    single_cell = rnn_cell.GRUCell(size)
    if use_lstm:
      single_cell = rnn_cell.BasicLSTMCell(size)
    cell = single_cell

    if dropout_rate > 0:   # TODO: check that this works
      # It seems like this does RNN dropout (Zaremba et al., 2015), i.e., no
      # dropout on the recurrent connections (see models/rnn/ptb/ptb_word_lm.py)
      keep_prob = 1 - dropout_rate
      cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    # TODO: how about dropout elsewhere (inputs and attention mechanism)?

    if num_layers > 1:
      cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)

    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []

    self.encoder_names = list(src_ext)
    self.decoder_name = trg_ext
    self.embedding_size = embedding_size if embedding_size is not None else size

    # last bucket is the largest one
    src_bucket_size, trg_bucket_size = buckets[-1]
    for encoder_name in self.encoder_names:
      encoder_inputs_ = []
      for i in xrange(src_bucket_size):
        encoder_inputs_.append(tf.placeholder(tf.int32, shape=[None], name="encoder_{}_{}".format(encoder_name, i)))

      self.encoder_inputs.append(encoder_inputs_)

    for i in xrange(trg_bucket_size + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder_{}_{}".format(self.decoder_name, i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight_{}_{}".format(self.decoder_name, i)))

    # our targets are decoder inputs shifted by one
    targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

    # training outputs and losses
    self.outputs, self.losses = decoders.model_with_buckets(
      self.encoder_inputs, self.decoder_inputs, targets,
      self.target_weights, buckets, softmax_loss_function=softmax_loss_function,
      encoder_names=self.encoder_names, decoder_name=self.decoder_name,
      cell=cell, num_encoder_symbols=src_vocab_size, num_decoder_symbols=self.trg_vocab_size,
      embedding_size=self.embedding_size, embeddings=embeddings,
      output_projection=output_projection,
      initial_state_attention=True, feed_previous=False, reuse=reuse
    )

    # useful only for greedy decoding (beam size = 1)
    self.greedy_outputs, _ = decoders.model_with_buckets(
      self.encoder_inputs, self.decoder_inputs, targets,
      self.target_weights, buckets, softmax_loss_function=softmax_loss_function,
      encoder_names=self.encoder_names, decoder_name=self.decoder_name,
      cell=cell, num_encoder_symbols=src_vocab_size, num_decoder_symbols=self.trg_vocab_size,
      embedding_size=self.embedding_size, embeddings=embeddings,
      output_projection=output_projection,
      initial_state_attention=True, feed_previous=True, reuse=True
    )
    if output_projection is not None:
      self.greedy_outputs = [
        [tf.matmul(output, output_projection[0]) + output_projection[1] for output in bucket_outputs]
        for bucket_outputs in self.greedy_outputs
      ]

    # gradients and SGD update operation for training the model
    params = tf.trainable_variables()

    self.gradient_norms = []
    self.updates = []
    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
    for bucket_loss in self.losses:
      gradients = tf.gradients(bucket_loss, params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
      self.gradient_norms.append(norm)
      self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

    # TODO: this is already computed
    self.attention_states, self.encoder_state = decoders.multi_encoder(
      self.encoder_inputs, self.encoder_names, cell,
      src_vocab_size, self.embedding_size,
      embeddings=embeddings, reuse=True)

    self.initial_state_placeholder = tf.placeholder(tf.float32, shape=[None, cell.state_size],
                                                    name='initial_state')
    # FIXME: variable length
    self.attention_states_placeholder = [tf.placeholder(tf.float32, shape=[None, 51, cell.output_size],
                                         name='attention_states')]
    self.attention_weights_placeholder = [tf.placeholder(tf.float32, shape=[None, 51], name='attention_weights')]

    self.decoder_output, self.decoder_state, self.attention_weights = decoders.attention_decoder(
      decoder_inputs=self.decoder_inputs[:1],
      initial_state=self.initial_state_placeholder,
      attention_states=self.attention_states_placeholder,
      encoder_names=self.encoder_names,
      decoder_name=self.decoder_name,
      cell=cell,
      num_decoder_symbols=self.trg_vocab_size,
      embedding_size=self.embedding_size,
      attention_weights=self.attention_weights_placeholder,
      feed_previous=False,
      output_projection=output_projection,
      embeddings=embeddings,
      initial_state_attention=True,
      reuse=True)

    self.decoder_output = self.decoder_output[0]
    self.attention_weights = self.attention_weights[0]

    if output_projection is not None:
      w, b = output_projection
      self.decoder_output = tf.nn.softmax(tf.matmul(self.decoder_output, w) + b)

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only=False):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of lists of numpy int vectors to feed as input to
      each encoder.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: disable backward step (no parameter update).
      decode: decoding mode. Feed its own output to the decoder
        (feed_previous set to True) and perform output projection if
        sampled softmax is on.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
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

    outputs = session.run(output_feed, input_feed)
    return outputs[0]  # losses

  def greedy_decoding(self, session, token_ids):
    bucket_id = min(b for b in xrange(len(self.buckets)) if all(self.buckets[b][0] > len(ids_) for ids_ in token_ids))
    # bucket_id = len(self.buckets) - 1
    data = [token_ids + [[]]]
    encoder_inputs, decoder_inputs, target_weights = self.get_batch({bucket_id: data}, bucket_id, batch_size=1)

    encoder_size, decoder_size = self.buckets[bucket_id]

    input_feed = {}
    for i in xrange(self.encoder_count):
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

  def beam_search_decoding(self, session, token_ids, beam_size, normalize=True):
    # TODO: variable sequence length
    # TODO: handle multiple encoders
    # TODO: check this initial_state_attention parameter (might cause the first word generated to be bad)
    # TODO: test with initial_state_attention=True with previous code version
    bucket_id = min(b for b in xrange(len(self.buckets)) if all(self.buckets[b][0] > len(ids_) for ids_ in token_ids))
    # bucket_id = len(self.buckets) - 1
    data = [token_ids + [[]]]
    encoder_inputs, decoder_inputs, target_weights = self.get_batch({bucket_id: data}, bucket_id, batch_size=1)
    encoder_size, decoder_size = self.buckets[bucket_id]
    input_feed = {}
    for i in xrange(self.encoder_count):
      for l in xrange(encoder_size):
        input_feed[self.encoder_inputs[i][l].name] = encoder_inputs[i][l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    output_feed = [self.encoder_state, self.attention_states[0]]
    state, attention_states = session.run(output_feed, input_feed)

    max_len = self.buckets[bucket_id][1]
    attention_weights = np.zeros([1, encoder_size])
    decoder_input = decoder_inputs[0]

    finished_hypotheses = []
    finished_scores = []

    hypotheses = [[]]
    scores = np.zeros([1], dtype=np.float32)

    for _ in range(max_len):
      input_feed = {
        self.attention_states_placeholder[0]: attention_states,
        self.attention_weights_placeholder[0]: attention_weights,   # shape=(beam_size, max_len)
        self.initial_state_placeholder: state,   # shape=(beam_size, cell.state_size)
        self.decoder_inputs[0]: decoder_input    # shape=(beam_size)
      }
      output_feed = [self.decoder_output, self.decoder_state, self.attention_weights[0]]

      # FIXME: decoder_output are not probabilities
      decoder_output, decoder_state, attention_weights = session.run(output_feed, input_feed)
      # decoder_output, shape=(beam_size, trg_vocab_size)
      # decoder_state, shape=(beam_size, cell.state_size)
      # attention_weights, shape=(beam_size, max_len)
      scores_ = scores[:, None] - np.log(decoder_output)
      scores_ = scores_.flatten()
      flat_ids = np.argsort(scores_)[:beam_size]

      token_ids_ = flat_ids % self.trg_vocab_size
      hyp_ids = flat_ids // self.trg_vocab_size

      new_hypotheses = []
      new_scores = []
      new_state = []
      new_input = []

      for flat_id, hyp_id, token_id in zip(flat_ids, hyp_ids, token_ids_):
        hypothesis = hypotheses[hyp_id] + [token_id]
        score = scores_[flat_id]

        if token_id == utils.EOS_ID:
          beam_size -= 1
          finished_hypotheses.append(hypothesis)
          finished_scores.append(score)
        else:
          new_hypotheses.append(hypothesis)
          new_state.append(decoder_state[hyp_id])
          new_scores.append(score)
          new_input.append(token_id)

      hypotheses = new_hypotheses
      state = np.array(new_state)
      scores = np.array(new_scores)
      decoder_input = np.array(new_input, dtype=np.int32)

      if beam_size <= 0:
        break

    hypotheses += finished_hypotheses
    scores = np.concatenate([scores, finished_scores])

    if normalize:  # normalize score by length
      scores /= map(len, hypotheses)

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
    batch_encoder_inputs = [[] for _ in range(self.encoder_count)]

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(batch_size):
      # Get a random tuple of sentences in this bucket.
      sentences = random.choice(data[bucket_id])
      src_sentences = sentences[0:-1]
      trg_sentence = sentences[-1]

      # Encoder inputs are padded and then reversed.
      for i, src_sentence in enumerate(src_sentences):
          encoder_pad = [utils.PAD_ID] * (encoder_size - len(src_sentence))
          reversed_sentence = list(reversed(src_sentence + encoder_pad))
          encoder_inputs[i].append(reversed_sentence)

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(trg_sentence) - 1
      decoder_inputs.append([utils.GO_ID] + trg_sentence +
                            [utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_decoder_inputs, batch_weights = [], []

    for i in range(self.encoder_count):
      for length_idx in xrange(encoder_size):
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

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights


  def read_data(self, train_set, buckets, max_train_size=None):
    src_train_ids, trg_train_ids = train_set
    self.train_set = utils.read_dataset(src_train_ids, trg_train_ids, buckets, max_train_size)

    train_bucket_sizes = map(len, self.train_set)
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    self.train_bucket_scales = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                                for i in xrange(len(train_bucket_sizes))]

  def get_embeddings(self, sess):
    """
    Returns a dict of embedding matrices for each extension
    """

    with variable_scope.variable_scope('many2one_rnn_seq2seq',
                                       reuse=True):
      shared_embeddings = self.shared_embeddings or []
      embeddings = {}
      names = self.encoder_names + [self.decoder_name]

      for name in names:
        part = 'decoder' if name == self.decoder_name else 'encoder'
        scope = 'shared_embeddings' if name in shared_embeddings else '{}_{}'.format(part, name)
        variable = tf.get_variable('{}/embedding'.format(scope))
        embeddings[name] = variable.eval(session=sess)

      return embeddings
