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

from translate import rnn_cell
from translate import seq2seq
from translate import utils

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

    self.feed_previous = tf.Variable(False, trainable=False, name='feed_previous')
    self.train_op = self.feed_previous.assign(False)
    self.decode_op = self.feed_previous.assign(True)

    #beam
    self.hidden_states = []
    self.decoder_initial_states = []
    self.attention_states = []


    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
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

    # Create the internal multi-layer cell for our RNN.
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

    def seq2seq_function(encoder_inputs, decoder_inputs):
      return seq2seq.many2one_rnn_seq2seq(encoder_inputs, decoder_inputs, self.encoder_names, self.decoder_name,
                                          cell, src_vocab_size, self.trg_vocab_size, self.embedding_size, embeddings,
                                          output_projection=output_projection, feed_previous=self.feed_previous)

    # training outputs and losses
    self.hidden_states, self.decoder_initial_states, self.attention_states, \
    self.outputs, self.losses = seq2seq.model_with_buckets(
      self.encoder_inputs, self.decoder_inputs, targets,
      self.target_weights, buckets, seq2seq_function,
      softmax_loss_function=softmax_loss_function, reuse=reuse)

    if output_projection is not None:
      self.decode_outputs = [
        [tf.matmul(output, output_projection[0]) + output_projection[1] for output in bucket_outputs]
        for bucket_outputs in self.outputs]
    else:
      self.decode_outputs = self.outputs

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

  """
  def beam_step(self, session, encoder_inputs, decoder_inputs, target_weights,
                bucket_id, beam_size=10, forward_only=False, decode=False, max_len=None,
                normalize=True):
      
    #en premier faire juste le pas de l'encodeur 
    for i in xrange(self.encoder_count):
      for l in xrange(encoder_size):
        input_feed[self.encoder_inputs[i][l].name] = encoder_inputs[i][l]      
      
    # we select the last element of hidden_states to keep as it is a list of hidden_states
    encoder_output_feed = [self.hidden_states[-1], self.decoder_initial_states, self.attention_states]

    ret = session.run(encoder_output_feed, encoder_input_feed)

    # here we get info to the decode step
    attention_states = ret[2]
    shape = ret[1][0].shape
    # decoder_init = numpy.tile(ret[1][0].reshape(1, shape[0]), (12, 1))
    decoder_init = ret[1][0].reshape(1, shape[0])
    decoder_states = np.zeros((1, 1, 1, self.decoder_size))
    
    #maintenant on commencer le decoder step by step

    sample = []
    sample_score = []

    live_hyp = 1
    dead_hyp = 0
    
    hyp_samples = [[]] * live_hyp
    hyp_scores = np.zeros(live_hyp).astype('float32')


    # we must retrieve the last state to feed the decoder run
    # @alex a voir a quoi ce que self.decoder_states dans  nmt_models.py, de la fonction decode(), 
    #g pas eu le temps de la trouver
    decoder_output_feed = [self.output, self.states, self.decoder_states]

    for ii in xrange(max_len):

        session.run(self.step_num.assign(ii + 2))
        session.run(self.step_num.assign(ii + 2))

        # we must feed decoder_initial_state and attention_states to run one decode step
        decoder_input_feed = {self.decoder_inputs[0].name: decoder_inputs,
                                  self.decoder_init_plcholder.name: decoder_init,
                                  self.attn_plcholder.name: attention_states}
        if self.decoder_attention_f:
            # if ii == 1:
            #     decoder_states = numpy.tile(decoder_states, (12, 1, 1, 1))
            decoder_input_feed[self.decoder_states_holders.name] = decoder_states

            # print "Step %d - States shape %s - Input shape %s" % (ii, decoder_states.shape, decoder_inputs.shape)

        ret = session.run(decoder_output_feed, decoder_input_feed)

        next_p = ret[0]
        next_state = ret[1]
        decoder_states = ret[2]

        cand_scores = hyp_scores[:, None] - np.log(next_p)
        cand_flat = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[:(beam_size - dead_hyp)]

        voc_size = next_p.shape[1]
        trans_indices = ranks_flat / voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        new_hyp_samples = []
        new_hyp_scores = np.zeros(beam_size - dead_hyp).astype('float32')
        new_hyp_states = []
        new_dec_states = []

        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti] + [wi])
            new_hyp_scores[idx] = copy.copy(costs[ti])
            new_hyp_states.append(copy.copy(next_state[ti]))
            new_dec_states.append(copy.copy(decoder_states[ti]))

            # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = []
        dec_states = []

        for idx in xrange(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == data_utils.EOS_ID:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_hyp += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states.append(new_hyp_states[idx])
                dec_states.append(new_dec_states[idx])

        dec_states = [d.reshape(1, d.shape[0], d.shape[1], d.shape[2]) for d in dec_states]
        dec_states = np.concatenate(dec_states, axis=0)

        hyp_scores = np.array(hyp_scores)
        live_hyp = new_live_k

        if new_live_k < 1:
            break
        if dead_hyp >= beam_size:
            break

        decoder_inputs = np.array([w[-1] for w in hyp_samples])
        decoder_init = np.array(hyp_states)
        decoder_states = dec_states

        # dump every remaining one
    if dump_remaining:
        if live_hyp > 0:
            for idx in xrange(live_hyp):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

        # normalize scores according to sequence lengths
    if normalize:
        lengths = np.array([len(s) for s in sample])
        sample_score = sample_score / lengths

    # sort the samples by score (it is in log-scale, therefore lower is better)
    sidx = np.argsort(sample_score)
    sample = np.array(sample)[sidx]
    sample_score = np.array(sample_score)[sidx]

    return sample.tolist(), sample_score.tolist()            
  """
    
        
  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only=False, decode=False):
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

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.

    if decode:
      forward_only = True  # decode => forward_only
      session.run(self.decode_op)  # toggles feed_previous
      outputs_ = self.decode_outputs   # output projection
    else:
      session.run(self.train_op)
      outputs_ = self.outputs

    input_feed = {}

    for i in xrange(self.encoder_count):
      for l in xrange(encoder_size):
        input_feed[self.encoder_inputs[i][l].name] = encoder_inputs[i][l]

    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros_like(input_feed.values()[0])

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      # variable
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(outputs_[bucket_id][l])

    outputs = session.run(output_feed, input_feed)

    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

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
