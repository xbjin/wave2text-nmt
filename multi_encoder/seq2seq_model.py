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

from multi_encoder import rnn_cell
from multi_encoder import many2one
from multi_encoder import data_utils

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

  def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm=True,
               num_samples=512, forward_only=False, encoder_count=1,
               reuse=None, encoder_num=None, model_name=None,
               embedding=None, dropout_rate=0):
    """Create the model.

    Args:
      source_vocab_size: size of the sources vocabularies.
      target_vocab_size: size of the target vocabulary.
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
      forward_only: if set, we do not construct the backward pass in the model.
      encoder_count: number of encoders to create
      encoder_num: list of encoders ids to put in the model
    """
    # TODO : changer le model_name en encoder_num? (c est aussi unique)
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.model_name = model_name

    with variable_scope.variable_scope(model_name or
         variable_scope.get_variable_scope(), reuse=reuse):
      # if `model_name` is specified, those parameters are specific to the
      # model.
      self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                                       name='learning_rate')
      with tf.device("/cpu:0"):  # cannot put ints on GPU
        self.global_step = tf.Variable(0, trainable=False,
                                       name='global_step')

    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.encoder_count = encoder_count

    self.forward_only = tf.Variable(forward_only, trainable=False,
                                    name='forward_only')
    self.train_op = self.forward_only.assign(False)
    self.decode_op = self.forward_only.assign(True)
    
    # TODO: For now, we assume that all source languages
    # have the same vocabulary size.
    source_vocab_sizes = [source_vocab_size for _ in range(self.encoder_count)]

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if 0 < num_samples < self.target_vocab_size:
      with tf.device("/cpu:0"):
        with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                           reuse=reuse):
          w = tf.get_variable("proj_w", [size, self.target_vocab_size])
          w_t = tf.transpose(w)
          b = tf.get_variable("proj_b", [self.target_vocab_size])
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        with tf.device("/cpu:0"):
          labels = tf.reshape(labels, [-1, 1])
          return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                            self.target_vocab_size)
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

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      return many2one.many2one_rnn_seq2seq(
          encoder_inputs, decoder_inputs, cell, source_vocab_sizes,
          target_vocab_size, output_projection=output_projection,
          feed_previous=do_decode, encoder_num=encoder_num, embedding=embedding)

    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []

    # Last bucket is the biggest one.
    src_bucket_size, trg_bucket_size = buckets[-1]
    for k in range(self.encoder_count):
      encoder_inputs_ = []
      start_index = k * src_bucket_size

      for i in xrange(src_bucket_size):
        encoder_inputs_.append(tf.placeholder(tf.int32, shape=[None],
                               name="encoder{0}".format(i + start_index)))

      self.encoder_inputs.append(encoder_inputs_)

    for i in xrange(trg_bucket_size + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    # Training outputs and losses.
    self.outputs, self.losses = many2one.model_with_buckets(
      self.encoder_inputs, self.decoder_inputs, targets,
      self.target_weights, buckets,
      lambda x, y: seq2seq_f(x, y, self.forward_only),
      softmax_loss_function=softmax_loss_function, reuse=reuse)

    if output_projection is not None:
      self.decode_outputs = [
        [tf.matmul(output, output_projection[0]) + output_projection[1]
         for output in self.outputs[b]]
        for b in range(len(buckets))]
    else:
      self.decode_outputs = self.outputs

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()

    self.gradient_norms = []
    self.updates = []
    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
    for b in xrange(len(buckets)):
      gradients = tf.gradients(self.losses[b], params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                       max_gradient_norm)
      self.gradient_norms.append(norm)
      self.updates.append(opt.apply_gradients(
        zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.all_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of lists of numpy int vectors to feed as input to
      each encoder.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

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

    if forward_only:    # change value of forward_only tensor
      session.run(self.decode_op)
    else:
      session.run(self.train_op)

    input_feed = {}

    for i in xrange(self.encoder_count):
      for l in xrange(encoder_size):
        input_feed[self.encoder_inputs[i][l].name] = encoder_inputs[i][l]

    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      # variable
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        # output_feed.append(self.outputs[bucket_id][l])
        output_feed.append(self.decode_outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)

    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of tuples of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    decoder_inputs = []

    encoder_inputs = [[] for _ in range(self.encoder_count)]
    batch_encoder_inputs = [[] for _ in range(self.encoder_count)]

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      # Get a random tuple of sentences in this bucket.
      sentences = random.choice(data[bucket_id])
      src_sentences = sentences[0:-1]
      trg_sentence = sentences[-1]

      # Encoder inputs are padded and then reversed.
      for i, src_sentence in enumerate(src_sentences):
          encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(src_sentence))
          reversed_sentence = list(reversed(src_sentence + encoder_pad))
          encoder_inputs[i].append(reversed_sentence)

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(trg_sentence) - 1
      decoder_inputs.append([data_utils.GO_ID] + trg_sentence +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_decoder_inputs, batch_weights = [], []

    for i in range(self.encoder_count):
      for length_idx in xrange(encoder_size):
        batch_encoder_inputs[i].append(
          np.array([encoder_inputs[i][batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights
