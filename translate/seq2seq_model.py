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

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import re

from translate import utils
from translate import decoders
from collections import namedtuple

from tensorflow.python.ops import variable_scope


class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention.

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

  def __init__(self, encoders, decoder, learning_rate, global_step, max_gradient_norm,
               num_samples=512, dropout_rate=0.0, freeze_variables=None, lm_weight=None,
               max_output_len=50, attention=True, buckets=None, feed_previous=0.0,
               optimizer='sgd', max_input_len=None, decode_only=False,
               initial_state_attention=True, len_normalization=1.0,
               residual_connections=False, **kwargs):
    self.lm_weight = lm_weight
    self.encoders = encoders
    self.decoder = decoder

    self.learning_rate = learning_rate
    self.global_step = global_step

    self.encoder_count = len(encoders)
    self.trg_vocab_size = decoder.vocab_size
    self.trg_cell_size = decoder.cell_size
    self.binary_input = [encoder.name for encoder in encoders if encoder.binary]

    self.max_output_len = max_output_len
    self.max_input_len = max_input_len
    self.buckets = buckets
    self.len_normalization = len_normalization

    # if we use sampled softmax, we need an output projection
    # sampled softmax only makes sense if we sample less than vocabulary size
    if num_samples == 0 or num_samples > self.trg_vocab_size:
      output_projection = None
      softmax_loss_function = None
    else:
      with tf.device("/cpu:0"):
        with variable_scope.variable_scope('decoder_{}'.format(decoder.name)):
          w = decoders.get_variable_unsafe("proj_w", [self.trg_cell_size, self.trg_vocab_size])
          w_t = tf.transpose(w)
          b = decoders.get_variable_unsafe("proj_b", [self.trg_vocab_size])
        output_projection = (w, b)

      def sampled_loss(inputs, labels):
        with tf.device("/cpu:0"):
          labels = tf.reshape(labels, [-1, 1])
          return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, self.trg_vocab_size)

      softmax_loss_function = sampled_loss

    if dropout_rate > 0:
      self.dropout = tf.Variable(1 - dropout_rate, trainable=False, name='dropout_keep_prob')
      self.dropout_off = self.dropout.assign(1.0)
      self.dropout_on = self.dropout.assign(1 - dropout_rate)
    else:
      self.dropout = None

    self.feed_previous = tf.constant(feed_previous, dtype=tf.float32)

    self.encoder_inputs = []
    self.encoder_input_length = []

    self.extensions = [encoder.name for encoder in encoders] + [decoder.name]
    self.encoder_names = [encoder.name for encoder in encoders]
    self.decoder_name = decoder.name
    self.extensions = self.encoder_names + [self.decoder_name]

    for encoder in self.encoders:
      if encoder.binary:
        placeholder = tf.placeholder(tf.float32, shape=[None, None, encoder.embedding_size],
                                     name="encoder_{}".format(encoder.name))
      else:
        placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                     name="encoder_{}".format(encoder.name))

      self.encoder_inputs.append(placeholder)
      self.encoder_input_length.append(
        tf.placeholder(tf.int64, shape=[None], name="encoder_{}_length".format(encoder.name))
      )

    self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, None],
                                         name="decoder_{}".format(self.decoder.name))
    self.decoder_input = tf.placeholder(tf.int32, shape=[None], name="beam_search_{}".format(decoder.name))
    self.target_weights = tf.placeholder(tf.float32, shape=[None, None],
                                         name="weight_{}".format(self.decoder.name))
    self.targets = tf.placeholder(tf.int32, shape=[None, None],
                                  name="target_{}".format(self.decoder.name))

    parameters = dict(
      encoders=encoders, decoder=decoder,
      dropout=self.dropout, output_projection=output_projection,
      initial_state_attention=initial_state_attention,
      residual_connections=residual_connections
    )

    self.attention_states, self.encoder_state = decoders.multi_encoder(
      self.encoder_inputs, encoder_input_length=self.encoder_input_length,
      **parameters
    )

    decoder = decoders.attention_decoder if attention else decoders.decoder

    self.outputs, self.decoder_states, self.attention_weights = decoder(
      attention_states=self.attention_states, initial_state=self.encoder_state,
      decoder_inputs=self.decoder_inputs, feed_previous=self.feed_previous, **parameters
    )

    self.beam_output, self.beam_tensors = decoders.beam_search_decoder(
      decoder_input=self.decoder_input, attention_states=self.attention_states,
      initial_state=self.encoder_state, **parameters)

    self.loss = decoders.sequence_loss(logits=self.outputs, targets=self.targets,
                                       weights=self.target_weights,
                                       softmax_loss_function=softmax_loss_function)

    if not decode_only:
      # gradients and SGD update operation for training the model
      if freeze_variables is None:
        freeze_variables = []

      # compute gradient only for variables that are not frozen
      frozen_parameters = [var.name for var in tf.trainable_variables()
                           if any(re.match(var_, var.name) for var_ in freeze_variables)]
      if frozen_parameters:
        utils.debug('frozen parameters: {}'.format(', '.join(frozen_parameters)))
      params = [var for var in tf.trainable_variables() if var.name not in frozen_parameters]

      if optimizer.lower() == 'adadelta':
        opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
      elif optimizer.lower() == 'adagrad':
        opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)
      elif optimizer.lower() == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
      else:
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

      gradients = tf.gradients(self.loss, params)
      clipped_gradients, self.gradient_norms = tf.clip_by_global_norm(gradients, max_gradient_norm)
      self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def tensor_prod(x, w, b):
      shape = tf.shape(x)
      x = tf.reshape(x, tf.pack([tf.mul(shape[0], shape[1]), shape[2]]))
      x = tf.matmul(x, w) + b
      x = tf.reshape(x, tf.pack([shape[0], shape[1], b.get_shape()[0]]))
      return x

    if output_projection is not None:
      w, b = output_projection
      self.outputs = tensor_prod(self.outputs, w, b)
      self.beam_output = tf.nn.xw_plus_b(self.beam_output, w, b)

    self.beam_output = tf.nn.softmax(self.beam_output)

  def step(self, session, data, forward_only=False, align=False):
    if self.dropout is not None:
      session.run(self.dropout_on)

    encoder_inputs, decoder_inputs, targets, target_weights, encoder_input_length = self.get_batch(data)
    tf.get_variable_scope().reuse_variables()

    input_feed = {}
    for i in xrange(self.encoder_count):
      input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
      input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

    input_feed[self.target_weights] = target_weights
    input_feed[self.decoder_inputs] = decoder_inputs
    input_feed[self.targets] = targets

    output_feed = {'loss': self.loss}
    if not forward_only:
      output_feed['updates'] = self.updates
    if align:
      output_feed['attn_weights'] = self.attention_weights

    res = session.run(output_feed, input_feed)
    return res['loss'], res.get('attn_weights')

  def greedy_decoding(self, session, token_ids):
    if self.dropout is not None:
      session.run(self.dropout_off)

    encoder_inputs, decoder_inputs, targets, target_weights, encoder_input_length = self.get_batch(
      [token_ids + [[]]], decoding=True)

    input_feed = {}
    for i in xrange(self.encoder_count):
      input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
      input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

    input_feed[self.target_weights] = target_weights
    input_feed[self.decoder_inputs] = decoder_inputs
    input_feed[self.targets] = targets
    input_feed[self.feed_previous] = 1.0

    outputs, attn_weights = session.run([self.outputs, self.attention_weights], input_feed)
    return [int(np.argmax(logit, axis=1)) for logit in outputs], attn_weights  # greedy decoder

  def beam_search_decoding(self, session, token_ids, beam_size, ngrams=None, weights=None, reverse_vocab=None):
    if not isinstance(session, list):
      session = [session]

    if self.dropout is not None:
      for session_ in session:
        session_.run(self.dropout_off)

    data = [token_ids + [[]]]
    encoder_inputs, decoder_inputs, targets, target_weights, encoder_input_length = self.get_batch(data, decoding=True)
    input_feed = {}
    for i in xrange(self.encoder_count):
      input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
      input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

    output_feed = [self.encoder_state] + self.attention_states
    res = [session_.run(output_feed, input_feed) for session_ in session]
    state, attn_states = zip(*[(res_[0], res_[1:]) for res_ in res])

    attns = [None for _ in session]
    attn_weights = [None for _ in session]

    decoder_input = decoder_inputs[0]  # GO symbol

    finished_hypotheses = []
    finished_scores = []

    hypotheses = [[]]
    scores = np.zeros([1], dtype=np.float32)

    # for initial state projection
    state = [session_.run(self.beam_tensors.state, {self.encoder_state: state_})
             for session_, state_ in zip(session, state)]

    for _ in range(self.max_output_len):
      # each session/model has its own input and output
      input_feed = [{
          self.beam_tensors.state: state_,
          self.decoder_input: decoder_input  # in beam-search decoder, we only feed the first input
        } for state_ in state]

      batch_size = decoder_input.shape[0]

      for input_feed_, attn_states_, attns_, attn_weights_ in zip(input_feed, attn_states, attns, attn_weights):
        for i in range(self.encoder_count):
          input_feed_[self.attention_states[i]] = attn_states_[i].repeat(batch_size, axis=0)
          if attn_weights_ is not None:
            input_feed_[self.beam_tensors.attn_weights[i]] = attn_weights_[i]

        if attns_ is not None:
          input_feed_[self.beam_tensors.attns] = attns_

      output_feed = namedtuple('beam_output', 'decoder_output decoder_state attns attn_weights')(
        self.beam_output,
        self.beam_tensors.new_state,
        self.beam_tensors.new_attns,
        self.beam_tensors.new_attn_weights
      )

      res = [session_.run(output_feed, input_feed_) for session_, input_feed_ in zip(session, input_feed)]
      decoder_output, decoder_state, attns, attn_weights = zip(*[(res_.decoder_output,
                                                                  res_.decoder_state,
                                                                  res_.attns,
                                                                  res_.attn_weights)
                                                                 for res_ in res])
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
              prob = utils.estimate_lm_score(history + [token_id], ngrams)
            score_.append(prob)

          lm_score.append(score_)
        lm_score = np.array(lm_score, dtype=np.float32)
      else:
        lm_score = np.zeros((1, self.trg_vocab_size))

      lm_weight = self.lm_weight or 0.2
      if ngrams is not None:
        weights = [(1 - lm_weight) / len(session)] * len(session) + [lm_weight]

      # FIXME: divide by zero encountered in log
      scores_ = scores[:, None] - np.average([np.log(decoder_output_) for decoder_output_ in decoder_output] +
                                             [lm_score], axis=0, weights=weights)
      scores_ = scores_.flatten()
      flat_ids = np.argsort(scores_)[:beam_size]

      token_ids_ = flat_ids % self.trg_vocab_size
      hyp_ids = flat_ids // self.trg_vocab_size

      new_hypotheses = []
      new_scores = []
      new_state = [[] for _ in session]
      new_input = []
      new_attns = [[] for _ in session]
      new_attn_weights = [[[] for _ in self.encoders] for _ in session]

      for flat_id, hyp_id, token_id in zip(flat_ids, hyp_ids, token_ids_):
        hypothesis = hypotheses[hyp_id] + [token_id]
        score = scores_[flat_id]

        # for debugging purposes
        # if reverse_vocab:
        #   hyp_str = ' '.join(reverse_vocab[id_] if 0 < id_ < len(reverse_vocab) else utils._UNK for id_ in hypothesis)

        if token_id == utils.EOS_ID:
          # early stop: hypothesis is finished, it is thus unnecessary to keep expanding it
          beam_size -= 1  # number of possible hypotheses is reduced by one
          finished_hypotheses.append(hypothesis)
          finished_scores.append(score)
        else:
          new_hypotheses.append(hypothesis)
          for session_id, decoder_state_ in enumerate(decoder_state):
            new_state[session_id].append(decoder_state_[hyp_id])
          new_scores.append(score)
          new_input.append(token_id)

          for session_id, attn_weights_ in enumerate(attn_weights):
            for j in range(len(self.encoders)):
              new_attn_weights[session_id][j].append(attn_weights_[j][hyp_id])
          for session_id, attns_ in enumerate(attns):
            new_attns[session_id].append(attns_[hyp_id])

      hypotheses = new_hypotheses
      state = [np.array(new_state_) for new_state_ in new_state]
      attn_weights = [[np.array(attn_weights_) for attn_weights_ in session_attn_weights]
                      for session_attn_weights in new_attn_weights]
      attns = [np.array(attns_) for attns_ in new_attns]
      scores = np.array(new_scores)
      decoder_input = np.array(new_input, dtype=np.int32)

      if beam_size <= 0:
        break

    hypotheses += finished_hypotheses
    scores = np.concatenate([scores, finished_scores])

    if self.len_normalization > 0:  # normalize score by length (to encourage longer sentences)
      scores /= [len(hypothesis) ** self.len_normalization for hypothesis in hypotheses]

    # sort best-list by score
    sorted_idx = np.argsort(scores)
    hypotheses = np.array(hypotheses)[sorted_idx].tolist()
    scores = scores[sorted_idx].tolist()
    return hypotheses, scores

  def get_batch(self, data, decoding=False):
    decoder_inputs = []
    batch_size = len(data)

    encoder_inputs = [[] for _ in range(self.encoder_count)]
    encoder_input_length = [[] for _ in range(self.encoder_count)]

    if self.buckets is not None:
      # truncate too long sentences
      data = [[x[:n] for x, n in zip(data_, self.buckets[-1])] for data_ in data]

    # maximum sentence length of each encoder in this batch
    max_input_len = [max(len(data[k][i]) for k in xrange(batch_size)) for i in range(self.encoder_count)]
    if self.max_input_len is not None:
      max_input_len = [min(len_, self.max_input_len) for len_ in max_input_len]

    if self.buckets is not None:
      matching_bucket = next(bucket for bucket in self.buckets
                             if all(a <= b for a, b in zip(max_input_len, bucket)))
    else:
      matching_bucket = None

    if decoding:
      max_output_len = self.max_output_len if matching_bucket is None else matching_bucket[-1]
    else:
      max_output_len = max(len(data[k][-1]) for k in xrange(batch_size)) + 1   # + 1 for EOS

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for k in xrange(batch_size):
      sentences = data[k]

      src_sentences = sentences[0:-1]
      trg_sentence = sentences[-1] + [utils.EOS_ID]

      for i, (encoder, src_sentence) in enumerate(zip(self.encoders, src_sentences)):
        if encoder.binary:
          pad = np.zeros([encoder.embedding_size], dtype=np.float32)
        else:
          pad = utils.PAD_ID

        if len(src_sentence) > max_input_len[i]:
          src_sentence = src_sentence[:max_input_len[i]]

        encoder_pad = [pad] * (max_input_len[i] - len(src_sentence))
        reversed_sentence = list(reversed(src_sentence)) + encoder_pad

        encoder_inputs[i].append(reversed_sentence)
        encoder_input_length[i].append(len(src_sentence))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = max_output_len - len(trg_sentence)
      decoder_inputs.append([utils.BOS_ID] + trg_sentence +
                            [utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_decoder_inputs, batch_targets, batch_weights = [], [], []

    encoder_input_length = [np.array(input_length_, dtype=np.int32) for input_length_ in encoder_input_length]

    batch_encoder_inputs = []
    for i, ext in enumerate(self.encoder_names):
      if ext in self.binary_input:
        encoder_inputs_ = np.array(encoder_inputs[i], dtype=np.float32)
      else:
        encoder_inputs_ = np.array(encoder_inputs[i], dtype=np.int32)
      batch_encoder_inputs.append(encoder_inputs_)

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(max_output_len):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))
      batch_targets.append(
          np.array([decoder_inputs[batch_idx][length_idx + 1]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(batch_size, dtype=np.float32)
      for batch_idx in xrange(batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < max_output_len:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if target == utils.PAD_ID:
          batch_weight[batch_idx] = 0.0

      batch_weights.append(batch_weight)

    batch_targets = np.array(batch_targets)
    batch_decoder_inputs = np.array(batch_decoder_inputs)
    batch_weights = np.array(batch_weights)
    return batch_encoder_inputs, batch_decoder_inputs, batch_targets, batch_weights, encoder_input_length
