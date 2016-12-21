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

import numpy as np
import tensorflow as tf
import math

from translate import utils, decoders
from collections import namedtuple


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

    def __init__(self, encoder, decoder, learning_rate, global_step, max_gradient_norm, dropout_rate=0.0,
                 max_output_len=50, feed_previous=0.0, optimizer='sgd', max_input_len=None, decode_only=False,
                 **kwargs):
        self.encoder = encoder
        self.decoder = decoder

        self.learning_rate = learning_rate
        self.global_step = global_step

        self.trg_vocab_size = decoder.vocab_size
        self.trg_cell_size = decoder.cell_size

        self.max_output_len = max_output_len
        self.max_input_len = max_input_len

        if dropout_rate > 0:
            self.dropout = tf.Variable(1 - dropout_rate, trainable=False, name='dropout_keep_prob')
            self.dropout_off = self.dropout.assign(1.0)
            self.dropout_on = self.dropout.assign(1 - dropout_rate)
        else:
            self.dropout = None

        self.feed_previous = tf.constant(feed_previous, dtype=tf.float32)

        # batch_size x time
        self.encoder_inputs = tf.placeholder(tf.int32, shape=[None, None],
                                             name='encoder_{}'.format(encoder.name))
        self.encoder_input_length = tf.placeholder(tf.int64, shape=[None], name='encoder_{}_length'.format(encoder.name))

        # time x batch_size
        self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, None],
                                             name='decoder_{}'.format(self.decoder.name))
        self.target_weights = tf.placeholder(tf.float32, shape=[None, None],
                                             name='weight_{}'.format(self.decoder.name))
        self.targets = tf.placeholder(tf.int32, shape=[None, None], name='target_{}'.format(self.decoder.name))

        self.decoder_input_length = tf.placeholder(tf.int64, shape=[None],
                                                   name='decoder_{}_length'.format(decoder.name))

        parameters = dict(encoder=encoder, decoder=decoder, dropout=self.dropout,
                          encoder_input_length=self.encoder_input_length)

        self.attention_states, self.encoder_state = decoders.build_encoder(self.encoder_inputs, **parameters)

        self.outputs, self.beam_tensors = decoders.attention_decoder(
            attention_states=self.attention_states, initial_state=self.encoder_state,
            decoder_inputs=self.decoder_inputs, feed_previous=self.feed_previous,
            decoder_input_length=self.decoder_input_length, **parameters
        )

        self.loss = decoders.sequence_loss(
            logits=self.outputs, targets=self.targets, weights=self.target_weights)

        if not decode_only:
            # gradients and SGD update operation for training the model
            if optimizer.lower() == 'adadelta':
                # same epsilon and rho as Bahdanau et al. 2015
                opt = tf.train.AdadeltaOptimizer(learning_rate=1.0, epsilon=1e-06, rho=0.95)
            elif optimizer.lower() == 'adagrad':
                opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            elif optimizer.lower() == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=0.001)
            else:
                opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)

            clipped_gradients, self.gradient_norms = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.beam_output = tf.nn.softmax(self.outputs[0,:,:])

    def step(self, session, data, forward_only=False):
        if self.dropout is not None:
            session.run(self.dropout_on)

        batch = self.get_batch(data)
        encoder_inputs, decoder_inputs, targets, target_weights, encoder_input_length, decoder_input_length = batch

        input_feed = {
            self.encoder_input_length: encoder_input_length,
            self.encoder_inputs: encoder_inputs,
            self.target_weights: target_weights,
            self.decoder_inputs: decoder_inputs,
            self.decoder_input_length: decoder_input_length,
            self.targets: targets
        }

        output_feed = {'loss': self.loss}
        if not forward_only:
            output_feed['updates'] = self.updates

        res = session.run(output_feed, input_feed)
        return res['loss']

    def greedy_decoding(self, session, token_ids):
        if self.dropout is not None:
            session.run(self.dropout_off)

        batch = self.get_batch([[token_ids, []]], decoding=True)
        encoder_inputs, decoder_inputs, targets, target_weights, encoder_input_length, decoder_input_length = batch

        input_feed = {
            self.encoder_input_length: encoder_input_length,
            self.encoder_inputs: encoder_inputs,
            self.target_weights: target_weights,
            self.decoder_inputs: decoder_inputs,
            self.decoder_input_length: decoder_input_length,
            self.targets: targets,
            self.feed_previous: 1.0
        }

        outputs = session.run(self.outputs, input_feed)

        return [int(np.argmax(logit, axis=1)) for logit in outputs]  # greedy decoder

    def beam_search_decoding(self, session, token_ids, beam_size):
        if self.dropout is not None:
            session.run(self.dropout_off)

        batch = self.get_batch([[token_ids, []]], decoding=True)
        encoder_inputs, decoder_inputs, targets, target_weights, encoder_input_length, _ = batch

        input_feed = {
            self.encoder_input_length: encoder_input_length,
            self.encoder_inputs: encoder_inputs
        }
        output_feed = [self.encoder_state, self.attention_states]
        state, attn_states = session.run(output_feed, input_feed)

        decoder_input = decoder_inputs[0]  # BOS symbol

        finished_hypotheses = []
        finished_scores = []

        hypotheses = [[]]
        scores = np.zeros([1], dtype=np.float32)

        # for initial state projection
        state = session.run(self.beam_tensors.state, {self.encoder_state: state})

        for i in range(self.max_output_len):
            batch_size = decoder_input.shape[0]

            input_feed = {
                self.encoder_input_length: np.tile(encoder_input_length, batch_size),
                self.beam_tensors.state: state,
                self.decoder_inputs: np.reshape(decoder_input, [1, batch_size]),
                self.decoder_input_length: [1] * batch_size,
                self.attention_states: attn_states.repeat(batch_size, axis=0)
            }

            output_feed = [self.beam_output, self.beam_tensors.new_state]
            decoder_output, decoder_state = session.run(output_feed, input_feed)

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
                    # early stop: hypothesis is finished, it is thus unnecessary to keep expanding it
                    beam_size -= 1  # number of possible hypotheses is reduced by one
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

        # normalize score by length (to encourage longer sentences)
        scores /= [len(hypothesis) for hypothesis in hypotheses]

        # sort best-list by score
        sorted_idx = np.argsort(scores)
        hypotheses = np.array(hypotheses)[sorted_idx].tolist()
        scores = scores[sorted_idx].tolist()
        return hypotheses, scores

    def get_batch(self, data, decoding=False):
        """
        :param data:
        :param decoding: set this parameter to True to output dummy
          data for the decoder side (using the maximum output size)
        :return:
        """
        src_sentences, trg_sentences = zip(*data)

        encoder_inputs = []
        encoder_input_length = []
        decoder_inputs = []
        decoder_input_length = []

        # maximum input length in this batch
        max_input_len = min(max(len(sentence) for sentence in src_sentences), self.max_input_len)
        # maximum output length in this batch
        max_output_len = min(max(len(sentence) for sentence in trg_sentences), self.max_output_len)

        if decoding:
            max_output_len = min(max_output_len, 2 * max_input_len)

        for src_sentence, trg_sentence in zip(src_sentences, trg_sentences):
            # pad sequences so that all sequences in the same batch have the same length
            src_sentence = src_sentence[:max_input_len]
            encoder_pad = [utils.EOS_ID] * (1 + max_input_len - len(src_sentence))

            encoder_inputs.append(src_sentence + encoder_pad)
            encoder_input_length.append(len(src_sentence) + 1)

            trg_sentence = trg_sentence[:max_output_len]
            if decoding:
                decoder_input_length.append(self.max_output_len)
                decoder_inputs.append([utils.BOS_ID] + [utils.EOS_ID] * self.max_output_len)
            else:
                decoder_pad_size = max_output_len - len(trg_sentence)
                decoder_input_length.append(len(trg_sentence) + 1)
                trg_sentence = [utils.BOS_ID] + trg_sentence + [utils.EOS_ID] + [-1] * decoder_pad_size
                decoder_inputs.append(trg_sentence)

        # convert lists to numpy arrays
        encoder_input_length = np.array(encoder_input_length, dtype=np.int32)
        decoder_input_length = np.array(decoder_input_length, dtype=np.int32)
        batch_encoder_inputs = np.array(encoder_inputs, np.int32)

        # time-major vectors: shape is (time, batch_size)
        batch_decoder_inputs = np.array(decoder_inputs)[:, :-1].T  # with BOS symbol, without EOS symbol
        batch_targets = np.array(decoder_inputs)[:, 1:].T  # without BOS symbol, with EOS symbol
        batch_weights = (batch_targets != -1).astype(np.float32)  # PAD symbols don't count for training

        batch_decoder_inputs[batch_decoder_inputs == -1] = utils.EOS_ID
        batch_targets[batch_targets == -1] = utils.EOS_ID

        return (batch_encoder_inputs,
                batch_decoder_inputs,
                batch_targets,
                batch_weights,
                encoder_input_length,
                decoder_input_length)
