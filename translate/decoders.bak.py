import tensorflow as tf
import functools
import math
from tensorflow.python.ops import rnn_cell, rnn
from translate.rnn import linear, multi_rnn, orthogonal_initializer
from translate.rnn import multi_bidirectional_rnn, GRUCell
from collections import namedtuple


globals_ = dict()


def build_encoder(encoder_inputs, encoder, encoder_input_length, dropout=None, **kwargs):
    """
    :param encoder_inputs: tensor of shape (batch_size, input_length)
    :param encoder: encoder configuration
    :param encoder_input_length: tensor of shape (batch_size)
    :param dropout: scalar tensor or None, specifying the keep probability (1 - dropout)
    :return:
      encoder outputs: a tensor of shape (batch_size, input_length, cell_size)
      encoder state: final state of the encoder, tensor of shape (batch_size, state_size)
    """
    # create embeddings in the global scope (allows sharing between encoder and decoder)
    # inputs are token ids, which need to be mapped to vectors (embeddings)
    initializer = None
    embedding_shape = [encoder.vocab_size, encoder.embedding_size]

    with tf.device('/cpu:0'):
        embedding = tf.get_variable('embedding_{}'.format(encoder.name), shape=embedding_shape,
                                    initializer=initializer)

    with tf.variable_scope('encoder_{}'.format(encoder.name)):
        # TODO: use state_is_tuple=True
        if encoder.use_lstm:
            cell = rnn_cell.BasicLSTMCell(encoder.cell_size, state_is_tuple=False)
        else:
            cell = GRUCell(encoder.cell_size, initializer=orthogonal_initializer())
            # cell = rnn_cell.BasicRNNCell(encoder.cell_size)

        if dropout is not None:
            cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)

        batch_size = tf.shape(encoder_inputs)[0]  # TODO: fix this time major stuff
        time_steps = tf.shape(encoder_inputs)[1]

        flat_inputs = tf.reshape(encoder_inputs, [tf.mul(batch_size, time_steps)])
        flat_inputs = tf.nn.embedding_lookup(embedding, flat_inputs)

        encoder_inputs = tf.reshape(flat_inputs,
                                    tf.pack([batch_size, time_steps, flat_inputs.get_shape()[1].value]))

        globals_['embedded_inputs'] = encoder_inputs

        # Contrary to Theano's RNN implementation, states after the sequence length are zero
        # (while Theano repeats last state)
        sequence_length = encoder_input_length   # TODO
        parameters = dict(
            inputs=encoder_inputs, sequence_length=sequence_length,
            dtype=tf.float32, swap_memory=encoder.swap_memory,
            parallel_iterations=encoder.parallel_iterations,
            trainable_initial_state=True
        )

        if encoder.bidir:
            encoder_outputs, _, _ = multi_bidirectional_rnn(cells=[(cell, cell)] * encoder.layers, **parameters)
            # Like Bahdanau et al., we use the first annotation h_1 of the backward encoder
            encoder_state = encoder_outputs[:, 0, encoder.cell_size:]
        else:
            encoder_outputs, encoder_state = multi_rnn(cells=[cell] * encoder.layers, **parameters)
            encoder_state = encoder_outputs[:, 0, :]   # FIXME

        return encoder_outputs, encoder_state


def compute_energy(hidden, state, attn_size, **kwargs):
    input_size = hidden.get_shape()[3].value
    batch_size = tf.shape(hidden)[0]
    time_steps = tf.shape(hidden)[1]

    # initializer = tf.random_normal_initializer(stddev=0.001)   # same as Bahdanau et al.
    initializer = None
    y = linear(state, attn_size, True, scope='W_a', initializer=initializer)
    y = tf.reshape(y, [-1, 1, attn_size])

    k = tf.get_variable('U_a', [input_size, attn_size], initializer=initializer)

    # dot product between tensors requires reshaping
    hidden = tf.reshape(hidden, tf.pack([tf.mul(batch_size, time_steps), input_size]))
    f = tf.matmul(hidden, k)
    f = tf.reshape(f, tf.pack([batch_size, time_steps, attn_size]))

    v = tf.get_variable('v_a', [attn_size])
    s = f + y

    return tf.reduce_sum(v * tf.tanh(s), [2])


def attention(state, hidden_states, encoder, encoder_input_length, scope=None, **kwargs):
    with tf.variable_scope(scope or 'attention_{}'.format(encoder.name)):
        e = compute_energy(hidden_states, state, attn_size=encoder.attn_size)
        e = e - tf.reduce_max(e, reduction_indices=(1,), keep_dims=True)

        # batch_size * time_steps
        # TODO: check this mask stuff
        mask = tf.sequence_mask(tf.cast(encoder_input_length, tf.int32), tf.shape(hidden_states)[1],
                                dtype=tf.float32)
        exp = tf.exp(e) * mask
        weights = exp / tf.reduce_sum(exp, reduction_indices=(-1,), keep_dims=True)

        # weights = tf.nn.softmax(e)
        # weights = weights * mask

        shape = tf.shape(weights)
        shape = tf.pack([shape[0], shape[1], 1, 1])

        return tf.reduce_sum(tf.reshape(weights, shape) * hidden_states, [1, 2]), weights, mask, e


def attention_decoder(decoder_inputs, initial_state, attention_states, encoder, decoder, encoder_input_length,
                      decoder_input_length=None, output_projection=None, dropout=None, feed_previous=0.0, **kwargs):
    """
    :param decoder_inputs: tensor of shape (batch_size, output_length)
    :param initial_state: initial state of the decoder (usually the final state of the encoder),
      as a tensor of shape (batch_size, initial_state_size). This state is mapped to the
      correct state size for the decoder.
    :param attention_states: list of tensors of shape (batch_size, input_length, encoder_cell_size),
      usually the encoder outputs (one tensor for each encoder).
    :param encoder: configuration of the encoder
    :param decoder: configuration of the decoder
    :param decoder_input_length:
    :param output_projection: None if no softmax sampling, or tuple (weight matrix, bias vector)
    :param dropout: scalar tensor or None, specifying the keep probability (1 - dropout)
    :param feed_previous: scalar tensor corresponding to the probability to use previous decoder output
      instead of the groundtruth as input for the decoder (1 when decoding, between 0 and 1 when training)
    :return:
      outputs of the decoder as a tensor of shape (batch_size, output_length, decoder_cell_size)
      attention weights as a tensor of shape (output_length, encoders, batch_size, input_length)
    """
    with tf.device('/cpu:0'):
        embedding = tf.get_variable('embedding_{}'.format(decoder.name),
                                    shape=[decoder.vocab_size, decoder.embedding_size])

    if decoder.use_lstm:
        cell = rnn_cell.BasicLSTMCell(decoder.cell_size, state_is_tuple=False)
    else:
        cell = GRUCell(decoder.cell_size, initializer=orthogonal_initializer())
        # cell = rnn_cell.BasicRNNCell(decoder.cell_size)

    if dropout is not None:
        cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)

    if decoder.layers > 1:
        cell = rnn_cell.MultiRNNCell([cell] * decoder.layers)

    with tf.variable_scope('decoder_{}'.format(decoder.name)):
        def extract_argmax_and_embed(prev):
            if output_projection is not None:
                prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        input_shape = tf.shape(decoder_inputs)
        time_steps = input_shape[0]
        batch_size = input_shape[1]
        state_size = cell.state_size

        flat_inputs = tf.reshape(decoder_inputs, [tf.mul(batch_size, time_steps)])
        flat_inputs = tf.nn.embedding_lookup(embedding, flat_inputs)
        decoder_inputs = tf.reshape(flat_inputs, tf.pack([time_steps, batch_size, flat_inputs.get_shape()[1].value]))

        if dropout is not None:
            initial_state = tf.nn.dropout(initial_state, dropout)

        state = tf.nn.tanh(linear(initial_state, state_size, True, scope='initial_state_projection'))

        sequence_length = tf.to_int32(decoder_input_length)
        min_sequence_length = tf.reduce_min(sequence_length)
        max_sequence_length = tf.reduce_max(sequence_length)

        time = tf.constant(0, dtype=tf.int32, name='time')

        zero_output = tf.zeros(tf.pack([batch_size, cell.output_size]), tf.float32)
        output_ta = tf.TensorArray(dtype=tf.float32, size=time_steps - 1, clear_after_read=False)
        input_ta = tf.TensorArray(dtype=tf.float32, size=time_steps, clear_after_read=False).unpack(decoder_inputs)

        # weights_ta = tf.TensorArray(dtype=tf.float32, size=time_steps - 1)

        output = tf.zeros(tf.pack([batch_size, cell.output_size]), dtype=tf.float32)

        def _time_step(time, state, output, output_ta):
            r = tf.random_uniform([])
            input_ = tf.cond(tf.logical_and(time > 0, r < feed_previous),
                             lambda: tf.stop_gradient(extract_argmax_and_embed(output_ta.read(time - 1))),
                             lambda: input_ta.read(time))
            input_.set_shape(decoder_inputs.get_shape()[1:])

            context_vector, weights, mask, e = attention(state, hidden_states=tf.expand_dims(attention_states, 2),
                                                         encoder=encoder, encoder_input_length=encoder_input_length)

            # output_1 = linear(state, decoder.cell_size, True, scope='maxout_1')
            # output_2 = linear(input_, decoder.cell_size, False, scope='maxout_2')
            # output_3 = linear(context_vector, decoder.cell_size, False, scope='maxout_3')

            # output_ = output_1 + output_2 + output_3
            #
            # output_ = linear(output_, decoder.vocab_size, True, scope='softmax')

            # maxout layer
            output_ = linear([state, input_, context_vector], decoder.cell_size, True, scope='maxout')
            output_ = tf.maximum(*tf.split(1, 2, output_))
            output_ = linear(output_, decoder.embedding_size, False, scope='softmax0')
            output_ = linear(output_, decoder.vocab_size, True, scope='softmax1')

            output_ta = output_ta.write(time, output_)

            input_ = tf.cond(tf.logical_and(time > 0, r < feed_previous),
                             lambda: tf.stop_gradient(extract_argmax_and_embed(output_)),
                             lambda: input_ta.read(time + 1))
            input_.set_shape(decoder_inputs.get_shape()[1:])

            # x = tf.concat(1, [input_, context_vector])

            call_cell = lambda: cell(input_, state, attn=context_vector)

            new_output, new_state = rnn._rnn_step(
                time=time,
                sequence_length=sequence_length,
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                zero_output=zero_output,
                state=state,
                call_cell=call_cell,
                state_size=state_size,
                skip_conditionals=True)

            return time + 1, new_state, new_output, output_ta

        _, new_state, new_output, output_final_ta = tf.while_loop(
            cond=lambda time, *_: time < time_steps + 1,
            body=_time_step,
            loop_vars=(time, state, output, output_ta),
            parallel_iterations=decoder.parallel_iterations,
            swap_memory=decoder.swap_memory)

        outputs = output_final_ta.pack()

        beam_tensors = namedtuple('beam_tensors', 'state new_state output new_output')
        return outputs, beam_tensors(state, new_state, output, new_output)


def sequence_loss(logits, targets, weights, average_across_timesteps=False, average_across_batch=True):
    time_steps = tf.shape(targets)[0]
    batch_size = tf.shape(targets)[1]

    logits_ = tf.reshape(logits, tf.pack([time_steps * batch_size, logits.get_shape()[2].value]))
    targets_ = tf.reshape(targets, tf.pack([time_steps * batch_size]))

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_, targets_)

    crossent = tf.reshape(crossent, tf.pack([time_steps, batch_size]))
    log_perp = tf.reduce_sum(crossent * weights, 0)

    if average_across_timesteps:
        total_size = tf.reduce_sum(weights, 0)
        total_size += 1e-12  # just to avoid division by 0 for all-0 weights
        log_perp /= total_size

    cost = tf.reduce_sum(log_perp)

    if average_across_batch:
        batch_size = tf.shape(targets)[1]
        return cost / tf.cast(batch_size, tf.float32)
    else:
        return cost
