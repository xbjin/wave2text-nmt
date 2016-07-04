from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.models.rnn
import functools
import math
from tensorflow.python.ops import rnn, rnn_cell


def unsafe_get_variable(name, *args, **kwargs):
  """ Gets a variable without worrying about the reuse parameter """
  try:
    return tf.get_variable(name, *args, **kwargs)
  except ValueError:
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      return tf.get_variable(name, *args, **kwargs)


def multi_encoder(encoder_inputs, encoder_names, cell, num_encoder_symbols, embedding_size,
                  encoder_input_length=None, embeddings=None, reuse=None, bidir=False, dynamic=False,
                  **kwargs):
  assert len(encoder_inputs) == len(encoder_names)

  # convert embeddings to tensors
  embeddings = {name: (tf.convert_to_tensor(embedding, dtype=tf.float32))
                for name, embedding in embeddings.items()}

  encoder_states = []
  encoder_outputs = []

  with tf.variable_scope('multi_encoder'):
    if reuse:
      tf.get_variable_scope().reuse_variables()

    if encoder_input_length is None:
      encoder_input_length = [None] * len(encoder_names)

    for encoder_name, encoder_inputs_, encoder_input_length_, num_encoder_symbols_ in zip(
        encoder_names, encoder_inputs, encoder_input_length, num_encoder_symbols):
      with tf.variable_scope('encoder_{}'.format(encoder_name)):
        initializer = embeddings.get(encoder_name)
        if initializer is None:
          initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
          embedding_shape = [num_encoder_symbols_, embedding_size]
        else:
          embedding_shape = None

        with tf.device('/cpu:0'):
          embedding = tf.get_variable('embedding', shape=embedding_shape,
                                      initializer=initializer)

        encoder_inputs_ = [tf.nn.embedding_lookup(embedding, i) for i in encoder_inputs_]

        if bidir:
          # TODO: wrong output shape, not compatible with `dynamic_rnn`
          raise NotImplementedError
          # encoder_outputs_, encoder_state_fw, encoder_state_bw = rnn.bidirectional_rnn(cell, cell, encoder_inputs_,
          #                                                                      sequence_length=encoder_input_length_,
          #                                                                      dtype=tf.float32)
          # encoder_state_ = rnn_cell.linear([encoder_state_fw, encoder_state_bw], cell.state_size, True)
        elif dynamic:
          encoder_inputs_ = tf.transpose(
            tf.reshape(tf.concat(0, encoder_inputs_), [len(encoder_inputs_), -1, embedding_size]),
            perm=[1, 0, 2])
          encoder_outputs_, encoder_state_ = rnn.dynamic_rnn(cell, encoder_inputs_,
                                                              sequence_length=encoder_input_length_,
                                                              dtype=tf.float32, parallel_iterations=1)
        else:
          encoder_input_length_ = None   # TODO: check impact of this parameter
          encoder_outputs_, encoder_state_ = rnn.rnn(cell, encoder_inputs_, sequence_length=encoder_input_length_,
                                                     dtype=tf.float32)

        encoder_states.append(encoder_state_)
        encoder_outputs.append(encoder_outputs_)

    encoder_state = tf.add_n(encoder_states)

    if dynamic and not bidir:
      attention_states = encoder_outputs
    else:
      top_states = [[tf.reshape(e, [-1, 1, cell.output_size]) for e in v]
                    for v in encoder_outputs]
      attention_states = [tf.concat(1, v) for v in top_states]

    return attention_states, encoder_state


def attention(state, prev_weights, hidden_states, encoder_names, attn_length, attn_size, batch_size,
              attention_filters=0, attention_filter_length=0, reuse=None):
  assert len(hidden_states) == len(encoder_names)

  with tf.variable_scope('attention', reuse):
    hidden_features = []
    v = []
    attn_filters = []
    u = []

    for encoder_name, hidden_ in zip(encoder_names, hidden_states):
      # attention_decoder/attention/W_{encoder_name}
      k = tf.get_variable('W_{}'.format(encoder_name), [1, 1, attn_size, attn_size])
      hidden_features.append(tf.nn.conv2d(hidden_, k, [1, 1, 1, 1], 'SAME'))  # same as a dot product
      # attention_decoder/attention/V_{encoder_name}
      v.append(tf.get_variable('V_{}'.format(encoder_name), [attn_size]))

      filter_ = None
      u_ = None
      if attention_filters > 0:
        filter_ = tf.get_variable('filter_{}'.format(encoder_name),
                                  [attention_filter_length * 2 + 1, 1, 1, attention_filters])
        u_ = tf.get_variable('U_{}'.format(encoder_name), [attention_filters, attn_size])
      u.append(u_)
      attn_filters.append(filter_)

    # attention_decoder/attention/Linear/Matrix
    # attention_decoder/attention/Linear/Bias
    y = rnn_cell.linear(state, attn_size, True)
    y = tf.reshape(y, [-1, 1, 1, attn_size])

    weights = []
    ds = []

    if prev_weights is None:
      prev_weights = len(encoder_names) * [None]

    # for each encoder
    for f, h, v_, prev_a, filter_, u_ in zip(hidden_features, hidden_states, v, prev_weights, attn_filters, u):
      # attention mask is a softmax of v^T * tanh(...)
      if filter_ is not None and u_ is not None:
        prev_a = tf.reshape(prev_a, [-1, attn_length, 1, 1])
        # compute convolution between prev_a and filter_
        conv = tf.nn.conv2d(prev_a, filter_, [1, 1, 1, 1], 'SAME')
        # flattening for dot product
        shape = tf.pack([tf.mul(batch_size, attn_length), attention_filters])
        conv = tf.reshape(conv, shape)
        z = tf.matmul(conv, u_)
        z = tf.reshape(z, [-1, attn_length, 1, attn_size])

        s = f + y + z
      else:
        s = f + y

      e = tf.reduce_sum(v_ * tf.tanh(s), [2, 3])
      a = tf.nn.softmax(e)
      weights.append(a)

      # now calculate the attention-weighted vector d
      d = tf.reduce_sum(tf.reshape(a, [-1, attn_length, 1, 1]) * h, [1, 2])
      ds.append(d)

    weighted_average = tf.add_n(ds)  # just sum the context vector of each encoder (TODO: add weights there)
    weighted_average = tf.reshape(weighted_average, [-1, attn_size])

    return weighted_average, weights


def attention_decoder(decoder_inputs, initial_state, attention_states,
                      encoder_names, decoder_name, cell, num_decoder_symbols, embedding_size,
                      attention_weights=None,
                      feed_previous=False, output_projection=None, embeddings=None,
                      initial_state_attention=False,
                      attention_filters=0, attention_filter_length=0, reuse=None, **kwargs):
  # TODO: dynamic RNN
  embedding_initializer = embeddings.get(decoder_name)
  if embedding_initializer is None:
    # embedding_initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
    embedding_initializer = None
    embedding_shape = [num_decoder_symbols, embedding_size]
  else:
    embedding_shape = None

  if output_projection is None:
    cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
    output_size = num_decoder_symbols
  else:
    output_size = cell.output_size

  if output_projection is not None:
    proj_weights = tf.convert_to_tensor(output_projection[0], dtype=tf.float32)
    proj_weights.get_shape().assert_is_compatible_with([cell.output_size, num_decoder_symbols])
    proj_biases = tf.convert_to_tensor(output_projection[1], dtype=tf.float32)
    proj_biases.get_shape().assert_is_compatible_with([num_decoder_symbols])

  with tf.variable_scope('attention_decoder'):
    if reuse:
      tf.get_variable_scope().reuse_variables()

    with tf.device('/cpu:0'):
      embedding = tf.get_variable('embedding', shape=embedding_shape,
                                  initializer=embedding_initializer)

    def extract_argmax_and_embed(prev):
      """ Loop_function that extracts the symbol from prev and embeds it """
      if output_projection is not None:
        prev = tf.nn.xw_plus_b(
          prev, output_projection[0], output_projection[1])
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
      return emb_prev

    loop_function = extract_argmax_and_embed if feed_previous else None
    decoder_inputs = [tf.nn.embedding_lookup(embedding, i) for i in decoder_inputs]

    batch_size = tf.shape(decoder_inputs[0])[0]  # needed for reshaping
    attn_length = attention_states[0].get_shape()[1].value
    attn_size = attention_states[0].get_shape()[2].value

    # to calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before
    hidden_states = [
      tf.reshape(states, [-1, attn_length, 1, attn_size])
      for states in attention_states
    ]

    attention_ = functools.partial(attention, hidden_states=hidden_states,
                                   encoder_names=encoder_names, attn_length=attn_length,
                                   attn_size=attn_size, batch_size=batch_size,
                                   attention_filters=attention_filters,
                                   attention_filter_length=attention_filter_length)

    state = initial_state

    outputs = []
    all_attention_weights = []
    prev = None
    decoder_states = []

    if attention_weights is None:
      attention_weights = [
        tf.zeros(tf.pack([batch_size, attn_length]))
        for _ in encoder_names
      ]
      for weights in attention_weights:
        weights.set_shape([None, attn_length])

    if initial_state_attention:
      # for beam-search decoder (feed_dict substitution)
      all_attention_weights.append(attention_weights)
      attns, attention_weights = attention_(initial_state, attention_weights)
    else:
      attns = tf.zeros(tf.pack([batch_size, attn_size]), dtype=tf.float32)
      attns.set_shape([None, attn_size])

    for i, inputs in enumerate(decoder_inputs):
      if i > 0: tf.get_variable_scope().reuse_variables()

      # if loop_function is set, we use it instead of decoder_inputs
      if loop_function is not None and prev is not None:
        with tf.variable_scope('loop_function', reuse=True):
          inputs = tf.stop_gradient(loop_function(prev))

      # merge input and previous attentions into one vector of the right size
      input_size = inputs.get_shape().with_rank(2)[1]
      # attention_decoder/Linear/Matrix
      # attention_decoder/Linear/Bias
      x = rnn_cell.linear([inputs, attns], input_size, True)

      # run the RNN
      cell_output, state = cell(x, state)
      all_attention_weights.append(attention_weights)
      decoder_states.append(state)

      # run the attention mechanism
      attns, attention_weights = attention_(state, attention_weights, reuse=initial_state_attention)

      if output_projection is None:
        output = cell_output
      else:
        # with tf.device('/cpu:0'):  # TODO try this
        with tf.variable_scope('attention_output_projection'):
          # FIXME: where does this come from?
          output = rnn_cell.linear([cell_output, attns], output_size, True)
      outputs.append(output)

      if loop_function is not None:
        # we do not propagate gradients over the loop function
        prev = tf.stop_gradient(output)

    return outputs, decoder_states, all_attention_weights


def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, softmax_loss_function=None, reuse=None, **kwargs):
  attention_states, encoder_state = encoder_with_buckets(encoder_inputs, buckets, reuse, **kwargs)
  outputs, _, _ = decoder_with_buckets(attention_states, encoder_state, decoder_inputs,
                                       buckets, reuse, **kwargs)
  losses = loss_with_buckets(outputs, targets, weights, buckets,
                             softmax_loss_function, reuse)
  return outputs, losses


def encoder_with_buckets(encoder_inputs, buckets, reuse=None, **kwargs):
  encoder_inputs_concat = [v for inputs in encoder_inputs for v in inputs]

  attention_states = []
  encoder_state = []

  with tf.op_scope(encoder_inputs_concat, 'model_with_buckets'):
    for j, bucket in enumerate(buckets):
      reuse_ = reuse or (True if j > 0 else None)

      with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_):
        encoder_size, _ = bucket

        encoder_inputs_trunc = [v[:encoder_size] for v in encoder_inputs]

        bucket_attention_states, bucket_encoder_state = multi_encoder(
          encoder_inputs_trunc, **kwargs)
        attention_states.append(bucket_attention_states)
        encoder_state.append(bucket_encoder_state)

  return attention_states, encoder_state


def decoder_with_buckets(attention_states, encoder_state, decoder_inputs,
                         buckets, reuse=None, **kwargs):
  outputs = []
  states = []
  attention_weights = []

  with tf.op_scope(decoder_inputs, 'model_with_buckets'):
    for bucket, bucket_attention_states, bucket_encoder_state in zip(buckets,
                                                                     attention_states,
                                                                     encoder_state):
      reuse_ = reuse or (True if bucket is not buckets[0] else None)

      with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_):
        _, decoder_size = bucket

        bucket_outputs, bucket_states, bucket_attention_weights = attention_decoder(
          decoder_inputs[:decoder_size], bucket_encoder_state, bucket_attention_states, **kwargs)

        outputs.append(bucket_outputs)
        states.append(bucket_states)
        attention_weights.append(bucket_attention_weights)

  return outputs, states, attention_weights


def loss_with_buckets(outputs, targets, weights, buckets, softmax_loss_function=None,
                      reuse=None):
  losses = []

  with tf.op_scope(targets + weights, 'model_with_buckets'):
    for bucket, bucket_outputs in zip(buckets, outputs):
      reuse_ = reuse or (True if bucket is not buckets[0] else None)

      with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_):
        _, decoder_size = bucket

        losses.append(tf.models.rnn.seq2seq.sequence_loss(
          bucket_outputs, targets[:decoder_size], weights[:decoder_size],
          softmax_loss_function=softmax_loss_function))

  return losses
