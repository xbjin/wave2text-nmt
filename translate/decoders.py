from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import functools
import math
from tensorflow.python.ops import rnn_cell, seq2seq
from translate import rnn


def unsafe_decorator(fun):
  """
  Wrapper that automatically handles the `reuse' parameter.
  This is rather unsafe, as it can lead to reusing variables
  by mistake, without knowing about it.
  """
  def fun_(*args, **kwargs):
    try:
      return fun(*args, **kwargs)
    except ValueError as e:
      if 'reuse' in str(e):
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
          return fun(*args, **kwargs)
      else:
        raise e
  return fun_


get_variable_unsafe = unsafe_decorator(tf.get_variable)
GRUCell_unsafe = unsafe_decorator(rnn_cell.GRUCell)
BasicLSTMCell_unsafe = unsafe_decorator(rnn_cell.BasicLSTMCell)
MultiRNNCell_unsafe = unsafe_decorator(rnn_cell.MultiRNNCell)
rnn_unsafe = unsafe_decorator(rnn.rnn)
dynamic_rnn_unsafe = unsafe_decorator(rnn.dynamic_rnn)
try:
  linear_unsafe = unsafe_decorator(rnn_cell.linear)
except AttributeError:
  linear_unsafe = unsafe_decorator(rnn_cell._linear)
multi_bidirectional_rnn_unsafe = unsafe_decorator(rnn.multi_bidirectional_rnn)


def multi_encoder(encoder_inputs, encoders, encoder_input_length=None, dropout=None, **kwargs):
  assert len(encoder_inputs) == len(encoders)
  encoder_states = []
  encoder_outputs = []

  # create embeddings in the global scope (allows sharing between encoder and decoder)
  embedding_variables = []
  for encoder in encoders:
    # inputs are token ids, which need to be mapped to vectors (embeddings)
    if not encoder.binary:
      if encoder.get('embedding') is not None:
        initializer = encoder.embedding
        embedding_shape = None
      else:
        initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
        embedding_shape = [encoder.vocab_size, encoder.embedding_size]

      # TODO: shared name
      with tf.device('/cpu:0'):
        embedding = get_variable_unsafe('embedding_{}'.format(encoder.name), shape=embedding_shape,
                                        initializer=initializer)
        embedding_variables.append(embedding)
    else:  # do nothing: inputs are already vectors
      embedding_variables.append(None)

  with tf.variable_scope('multi_encoder'):
    if encoder_input_length is None:
      encoder_input_length = [None] * len(encoders)

    for i, encoder in enumerate(encoders):
      with tf.variable_scope(encoder.name):
        encoder_inputs_ = encoder_inputs[i]
        encoder_input_length_ = encoder_input_length[i]

        if encoder.use_lstm:
          # FIXME: state_is_tuple=False
          cell = rnn_cell.BasicLSTMCell(encoder.cell_size)
        else:
          cell = rnn_cell.GRUCell(encoder.cell_size)

        if dropout is not None:
            cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)

        embedding = embedding_variables[i]
        if embedding is not None:
          encoder_inputs_ = [tf.nn.embedding_lookup(embedding, i) for i in encoder_inputs_]

          if not encoder.bidir and encoder.layers > 1:  # bidir requires custom multi-rnn
            cell = rnn_cell.MultiRNNCell([cell] * encoder.layers)

          # TODO: pooling over time for non-bidir encoders
          if encoder.bidir:  # FIXME: not compatible with `dynamic`
            encoder_outputs_, encoder_state_fw, encoder_state_bw = multi_bidirectional_rnn_unsafe(
              [(cell, cell)] * encoder.layers, encoder_inputs_, time_pooling=encoder.time_pooling,
              pooling_avg=encoder.pooling_avg, dtype=tf.float32
            )
            encoder_state_ = encoder_state_bw
            # same as Bahdanau et al.:
            # encoder_state_ = linear_unsafe(encoder_state_bw, cell.state_size, True,
            #                                scope='bidir_final_state')
            # slightly different (they do projection later):
            encoder_outputs_ = [
              linear_unsafe(outputs_, cell.output_size, False,
                            scope='bidir_projection') for outputs_ in encoder_outputs_]
          elif encoder.dynamic:
            encoder_inputs_ = tf.transpose(
              tf.reshape(tf.concat(0, encoder_inputs_), [len(encoder_inputs_), -1, encoder.embedding_size]),
              perm=[1, 0, 2])
            encoder_outputs_, encoder_state_ = dynamic_rnn_unsafe(cell, encoder_inputs_,
                                                                  sequence_length=encoder_input_length_,
                                                                  dtype=tf.float32, parallel_iterations=1)
          else:
            encoder_input_length_ = None   # TODO: check impact of this parameter
            encoder_outputs_, encoder_state_ = rnn_unsafe(cell, encoder_inputs_, sequence_length=encoder_input_length_,
                                                          dtype=tf.float32)

          if encoder.bidir or not encoder.dynamic:
            encoder_outputs_ = [tf.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs_]
            encoder_outputs_ = tf.concat(1, encoder_outputs_)

          encoder_outputs.append(encoder_outputs_)
          encoder_states.append(encoder_state_)

    encoder_state = tf.concat(1, encoder_states)
    return encoder_outputs, encoder_state


def compute_energy(hidden, state, name, **kwargs):
  attn_size = hidden.get_shape()[3].value

  y = linear_unsafe(state, attn_size, True, scope=name)
  y = tf.reshape(y, [-1, 1, 1, attn_size])

  k = get_variable_unsafe('W_{}'.format(name), [1, 1, attn_size, attn_size])
  # complicated way to do a dot product
  f = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], 'SAME')

  v = get_variable_unsafe('V_{}'.format(name), [attn_size])
  s = f + y
  return tf.reduce_sum(v * tf.tanh(s), [2, 3])


def compute_energy_with_filter(hidden, state, name, prev_weights, attention_filters,
                               attention_filter_length, **kwargs):
  attn_length = hidden.get_shape()[1].value
  attn_size = hidden.get_shape()[3].value
  batch_size = tf.shape(hidden)[0]

  filter_shape = [attention_filter_length * 2 + 1, 1, 1, attention_filters]
  filter_ = get_variable_unsafe('filter_{}'.format(name), filter_shape)
  u = get_variable_unsafe('U_{}'.format(name), [attention_filters, attn_size])
  prev_weights = tf.reshape(prev_weights, [-1, attn_length, 1, 1])
  conv = tf.nn.conv2d(prev_weights, filter_, [1, 1, 1, 1], 'SAME')
  shape = tf.pack([tf.mul(batch_size, attn_length), attention_filters])
  conv = tf.reshape(conv, shape)
  z = tf.matmul(conv, u)
  z = tf.reshape(z, [-1, attn_length, 1, attn_size])

  y = linear_unsafe(state, attn_size, True)
  y = tf.reshape(y, [-1, 1, 1, attn_size])

  k = get_variable_unsafe('W_{}'.format(name), [1, 1, attn_size, attn_size])
  f = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], 'SAME')

  v = get_variable_unsafe('V_{}'.format(name), [attn_size])
  s = f + y + z
  return tf.reduce_sum(v * tf.tanh(s), [2, 3])


def global_attention(state, prev_weights, hidden_states, encoder, **kwargs):
  attn_length = hidden_states.get_shape()[1].value

  with tf.variable_scope('attention'):

    compute_energy_ = compute_energy_with_filter if encoder.attention_filters > 0 else compute_energy
    e = compute_energy_(hidden_states, state, encoder.name,
                        prev_weights=prev_weights, attention_filters=encoder.attention_filters,
                        attention_filter_length=encoder.attention_filter_length)
    weights = tf.nn.softmax(e)
    weighted_average = tf.reduce_sum(tf.reshape(weights, [-1, attn_length, 1, 1]) * hidden_states, [1, 2])
    return weighted_average, weights


def local_attention(state, prev_weights, hidden_states, encoder, **kwargs):
  """
  Local attention of Luong et al. (http://arxiv.org/abs/1508.04025)
  """
  attn_length = hidden_states.get_shape()[1].value
  state_size = state.get_shape()[1].value

  with tf.variable_scope('attention'):
    S = hidden_states.get_shape()[1].value   # source length
    wp = get_variable_unsafe('Wp_{}'.format(encoder.name), [state_size, state_size])
    vp = get_variable_unsafe('vp_{}'.format(encoder.name), [state_size, 1])

    pt = tf.nn.sigmoid(tf.matmul(tf.nn.tanh(tf.matmul(state, wp)), vp))
    pt = tf.floor(S * tf.reshape(pt, [-1, 1]))  # aligned position in the source sentence

    # state's shape is (?, bucket_size, 1, state_size)
    batch_size = tf.shape(state)[0]

    indices = tf.convert_to_tensor(range(attn_length), dtype=tf.float32)
    idx = tf.tile(indices, tf.pack([batch_size]))
    idx = tf.reshape(idx, [-1, attn_length])

    low = pt - encoder.attention_window_size
    high = pt + encoder.attention_window_size

    # FIXME: is this really more efficient than global attention?
    mlow = tf.to_float(idx < low)
    mhigh =  tf.to_float(idx > high)
    m = mlow + mhigh
    mask = tf.to_float(tf.equal(m, 0.0))

    compute_energy_ = compute_energy_with_filter if encoder.attention_filters > 0 else compute_energy
    e = compute_energy_(hidden_states, state, encoder.name,
                        prev_weights=prev_weights, attention_filters=encoder.attention_filters,
                        attention_filter_length=encoder.attention_filter_length)

    weights = tf.nn.softmax(e * mask)

    sigma = encoder.attention_window_size / 2
    numerator = -tf.pow((idx - pt), tf.convert_to_tensor(2, dtype=tf.float32))
    div = tf.truediv(numerator, sigma ** 2)

    weights = weights * tf.exp(div)   # result of the truncated normal distribution
    weighted_average = tf.reduce_sum(tf.reshape(weights, [-1, attn_length, 1, 1]) * hidden_states, [1, 2])
    return weighted_average, weights


def attention(state, prev_weights, hidden_states, encoder, **kwargs):
  """
  Proxy for `local_attention` and `global_attention`
  """
  if encoder.attention_window_size > 0:
    attention_ = local_attention
  else:
    attention_ = global_attention

  return attention_(state, prev_weights, hidden_states, encoder, **kwargs)


def multi_attention(state, prev_weights, hidden_states, encoders, **kwargs):
  """
  Same as `attention` except that prev_weights, hidden_states and encoders
  are lists whose length is the number of encoders.
  """
  ds, weights = zip(*[attention(state, weights_, hidden, encoder)
                      for weights_, hidden, encoder in zip(prev_weights, hidden_states, encoders)])

  return tf.concat(1, ds), list(weights)


def decoder(decoder_inputs, initial_state, decoder_name,
            cell, num_decoder_symbols, embedding_size, layers,
            feed_previous=False, output_projection=None,
            **kwargs):
  """ Decoder without attention """
  # FIXME, not the same parameters as `attention_decoder`
  # embeddings = embeddings or {}
  embeddings = {}
  embedding_name = decoder_name
  if embedding_name in embeddings:
    embedding_initializer = embeddings[embedding_name]
    embedding_shape = None
  else:
    embedding_initializer = None
    embedding_shape = [num_decoder_symbols, embedding_size[-1]]

  with tf.device('/cpu:0'):
    embedding = get_variable_unsafe('embedding_{}'.format(decoder_name),
                                    shape=embedding_shape,
                                    initializer=embedding_initializer)

  if layers[-1] > 1:
      cell = rnn_cell.MultiRNNCell([cell] * layers[-1])

  if output_projection is None:
    cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

  if output_projection is not None:
    proj_weights = tf.convert_to_tensor(output_projection[0], dtype=tf.float32)
    proj_weights.get_shape().assert_is_compatible_with([cell.output_size, num_decoder_symbols])
    proj_biases = tf.convert_to_tensor(output_projection[1], dtype=tf.float32)
    proj_biases.get_shape().assert_is_compatible_with([num_decoder_symbols])

  with tf.variable_scope('decoder_{}'.format(decoder_name)):
    def extract_argmax_and_embed(prev):
      if output_projection is not None:
        prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
      return emb_prev

    loop_function = extract_argmax_and_embed if feed_previous else None
    decoder_inputs = [tf.nn.embedding_lookup(embedding, i) for i in decoder_inputs]
    # new parameter: allows encoders and decoder of different sizes,
    # and concatenation of encoders' last states (instead of a sum)
    state = linear_unsafe(initial_state, cell.state_size, False, scope='initial_state_projection')

    outputs = []
    prev = None
    decoder_states = []

    for i, inputs in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with tf.variable_scope('loop_function'):
          inputs = tf.stop_gradient(loop_function(prev))

      cell_output, state = unsafe_decorator(cell)(inputs, state)
      decoder_states.append(state)
      outputs.append(cell_output)

      if loop_function is not None:
        prev = tf.stop_gradient(cell_output)

    return outputs, decoder_states, None


def attention_decoder(decoder_inputs, initial_state, attention_states, encoders, decoder,
                      attention_weights=None, output_projection=None,
                      initial_state_attention=False, dropout=None,
                      feed_previous=False, **kwargs):
  # TODO: dynamic RNN
  if decoder.get('embedding') is not None:
    embedding_initializer = decoder.embedding
    embedding_shape = None
  else:
    # embedding_initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
    embedding_initializer = None
    embedding_shape = [decoder.vocab_size, decoder.embedding_size]

  with tf.device('/cpu:0'):
    embedding = get_variable_unsafe('embedding_{}'.format(decoder.name),
                                    shape=embedding_shape,
                                    initializer=embedding_initializer)

  if decoder.use_lstm:
    cell = rnn_cell.BasicLSTMCell(decoder.cell_size)
  else:
    cell = rnn_cell.GRUCell(decoder.cell_size)

  if dropout is not None:
    cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)

  if decoder.layers > 1:
    cell = rnn_cell.MultiRNNCell([cell] * decoder.layers)

  if output_projection is None:
    cell = rnn_cell.OutputProjectionWrapper(cell, decoder.vocab_size)
    output_size = decoder.vocab_size
  else:
    output_size = cell.output_size

  if output_projection is not None:
    proj_weights = tf.convert_to_tensor(output_projection[0], dtype=tf.float32)
    proj_weights.get_shape().assert_is_compatible_with([cell.output_size, decoder.vocab_size])
    proj_biases = tf.convert_to_tensor(output_projection[1], dtype=tf.float32)
    proj_biases.get_shape().assert_is_compatible_with([decoder.vocab_size])

  with tf.variable_scope('decoder_{}'.format(decoder.name)):
    def extract_argmax_and_embed(prev):
      # loop_function that extracts the symbol from prev and embeds it
      # TODO: check that this does right by the beam-search decoder
      if output_projection is not None:
        prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
      return emb_prev

    loop_function = extract_argmax_and_embed if feed_previous else None
    decoder_inputs = [tf.nn.embedding_lookup(embedding, i) for i in decoder_inputs]

    batch_size = tf.shape(decoder_inputs[0])[0]  # needed for reshaping
    attn_lengths = [states.get_shape()[1].value for states in attention_states]
    attn_size = sum(states.get_shape()[2].value for states in attention_states)

    # to calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before
    hidden_states = [tf.expand_dims(states, 2) for states in attention_states]
    attention_ = functools.partial(multi_attention, hidden_states=hidden_states, encoders=encoders)

    # decoder's first state is the encoder's last state
    # however, their shapes don't necessarily match (multiple encoders, non-matching layers, etc.)
    # TODO: should be a parameter of the encoder rather than the decoder
    if initial_state.get_shape()[1] == cell.state_size:
      state = initial_state
    else:
      # FIXME (projection seems to give worse results)
      # TODO: see other methods (e.g. summing)
      # does it make sense that the first state of the decoder is the last state of the encoder?
      state = linear_unsafe(initial_state, cell.state_size, False, scope='initial_state_projection')

    outputs = []
    all_attention_weights = []
    prev = None
    decoder_states = []

    if attention_weights is None:
      attention_weights = [tf.zeros(tf.pack([batch_size, length])) for length in attn_lengths]

    if initial_state_attention:
      # for beam-search decoder (feed_dict substitution)
      all_attention_weights.append(attention_weights)
      # 09/09: attention model using cell output instead of full state
      # just getting the last output of each encoder and concatenating them
      # TODO: projection
      # FIXME: this won't work with the beam-search decoder (as it substitutes the initial_state parameter)
      # cell_output = tf.concat(2, [tf.slice(states, (0, length - 1, 0), (-1, 1, -1))
      #                             for length, states in zip(attn_lengths, attention_states)])
      # cell_output = tf.reshape(cell_output, [-1, cell.output_size])
      # attns, attention_weights = attention_(cell_output, prev_weights=attention_weights)
      attns, attention_weights = attention_(state, prev_weights=attention_weights)
    else:
      attns = tf.zeros(tf.pack([batch_size, attn_size]), dtype=tf.float32)
      attns.set_shape([None, attn_size])

    for i, inputs in enumerate(decoder_inputs):
      # if loop_function is set, we use it instead of decoder_inputs
      if loop_function is not None and prev is not None:  # TODO: check this
        with tf.variable_scope('loop_function'):
          inputs = tf.stop_gradient(loop_function(prev))

      # merge input and previous attentions into one vector of the right size
      input_size = inputs.get_shape().with_rank(2)[1]
      x = linear_unsafe([inputs, attns], input_size, True)

      # run the RNN
      cell_output, state = unsafe_decorator(cell)(x, state)
      all_attention_weights.append(attention_weights)
      decoder_states.append(state)

      # run the attention mechanism
      # attns, attention_weights = attention_(cell_output, prev_weights=attention_weights)
      attns, attention_weights = attention_(state, prev_weights=attention_weights)

      # 09/09: removed this bullshit projection
      if output_projection is None:
        output = cell_output
      else:
        with tf.variable_scope('attention_output_projection'):
          # FIXME: where does this come from? This greatly impacts performance
          output = linear_unsafe([cell_output, attns], output_size, True)

      outputs.append(output)

      if loop_function is not None:
        # we do not propagate gradients over the loop function
        prev = tf.stop_gradient(output)

    return outputs, decoder_states, all_attention_weights


def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, softmax_loss_function=None, **kwargs):
  attention_states, encoder_state = encoder_with_buckets(encoder_inputs, buckets, **kwargs)
  outputs, _, _ = decoder_with_buckets(attention_states, encoder_state, decoder_inputs,
                                       buckets, **kwargs)
  losses = loss_with_buckets(outputs, targets, weights, buckets,
                             softmax_loss_function)
  return outputs, losses


def encoder_with_buckets(encoder_inputs, buckets, **kwargs):
  encoder_inputs_concat = [v for inputs in encoder_inputs for v in inputs]

  attention_states = []
  encoder_state = []

  with tf.op_scope(encoder_inputs_concat, 'model_with_buckets'):
    for j, bucket in enumerate(buckets):
      with tf.variable_scope(tf.get_variable_scope()):
        encoder_sizes = bucket[:-1]

        encoder_inputs_trunc = [v[:size] for size, v in zip(encoder_sizes, encoder_inputs)]

        bucket_attention_states, bucket_encoder_state = multi_encoder(
          encoder_inputs_trunc, **kwargs)
        attention_states.append(bucket_attention_states)
        encoder_state.append(bucket_encoder_state)

  return attention_states, encoder_state


def decoder_with_buckets(attention_states, encoder_state, decoder_inputs,
                         buckets, no_attention=False, **kwargs):
  outputs = []
  states = []
  attention_weights = []

  decoder_ = decoder if no_attention else attention_decoder

  with tf.op_scope(decoder_inputs, 'model_with_buckets'):
    for bucket, bucket_attention_states, bucket_encoder_state in zip(buckets,
                                                                     attention_states,
                                                                     encoder_state):
      with tf.variable_scope(tf.get_variable_scope(), ):
        decoder_size = bucket[-1]

        # decoder_inputs, initial_state, attention_states
        bucket_outputs, bucket_states, bucket_attention_weights = decoder_(
          decoder_inputs=decoder_inputs[:decoder_size],
          initial_state=bucket_encoder_state,
          attention_states=bucket_attention_states, **kwargs)

        outputs.append(bucket_outputs)
        states.append(bucket_states)
        attention_weights.append(bucket_attention_weights)

  return outputs, states, attention_weights


def loss_with_buckets(outputs, targets, weights, buckets, softmax_loss_function=None):
  losses = []

  with tf.op_scope(targets + weights, 'model_with_buckets'):
    for bucket, bucket_outputs in zip(buckets, outputs):
      with tf.variable_scope(tf.get_variable_scope()):
        decoder_size = bucket[-1]

        losses.append(seq2seq.sequence_loss(
          bucket_outputs, targets[:decoder_size], weights[:decoder_size],
          softmax_loss_function=softmax_loss_function))

  return losses
