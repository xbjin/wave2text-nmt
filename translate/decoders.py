from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import functools
import math
from tensorflow.python.ops import rnn_cell, rnn
from translate.rnn import multi_rnn, multi_bidirectional_rnn


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
linear_unsafe = unsafe_decorator(rnn_cell._linear)
multi_rnn_unsafe = unsafe_decorator(multi_rnn)
multi_bidirectional_rnn_unsafe = unsafe_decorator(multi_bidirectional_rnn)


def multi_encoder(encoder_inputs, encoders, encoder_input_length=None, dropout=None,
                  **kwargs):
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
          cell = rnn_cell.BasicLSTMCell(encoder.cell_size)
        else:
          cell = rnn_cell.GRUCell(encoder.cell_size)

        if dropout is not None:
          cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)

        embedding = embedding_variables[i]
        if embedding is not None:
          fn = lambda x: tf.nn.embedding_lookup(embedding, x)
          encoder_inputs_ = tf.map_fn(fn, encoder_inputs_, dtype=tf.float32)

        encoder_inputs_ = tf.transpose(encoder_inputs_, perm=[1, 0, 2])   # put batch_size first
        sequence_length = encoder_input_length_
        parameters = dict(
          inputs=encoder_inputs_, sequence_length=sequence_length,
          time_pooling=encoder.time_pooling, pooling_avg=encoder.pooling_avg, dtype=tf.float32
        )

        if encoder.bidir:
          encoder_outputs_, _, encoder_state_ = multi_bidirectional_rnn_unsafe(
            cells=[(cell, cell)] * encoder.layers, **parameters)
        else:
          encoder_outputs_, encoder_state_ = multi_rnn_unsafe(
            cells=[cell] * encoder.layers, **parameters)

        if encoder.bidir:  # map to correct output dimension
          # there is not tensor product operation, so we need to flatten our tensor to
          # a matrix to perform a dot product
          shape = tf.shape(encoder_outputs_)
          batch_size = shape[0]
          seq_len = shape[1]
          dim = encoder_outputs_.get_shape()[2]
          outputs_ = tf.reshape(encoder_outputs_, tf.pack([tf.mul(batch_size, seq_len), dim]))
          outputs_ = linear_unsafe(outputs_, cell.output_size, False, scope='bidir_projection')
          encoder_outputs_ = tf.reshape(outputs_, tf.pack([batch_size, seq_len, cell.output_size]))

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
  attn_length = tf.shape(hidden)[1]
  attn_size = hidden.get_shape()[3].value
  batch_size = tf.shape(hidden)[0]

  filter_shape = [attention_filter_length * 2 + 1, 1, 1, attention_filters]
  filter_ = get_variable_unsafe('filter_{}'.format(name), filter_shape)
  u = get_variable_unsafe('U_{}'.format(name), [attention_filters, attn_size])
  prev_weights = tf.reshape(prev_weights, tf.pack([batch_size, attn_length, 1, 1]))
  conv = tf.nn.conv2d(prev_weights, filter_, [1, 1, 1, 1], 'SAME')
  shape = tf.pack([tf.mul(batch_size, attn_length), attention_filters])
  conv = tf.reshape(conv, shape)
  z = tf.matmul(conv, u)
  z = tf.reshape(z, tf.pack([batch_size, attn_length, 1, attn_size]))

  y = linear_unsafe(state, attn_size, True)
  y = tf.reshape(y, [-1, 1, 1, attn_size])

  k = get_variable_unsafe('W_{}'.format(name), [1, 1, attn_size, attn_size])
  f = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], 'SAME')

  v = get_variable_unsafe('V_{}'.format(name), [attn_size])
  s = f + y + z
  return tf.reduce_sum(v * tf.tanh(s), [2, 3])


def global_attention(state, prev_weights, hidden_states, encoder, **kwargs):
  with tf.variable_scope('attention'):

    compute_energy_ = compute_energy_with_filter if encoder.attention_filters > 0 else compute_energy
    e = compute_energy_(hidden_states, state, encoder.name,
                        prev_weights=prev_weights, attention_filters=encoder.attention_filters,
                        attention_filter_length=encoder.attention_filter_length)
    weights = tf.nn.softmax(e)

    shape = tf.shape(weights)
    shape = tf.pack([shape[0], shape[1], 1, 1])

    weighted_average = tf.reduce_sum(tf.reshape(weights, shape) * hidden_states, [1, 2])
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


def decoder(decoder_inputs, initial_state, decoder,
            decoder_input_length=None, output_projection=None, dropout=None,
            feed_previous=False, parallel_iterations=32, **kwargs):
  if decoder.get('embedding') is not None:
    embedding_initializer = decoder.embedding
    embedding_shape = None
  else:
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
      if output_projection is not None:
        prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
      return emb_prev

    loop_function = extract_argmax_and_embed if feed_previous else None

    fn = lambda x: tf.nn.embedding_lookup(embedding, x)
    decoder_inputs = tf.map_fn(fn, decoder_inputs, dtype=tf.float32)

    # if initial_state.get_shape()[1] == cell.state_size:
    #   state = initial_state
    # else:
    # TODO: optional
    state = linear_unsafe(initial_state, cell.state_size, False, scope='initial_state_projection')

    sequence_length = decoder_input_length
    if sequence_length is not None:
      sequence_length = tf.to_int32(sequence_length)
      min_sequence_length = tf.reduce_min(sequence_length)
      max_sequence_length = tf.reduce_max(sequence_length)

    time = tf.constant(0, dtype=tf.int32, name="time")

    input_shape = tf.shape(decoder_inputs)
    time_steps = input_shape[0]
    batch_size = input_shape[1]
    output_size = cell.output_size   # FIXME  (when output_projection is None)
    state_size = cell.state_size

    zero_output = tf.zeros(tf.pack([batch_size, cell.output_size]), tf.float32)

    state_ta = tf.TensorArray(dtype=tf.float32, size=time_steps)
    output_ta = tf.TensorArray(dtype=tf.float32, size=time_steps, clear_after_read=False)
    input_ta = tf.TensorArray(dtype=tf.float32, size=time_steps).unpack(decoder_inputs)

    def _time_step(time, state, output_ta_t, state_ta_t):
      input_t = input_ta.read(time)
      # restore some shape information
      input_t.set_shape(decoder_inputs.get_shape()[1:])

      if loop_function is not None:
        input_t = tf.cond(time > 0,
                          lambda: tf.stop_gradient(loop_function(output_ta_t.read(time - 1))),
                          lambda: input_t)

      # TODO: optional
      x = linear_unsafe([input_t], input_t.get_shape()[1], True)
      call_cell = lambda: unsafe_decorator(cell)(x, state)

      if sequence_length is not None:
        output, new_state = rnn._rnn_step(
          time=time,
          sequence_length=sequence_length,
          min_sequence_length=min_sequence_length,
          max_sequence_length=max_sequence_length,
          zero_output=zero_output,
          state=state,
          call_cell=call_cell,
          state_size=state_size,
          skip_conditionals=True)
      else:
        output, new_state = call_cell()

      state_ta_t = state_ta_t.write(time, new_state)

      # TODO: optional
      if output_projection is not None:
        with tf.variable_scope('output_projection'):
          output = linear_unsafe([output], output_size, True)

      output_ta_t = output_ta_t.write(time, output)
      return time + 1, new_state, output_ta_t, state_ta_t

    _, _, output_final_ta, state_final_ta = tf.while_loop(
      cond=lambda time, *_: time < time_steps,
      body=_time_step,
      loop_vars=(time, state, output_ta, state_ta),
      parallel_iterations=parallel_iterations,
      swap_memory=False)

    outputs = output_final_ta.pack()
    decoder_states = state_final_ta.pack()

    return outputs, decoder_states, None


def attention_decoder(decoder_inputs, initial_state, attention_states, encoders, decoder,
                      decoder_input_length=None, attention_weights=None, output_projection=None,
                      initial_state_attention=False, dropout=None,
                      feed_previous=0.0, parallel_iterations=32, **kwargs):
  if decoder.get('embedding') is not None:
    embedding_initializer = decoder.embedding
    embedding_shape = None
  else:
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
      if output_projection is not None:
        prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
      return emb_prev

    fn = lambda x: tf.nn.embedding_lookup(embedding, x)
    decoder_inputs = tf.map_fn(fn, decoder_inputs, dtype=tf.float32)

    attn_lengths = [tf.shape(states)[1] for states in attention_states]
    attn_size = sum(states.get_shape()[2].value for states in attention_states)

    hidden_states = [tf.expand_dims(states, 2) for states in attention_states]
    attention_ = functools.partial(multi_attention, hidden_states=hidden_states, encoders=encoders)

    if initial_state.get_shape()[1] == cell.state_size:
      state = initial_state
    else:
      state = linear_unsafe(initial_state, cell.state_size, False, scope='initial_state_projection')

    sequence_length = decoder_input_length
    if sequence_length is not None:
      sequence_length = tf.to_int32(sequence_length)
      min_sequence_length = tf.reduce_min(sequence_length)
      max_sequence_length = tf.reduce_max(sequence_length)

    time = tf.constant(0, dtype=tf.int32, name="time")

    input_shape = tf.shape(decoder_inputs)
    time_steps = input_shape[0]
    batch_size = input_shape[1]
    output_size = cell.output_size   # FIXME  (when output_projection is None)
    state_size = cell.state_size

    zero_output = tf.zeros(tf.pack([batch_size, cell.output_size]), tf.float32)

    state_ta = tf.TensorArray(dtype=tf.float32, size=time_steps)
    output_ta = tf.TensorArray(dtype=tf.float32, size=time_steps, clear_after_read=False)
    input_ta = tf.TensorArray(dtype=tf.float32, size=time_steps).unpack(decoder_inputs)
    attn_weights_ta = tf.TensorArray(dtype=tf.float32, size=time_steps)

    if attention_weights is None:
      attention_weights = [tf.zeros(tf.pack([batch_size, length])) for length in attn_lengths]

    if initial_state_attention:
      attns, attention_weights = attention_(state, prev_weights=attention_weights)
    else:
      attns = tf.zeros(tf.pack([batch_size, attn_size]), dtype=tf.float32)
      attns.set_shape([None, attn_size])

    def _time_step(time, state, attns, attn_weights, output_ta_t, state_ta_t, attn_weights_ta_t):
      input_t = input_ta.read(time)
      # restore some shape information
      input_t.set_shape(decoder_inputs.get_shape()[1:])

      r = tf.random_uniform([])

      input_t = tf.cond(tf.logical_and(time > 0, r < feed_previous),
                        lambda: tf.stop_gradient(extract_argmax_and_embed(output_ta_t.read(time - 1))),
                        lambda: input_t)

      x = linear_unsafe([input_t, attns], input_t.get_shape()[1], True)
      call_cell = lambda: unsafe_decorator(cell)(x, state)

      if sequence_length is not None:
        output, new_state = rnn._rnn_step(
          time=time,
          sequence_length=sequence_length,
          min_sequence_length=min_sequence_length,
          max_sequence_length=max_sequence_length,
          zero_output=zero_output,
          state=state,
          call_cell=call_cell,
          state_size=state_size,
          skip_conditionals=True)
      else:
        output, new_state = call_cell()

      state_ta_t = state_ta_t.write(time, new_state)
      attn_weights_ta_t.write(time, attn_weights)
      new_attns, new_attn_weights = attention_(new_state, prev_weights=attn_weights)

      if output_projection is not None:
        with tf.variable_scope('attention_output_projection'):
          output = linear_unsafe([output, new_attns], output_size, True)

      output_ta_t = output_ta_t.write(time, output)
      return time + 1, new_state, new_attns, new_attn_weights, output_ta_t, state_ta_t, attn_weights_ta_t

    _, _, _, _, output_final_ta, state_final_ta, attn_weights_final = tf.while_loop(
      cond=lambda time, *_: time < time_steps,
      body=_time_step,
      loop_vars=(time, state, attns, attention_weights, output_ta, state_ta, attn_weights_ta),
      parallel_iterations=parallel_iterations,
      swap_memory=False)

    outputs = output_final_ta.pack()
    decoder_states = state_final_ta.pack()
    attention_weights = tf.concat(0, [tf.expand_dims(attention_weights, 0),
                                      attn_weights_final.pack()])

    return outputs, decoder_states, attention_weights


def beam_search_decoder(decoder_input, state, attention_states, encoders, decoder,
                        attention_weights=None, output_projection=None,
                        initial_state_attention=False, dropout=None, **kwargs):
  if decoder.get('embedding') is not None:
    embedding_initializer = decoder.embedding
    embedding_shape = None
  else:
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
    proj_weights = tf.convert_to_tensor(output_projection[0], dtype=tf.float32)
    proj_weights.get_shape().assert_is_compatible_with([cell.output_size, decoder.vocab_size])
    proj_biases = tf.convert_to_tensor(output_projection[1], dtype=tf.float32)
    proj_biases.get_shape().assert_is_compatible_with([decoder.vocab_size])

  with tf.variable_scope('decoder_{}'.format(decoder.name)):
    decoder_input = tf.nn.embedding_lookup(embedding, decoder_input)

    attn_lengths = [tf.shape(states)[1] for states in attention_states]
    attn_size = sum(states.get_shape()[2].value for states in attention_states)
    hidden_states = [tf.expand_dims(states, 2) for states in attention_states]
    attention_ = functools.partial(multi_attention, hidden_states=hidden_states, encoders=encoders)

    if state.get_shape()[1] != cell.state_size:  # FIXME: broken with beam-search decoder
      state = linear_unsafe(state, cell.state_size, False, scope='initial_state_projection')

    batch_size = tf.shape(decoder_input)[0]

    if attention_weights is None:
      attention_weights = [tf.zeros(tf.pack([batch_size, length])) for length in attn_lengths]

    if initial_state_attention:
      attns, attention_weights = attention_(state, prev_weights=attention_weights)
    else:
      attns = tf.zeros(tf.pack([batch_size, attn_size]), dtype=tf.float32)
      attns.set_shape([None, attn_size])

    input_size = decoder_input.get_shape()[1]
    x = linear_unsafe([decoder_input, attns], input_size, True)
    cell_output, state = unsafe_decorator(cell)(x, state)
    attns, attention_weights = attention_(state, prev_weights=attention_weights)
    if output_projection is None:
      output = cell_output
    else:
      with tf.variable_scope('attention_output_projection'):
        output = linear_unsafe([cell_output, attns], output_size, True)
    return output, state, attention_weights


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
  with tf.op_scope([logits, targets, weights], name, "sequence_loss"):

    time = tf.constant(0, dtype=tf.int32, name="time")
    time_steps = tf.shape(targets)[0]

    logits_ta = tf.TensorArray(dtype=tf.float32, size=tf.shape(logits)[0]).unpack(logits)
    targets_ta = tf.TensorArray(dtype=tf.int32, size=tf.shape(targets)[0]).unpack(targets)
    weights_ta = tf.TensorArray(dtype=tf.float32, size=tf.shape(weights)[0]).unpack(weights)
    log_perp_ta = tf.TensorArray(dtype=tf.float32, size=time_steps)

    def _time_step(time, log_perp_ta_t):
      logit = logits_ta.read(time)
      target = targets_ta.read(time)
      weight = weights_ta.read(time)

      if softmax_loss_function is None:
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logit, target)
      else:
        crossent = softmax_loss_function(logit, target)
      log_perp_ta_t = log_perp_ta_t.write(time, crossent * weight)
      return time + 1, log_perp_ta_t

    _, log_perp_final = tf.while_loop(
      cond=lambda time, *_: time < time_steps,
      body=_time_step,
      loop_vars=(time, log_perp_ta),
      parallel_iterations=1,
      swap_memory=False)

    log_perp = log_perp_final.pack()
    log_perp = tf.reduce_sum(log_perp, 0)

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
