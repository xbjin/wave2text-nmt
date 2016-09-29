from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn


def multi_bidirectional_rnn(cells, inputs, sequence_length=None,
                            initial_state_fw=None, initial_state_bw=None,
                            dtype=None, parallel_iterations=None,
                            swap_memory=False, time_major=False, scope=None,
                            time_pooling=None, pooling_avg=None,
                            residual_connections=False, **kwargs):
  if not time_major:
    time_dim = 1
    batch_dim = 0
  else:
    time_dim = 0
    batch_dim = 1

  name = scope or "MultiBiRNN"

  output_states_fw = []
  output_states_bw = []
  for i, (cell_fw, cell_bw) in enumerate(cells):
    # Forward direction
    with tf.variable_scope('{}_FW_{}'.format(name, i)) as fw_scope:
      inputs_fw, output_state_fw = rnn.dynamic_rnn(
        cell=cell_fw, inputs=inputs, sequence_length=sequence_length,
        initial_state=initial_state_fw, dtype=dtype,
        parallel_iterations=parallel_iterations, swap_memory=swap_memory,
        time_major=time_major, scope=fw_scope)

    # Backward direction
    inputs_reversed = tf.reverse_sequence(
      input=inputs, seq_lengths=sequence_length,
      seq_dim=time_dim, batch_dim=batch_dim)

    with tf.variable_scope('{}_BW_{}'.format(name, i)) as bw_scope:
      inputs_bw, output_state_bw = rnn.dynamic_rnn(
        cell=cell_bw, inputs=inputs_reversed, sequence_length=sequence_length,
        initial_state=initial_state_bw, dtype=dtype,
        parallel_iterations=parallel_iterations, swap_memory=swap_memory,
        time_major=time_major, scope=bw_scope)

    inputs_bw_reversed = tf.reverse_sequence(
      input=inputs_bw, seq_lengths=sequence_length,
      seq_dim=time_dim, batch_dim=batch_dim)
    new_inputs = tf.concat(2, [inputs_fw, inputs_bw_reversed])

    # import pdb; pdb.set_trace()
    if residual_connections and i < len(cells) - 1:
      # inputs = new_inputs + inputs
      inputs = new_inputs
    else:
      inputs = new_inputs

    if time_pooling and i < len(cells) - 1:
      inputs, sequence_length = apply_time_pooling(inputs, sequence_length, time_pooling[i], pooling_avg)

    output_states_fw.append(output_state_fw)
    output_states_bw.append(output_state_bw)

  return inputs, tf.concat(1, output_states_fw), tf.concat(1, output_states_bw)


def multi_rnn(cells, inputs, sequence_length=None, initial_state=None,
              dtype=None, parallel_iterations=None, swap_memory=False,
              time_major=False, scope=None, time_pooling=None, pooling_avg=None,
              residual_connections=False, **kwargs):
  name = scope or "MultiRNN"

  assert time_pooling is None or len(time_pooling) == len(cells) - 1

  output_states = []
  for i, cell in enumerate(cells):
    with tf.variable_scope('{}_{}'.format(name, i)) as scope:
      new_inputs, output_state = rnn.dynamic_rnn(
        cell=cell, inputs=inputs, sequence_length=sequence_length,
        initial_state=initial_state, dtype=dtype,
        parallel_iterations=parallel_iterations, swap_memory=swap_memory,
        time_major=time_major, scope=scope)

    if residual_connections and i < len(cells) - 1:
      inputs = new_inputs + inputs
    else:
      inputs = new_inputs

    if time_pooling and i < len(cells) - 1:
      inputs, sequence_length = apply_time_pooling(inputs, sequence_length, time_pooling[i], pooling_avg)

    output_states.append(output_state)

  return inputs, tf.concat(1, output_states)


def apply_time_pooling(inputs, sequence_length, stride, pooling_avg=False):
  shape = [tf.shape(inputs)[0], tf.shape(inputs)[1], inputs.get_shape()[2].value]

  if pooling_avg:
    inputs_ = [inputs[:, i::stride, :] for i in range(stride)]

    max_len = tf.shape(inputs_[0])[1]
    for k in range(1, stride):
      len_ = tf.shape(inputs_[k])[1]
      paddings = tf.pack([[0, 0], [0, max_len - len_], [0, 0]])
      inputs_[k] = tf.pad(inputs_[k], paddings=paddings)

    inputs = tf.reduce_sum(inputs_, 0) / len(inputs_)
  else:
    inputs = inputs[:, ::stride, :]

  inputs = tf.reshape(inputs, tf.pack([shape[0], tf.shape(inputs)[1], shape[2]]))
  sequence_length = (sequence_length + stride - 1) // stride  # rounding up

  return inputs, sequence_length