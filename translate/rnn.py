from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn


def multi_bidirectional_rnn(cells, inputs, sequence_length=None,
                            initial_state_fw=None, initial_state_bw=None,
                            dtype=None, parallel_iterations=None,
                            swap_memory=False, time_major=False, scope=None,
                            time_pooling=None, pooling_avg=None, **kwargs):
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
    inputs = tf.concat(2, [inputs_fw, inputs_bw_reversed])

    output_states_fw.append(output_state_fw)
    output_states_bw.append(output_state_bw)

  return inputs, tf.concat(1, output_states_fw), tf.concat(1, output_states_bw)


def multi_rnn(cells, inputs, sequence_length=None, initial_state=None,
              dtype=None, parallel_iterations=None, swap_memory=False,
              time_major=False, scope=None,
              time_pooling=None, pooling_avg=None, **kwargs):
  name = scope or "MultiRNN"

  output_states = []
  for i, cell in enumerate(cells):
    # Forward direction
    import pdb; pdb.set_trace()
    with tf.variable_scope('{}_{}'.format(name, i)) as scope:
      inputs, output_state = rnn.dynamic_rnn(
        cell=cell, inputs=inputs, sequence_length=sequence_length,
        initial_state=initial_state, dtype=dtype,
        parallel_iterations=parallel_iterations, swap_memory=swap_memory,
        time_major=time_major, scope=scope)

    if time_pooling:
      stride = time_pooling[i]

      import pdb; pdb.set_trace()
      inputs = tf.strided_slice(
        inputs, begin=[0, 0, 0], end=[tf.shape(inputs)[0], tf.shape(inputs)[1], inputs.get_shape()[2]], strides=[1, stride, 1]
      )
      import pdb; pdb.set_trace()
    # TODO: pooling_avg

    output_states.append(output_state)

  return inputs, tf.concat(1, output_states)
