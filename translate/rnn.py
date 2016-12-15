import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell


def multi_bidirectional_rnn(cells, inputs, sequence_length=None, dtype=None, parallel_iterations=None,
                            swap_memory=False, time_major=False, trainable_initial_state=True, **kwargs):
    if not time_major:
        time_dim = 1
        batch_dim = 0
    else:
        time_dim = 0
        batch_dim = 1

    batch_size = tf.shape(inputs)[batch_dim]

    output_states_fw = []
    output_states_bw = []
    for i, (cell_fw, cell_bw) in enumerate(cells):
        # forward direction
        with tf.variable_scope('forward_{}'.format(i + 1)) as fw_scope:
            if trainable_initial_state:
                initial_state = get_variable_unsafe('initial_state', initializer=tf.zeros([cell_fw.state_size]),
                                                    dtype=dtype)
                initial_state = tf.reshape(tf.tile(initial_state, [batch_size]),
                                           shape=[batch_size, cell_fw.state_size])
            else:
                initial_state = None

            inputs_fw, output_state_fw = rnn.dynamic_rnn(
                cell=cell_fw, inputs=inputs, sequence_length=sequence_length, initial_state=initial_state,
                dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                time_major=time_major, scope=fw_scope
            )

        # backward direction
        inputs_reversed = tf.reverse_sequence(
            input=inputs, seq_lengths=sequence_length, seq_dim=time_dim, batch_dim=batch_dim
        )

        with tf.variable_scope('backward_{}'.format(i + 1)) as bw_scope:
            if trainable_initial_state:
                initial_state = get_variable_unsafe('initial_state', initializer=tf.zeros([cell_bw.state_size]),
                                                    dtype=dtype)
                initial_state = tf.reshape(tf.tile(initial_state, [batch_size]),
                                           shape=[batch_size, cell_bw.state_size])
            else:
                initial_state = None

            inputs_bw, output_state_bw = rnn.dynamic_rnn(
                cell=cell_bw, inputs=inputs_reversed, sequence_length=sequence_length, initial_state=initial_state,
                dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory, time_major=time_major,
                scope=bw_scope
            )

        inputs_bw_reversed = tf.reverse_sequence(
            input=inputs_bw, seq_lengths=sequence_length,
            seq_dim=time_dim, batch_dim=batch_dim
        )
        inputs = tf.concat(2, [inputs_fw, inputs_bw_reversed])

        output_states_fw.append(output_state_fw)
        output_states_bw.append(output_state_bw)

    return inputs, tf.concat(1, output_states_fw), tf.concat(1, output_states_bw)


def multi_rnn(cells, inputs, sequence_length=None, dtype=None, parallel_iterations=None, swap_memory=False,
              time_major=False, trainable_initial_state=True, **kwargs):
    batch_size = tf.shape(inputs)[0]     # TODO: Fix time major stuff

    output_states = []
    for i, cell in enumerate(cells):
        with tf.variable_scope('forward_{}'.format(i + 1)) as scope:
            if trainable_initial_state:
                initial_state = get_variable_unsafe('initial_state', initializer=tf.zeros([cell.state_size]),
                                                    dtype=dtype)
                initial_state = tf.reshape(tf.tile(initial_state, [batch_size]),
                                           shape=[batch_size, cell.state_size])
            else:
                initial_state = None

            inputs, output_state = rnn.dynamic_rnn(
                cell=cell, inputs=inputs, sequence_length=sequence_length, initial_state=initial_state, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory, time_major=time_major, scope=scope
            )

        output_states.append(output_state)

    return inputs, tf.concat(1, output_states)


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


class GRUCell(rnn_cell.RNNCell):
    def __init__(self, num_units, activation=tf.nn.tanh, initializer=None):
        self._num_units = num_units
        self._activation = activation
        self._initializer = initializer

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # we start with bias of 1.0 to not reset and not update
            r = tf.nn.sigmoid(
                linear(inputs, self._num_units, True, 1.0, scope='W_r') +
                linear(state, self._num_units, False, scope='U_r', initializer=self._initializer)    # state to gates
            )

            z = tf.nn.sigmoid(
                linear(inputs, self._num_units, True, 1.0, scope='W_z') +
                linear(state, self._num_units, False, scope='U_z', initializer=self._initializer)    # state to gates
            )

            h_ = self._activation(
                linear(inputs, self._num_units, True, scope='W') +
                linear(r * state, self._num_units, False, scope='U', initializer=self._initializer)  # state to state
            )

            new_h = z * state + (1 - z) * h_
        return new_h, new_h


def linear(args, output_size, bias, bias_start=0.0, scope=None, initializer=None):
  """
  Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Same as as `tf.nn.rnn_cell._linear`, with the addition of an `initializer` parameter.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
    initializer: used to initialize W

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (tf.nn.nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not tf.nn.nest.is_sequence(args):
    args = [args]

  # calculate the total size of arguments on dimension 1
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size], dtype=dtype,
                             initializer=initializer)
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable("Bias", [output_size], dtype=dtype,
                                initializer=tf.constant_initializer(bias_start, dtype=dtype))
  return res + bias_term


def orthogonal_initializer(scale=1.0, dtype=tf.float32):
    """Initialize a random orthogonal matrix.

    Only works for 2D arrays.

    Parameters
    ----------
    scale : float, optional
        Multiply the resulting matrix with a scalar. Defaults to 1.
        For a discussion of the importance of scale for training time
        and generalization refer to [Saxe2013]_.

        .. [Saxe2013] Saxe, A.M., McClelland, J.L., Ganguli, S., 2013.,
           *Exact solutions to the nonlinear dynamics of learning in deep
           linear neural networks*,
           arXiv:1312.6120 [cond-mat, q-bio, stat].

    """

    def _initializer(shape, dtype=dtype, partition_info=None):
        if len(shape) != 2:
            raise ValueError

        if shape[0] == shape[1]:
            # For square weight matrices we can simplify the logic
            # and be more exact:
            M = np.random.randn(*shape)
            Q, R = np.linalg.qr(M)
            Q = Q * np.sign(np.diag(R))
            return Q * scale

        M1 = np.random.randn(shape[0], shape[0])
        M2 = np.random.randn(shape[1], shape[1])

        # QR decomposition of matrix with entries in N(0, 1) is random
        Q1, R1 = np.linalg.qr(M1)
        Q2, R2 = np.linalg.qr(M2)
        # Correct that NumPy doesn't force diagonal of R to be non-negative
        Q1 = Q1 * np.sign(np.diag(R1))
        Q2 = Q2 * np.sign(np.diag(R2))

        n_min = min(shape[0], shape[1])
        return tf.constant(np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale, dtype=dtype)
    return _initializer


get_variable_unsafe = unsafe_decorator(tf.get_variable)
GRUCell_unsafe = unsafe_decorator(rnn_cell.GRUCell)
BasicLSTMCell_unsafe = unsafe_decorator(rnn_cell.BasicLSTMCell)
MultiRNNCell_unsafe = unsafe_decorator(rnn_cell.MultiRNNCell)
linear_unsafe = unsafe_decorator(linear)
multi_rnn_unsafe = unsafe_decorator(multi_rnn)
multi_bidirectional_rnn_unsafe = unsafe_decorator(multi_bidirectional_rnn)
