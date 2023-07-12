# -*- coding: utf-8 -*-

import warnings
from typing import Union, Callable, Sequence, Optional, Tuple

import jax.numpy as jnp

import brainpy.math as bm
from brainpy.math import activations
from brainpy._src.dnn.base import Layer
from brainpy.check import (is_integer,
                           is_initializer)
from brainpy.initialize import (XavierNormal,
                                ZeroInit,
                                Orthogonal,
                                parameter,
                                variable,
                                Initializer)
from brainpy.types import ArrayType
from .conv import _GeneralConv

__all__ = [
  'RNNCell', 'GRUCell', 'LSTMCell',
  'Conv1dLSTMCell', 'Conv2dLSTMCell', 'Conv3dLSTMCell',
  # deprecated
  'VanillaRNN', 'GRU', 'LSTM',
]


class RNNCell(Layer):
  r"""Basic fully-connected RNN core.

  Given :math:`x_t` and the previous hidden state :math:`h_{t-1}` the
  core computes

  .. math::

     h_t = \mathrm{ReLU}(w_i x_t + b_i + w_h h_{t-1} + b_h)

  The output is equal to the new state, :math:`h_t`.


  Parameters
  ----------
  num_out: int
    The number of hidden unit in the node.
  state_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The state initializer.
  Wi_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The input weight initializer.
  Wh_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The hidden weight initializer.
  b_initializer: optional, callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The bias weight initializer.
  activation: str, callable
    The activation function. It can be a string or a callable function.
    See ``brainpy.math.activations`` for more details.

  """

  def __init__(
      self,
      num_in: int,
      num_out: int,
      state_initializer: Union[ArrayType, Callable, Initializer] = ZeroInit(),
      Wi_initializer: Union[ArrayType, Callable, Initializer] = XavierNormal(),
      Wh_initializer: Union[ArrayType, Callable, Initializer] = XavierNormal(),
      b_initializer: Union[ArrayType, Callable, Initializer] = ZeroInit(),
      activation: str = 'relu',
      mode: bm.Mode = None,
      train_state: bool = False,
      name: str = None,
  ):
    super(RNNCell, self).__init__(mode=mode, name=name)

    # parameters
    self._state_initializer = state_initializer
    is_initializer(state_initializer, 'state_initializer', allow_none=False)
    self.num_out = num_out
    is_integer(num_out, 'num_out', min_bound=1, allow_none=False)
    self.train_state = train_state

    # parameters
    self.num_in = num_in
    is_integer(num_in, 'num_in', min_bound=1, allow_none=False)

    # initializers
    self._Wi_initializer = Wi_initializer
    self._Wh_initializer = Wh_initializer
    self._b_initializer = b_initializer
    is_initializer(Wi_initializer, 'wi_initializer', allow_none=False)
    is_initializer(Wh_initializer, 'wh_initializer', allow_none=False)
    is_initializer(b_initializer, 'b_initializer', allow_none=True)

    # activation function
    self.activation = getattr(activations, activation)

    # weights
    self.Wi = parameter(self._Wi_initializer, (num_in, self.num_out))
    self.Wh = parameter(self._Wh_initializer, (self.num_out, self.num_out))
    self.b = parameter(self._b_initializer, (self.num_out,))
    if isinstance(self.mode, bm.TrainingMode):
      self.Wi = bm.TrainVar(self.Wi)
      self.Wh = bm.TrainVar(self.Wh)
      self.b = None if (self.b is None) else bm.TrainVar(self.b)

    # state
    self.state = variable(jnp.zeros, self.mode, self.num_out)
    if train_state and isinstance(self.mode, bm.TrainingMode):
      self.state2train = bm.TrainVar(parameter(state_initializer, (self.num_out,), allow_none=False))
      self.state[:] = self.state2train

  def reset_state(self, batch_size=None):
    self.state.value = parameter(self._state_initializer, (batch_size, self.num_out), allow_none=False)
    if self.train_state:
      self.state2train.value = parameter(self._state_initializer, self.num_out, allow_none=False)
      self.state[:] = self.state2train

  def update(self, x):
    h = x @ self.Wi
    h += self.state.value @ self.Wh
    if self.b is not None:
      h += self.b
    self.state.value = self.activation(h)
    return self.state.value


class GRUCell(Layer):
  r"""Gated Recurrent Unit.

  The implementation is based on (Chung, et al., 2014) [1]_ with biases.

  Given :math:`x_t` and the previous state :math:`h_{t-1}` the core computes

  .. math::

     \begin{array}{ll}
     z_t &= \sigma(W_{iz} x_t + W_{hz} h_{t-1} + b_z) \\
     r_t &= \sigma(W_{ir} x_t + W_{hr} h_{t-1} + b_r) \\
     a_t &= \tanh(W_{ia} x_t + W_{ha} (r_t \bigodot h_{t-1}) + b_a) \\
     h_t &= (1 - z_t) \bigodot h_{t-1} + z_t \bigodot a_t
     \end{array}

  where :math:`z_t` and :math:`r_t` are reset and update gates.

  The output is equal to the new hidden state, :math:`h_t`.

  Warning: Backwards compatibility of GRU weights is currently unsupported.

  Parameters
  ----------
  num_out: int
    The number of hidden unit in the node.
  state_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The state initializer.
  Wi_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The input weight initializer.
  Wh_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The hidden weight initializer.
  b_initializer: optional, callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The bias weight initializer.
  activation: str, callable
    The activation function. It can be a string or a callable function.
    See ``brainpy.math.activations`` for more details.

  References
  ----------
  .. [1] Chung, J., Gulcehre, C., Cho, K. and Bengio, Y., 2014. Empirical
         evaluation of gated recurrent neural networks on sequence modeling.
         arXiv preprint arXiv:1412.3555.
  """

  def __init__(
      self,
      num_in: int,
      num_out: int,
      Wi_initializer: Union[ArrayType, Callable, Initializer] = Orthogonal(),
      Wh_initializer: Union[ArrayType, Callable, Initializer] = Orthogonal(),
      b_initializer: Union[ArrayType, Callable, Initializer] = ZeroInit(),
      state_initializer: Union[ArrayType, Callable, Initializer] = ZeroInit(),
      activation: str = 'tanh',
      mode: bm.Mode = None,
      train_state: bool = False,
      name: str = None,
  ):
    super(GRUCell, self).__init__(mode=mode, name=name)

    # parameters
    self._state_initializer = state_initializer
    is_initializer(state_initializer, 'state_initializer', allow_none=False)
    self.num_out = num_out
    is_integer(num_out, 'num_out', min_bound=1, allow_none=False)
    self.train_state = train_state
    self.num_in = num_in
    is_integer(num_in, 'num_in', min_bound=1, allow_none=False)

    # initializers
    self._Wi_initializer = Wi_initializer
    self._Wh_initializer = Wh_initializer
    self._b_initializer = b_initializer
    is_initializer(Wi_initializer, 'Wi_initializer', allow_none=False)
    is_initializer(Wh_initializer, 'Wh_initializer', allow_none=False)
    is_initializer(b_initializer, 'b_initializer', allow_none=True)

    # activation function
    self.activation = getattr(activations, activation)

    # weights
    self.Wi = parameter(self._Wi_initializer, (num_in, self.num_out * 3), allow_none=False)
    self.Wh = parameter(self._Wh_initializer, (self.num_out, self.num_out * 3), allow_none=False)
    self.b = parameter(self._b_initializer, (self.num_out * 3,))
    if isinstance(self.mode, bm.TrainingMode):
      self.Wi = bm.TrainVar(self.Wi)
      self.Wh = bm.TrainVar(self.Wh)
      self.b = bm.TrainVar(self.b) if (self.b is not None) else None

    # state
    self.state = variable(jnp.zeros, self.mode, self.num_out)
    if train_state and isinstance(self.mode, bm.TrainingMode):
      self.state2train = bm.TrainVar(parameter(state_initializer, (self.num_out,), allow_none=False))
      self.state[:] = self.state2train

  def reset_state(self, batch_size=None):
    self.state.value = parameter(self._state_initializer, (batch_size, self.num_out), allow_none=False)
    if self.train_state:
      self.state2train.value = parameter(self._state_initializer, self.num_out, allow_none=False)
      self.state[:] = self.state2train

  def update(self, x):
    gates_x = jnp.matmul(x, bm.as_jax(self.Wi))
    zr_x, a_x = jnp.split(gates_x, indices_or_sections=[2 * self.num_out], axis=-1)
    w_h_z, w_h_a = jnp.split(bm.as_jax(self.Wh), indices_or_sections=[2 * self.num_out], axis=-1)
    zr_h = jnp.matmul(bm.as_jax(self.state), w_h_z)
    zr = zr_x + zr_h
    has_bias = (self.b is not None)
    if has_bias:
      b_z, b_a = jnp.split(bm.as_jax(self.b), indices_or_sections=[2 * self.num_out], axis=0)
      zr += jnp.broadcast_to(b_z, zr_h.shape)
    z, r = jnp.split(bm.sigmoid(zr), indices_or_sections=2, axis=-1)
    a_h = jnp.matmul(r * self.state, w_h_a)
    if has_bias:
      a = self.activation(a_x + a_h + jnp.broadcast_to(b_a, a_h.shape))
    else:
      a = self.activation(a_x + a_h)
    next_state = (1 - z) * self.state + z * a
    self.state.value = next_state
    return self.state.value


class LSTMCell(Layer):
  r"""Long short-term memory (LSTM) RNN core.

  The implementation is based on (zaremba, et al., 2014) [1]_. Given
  :math:`x_t` and the previous state :math:`(h_{t-1}, c_{t-1})` the core
  computes

  .. math::

     \begin{array}{ll}
     i_t = \sigma(W_{ii} x_t + W_{hi} h_{t-1} + b_i) \\
     f_t = \sigma(W_{if} x_t + W_{hf} h_{t-1} + b_f) \\
     g_t = \tanh(W_{ig} x_t + W_{hg} h_{t-1} + b_g) \\
     o_t = \sigma(W_{io} x_t + W_{ho} h_{t-1} + b_o) \\
     c_t = f_t c_{t-1} + i_t g_t \\
     h_t = o_t \tanh(c_t)
     \end{array}

  where :math:`i_t`, :math:`f_t`, :math:`o_t` are input, forget and
  output gate activations, and :math:`g_t` is a vector of cell updates.

  The output is equal to the new hidden, :math:`h_t`.

  Notes
  -----

  Forget gate initialization: Following (Jozefowicz, et al., 2015) [2]_ we add 1.0
  to :math:`b_f` after initialization in order to reduce the scale of forgetting in
  the beginning of the training.


  Parameters
  ----------
  num_out: int
    The number of hidden unit in the node.
  state_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The state initializer.
  Wi_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The input weight initializer.
  Wh_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The hidden weight initializer.
  b_initializer: optional, callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The bias weight initializer.
  activation: str, callable
    The activation function. It can be a string or a callable function.
    See ``brainpy.math.activations`` for more details.

  References
  ----------

  .. [1] Zaremba, Wojciech, Ilya Sutskever, and Oriol Vinyals. "Recurrent neural
         network regularization." arXiv preprint arXiv:1409.2329 (2014).
  .. [2] Jozefowicz, Rafal, Wojciech Zaremba, and Ilya Sutskever. "An empirical
         exploration of recurrent network architectures." In International conference
         on machine learning, pp. 2342-2350. PMLR, 2015.
  """

  def __init__(
      self,
      num_in: int,
      num_out: int,
      Wi_initializer: Union[ArrayType, Callable, Initializer] = XavierNormal(),
      Wh_initializer: Union[ArrayType, Callable, Initializer] = XavierNormal(),
      b_initializer: Union[ArrayType, Callable, Initializer] = ZeroInit(),
      state_initializer: Union[ArrayType, Callable, Initializer] = ZeroInit(),
      activation: str = 'tanh',
      mode: bm.Mode = None,
      train_state: bool = False,
      name: str = None,
  ):
    super(LSTMCell, self).__init__(mode=mode, name=name)

    # parameters
    self._state_initializer = state_initializer
    is_initializer(state_initializer, 'state_initializer', allow_none=False)
    self.num_out = num_out
    is_integer(num_out, 'num_out', min_bound=1, allow_none=False)
    self.train_state = train_state
    self.num_in = num_in
    is_integer(num_in, 'num_in', min_bound=1, allow_none=False)

    # initializers
    self._state_initializer = state_initializer
    self._Wi_initializer = Wi_initializer
    self._Wh_initializer = Wh_initializer
    self._b_initializer = b_initializer
    is_initializer(Wi_initializer, 'wi_initializer', allow_none=False)
    is_initializer(Wh_initializer, 'wh_initializer', allow_none=False)
    is_initializer(b_initializer, 'b_initializer', allow_none=True)
    is_initializer(state_initializer, 'state_initializer', allow_none=False)

    # activation function
    self.activation = getattr(activations, activation)

    # weights
    self.Wi = parameter(self._Wi_initializer, (num_in, self.num_out * 4))
    self.Wh = parameter(self._Wh_initializer, (self.num_out, self.num_out * 4))
    self.b = parameter(self._b_initializer, (self.num_out * 4,))
    if isinstance(self.mode, bm.TrainingMode):
      self.Wi = bm.TrainVar(self.Wi)
      self.Wh = bm.TrainVar(self.Wh)
      self.b = None if (self.b is None) else bm.TrainVar(self.b)

    # state
    self.state = variable(jnp.zeros, self.mode, self.num_out * 2)
    if train_state and isinstance(self.mode, bm.TrainingMode):
      self.state2train = bm.TrainVar(parameter(state_initializer, (self.num_out * 2,), allow_none=False))
      self.state[:] = self.state2train

  def reset_state(self, batch_size=None):
    self.state.value = parameter(self._state_initializer, (batch_size, self.num_out * 2), allow_none=False)
    if self.train_state:
      self.state2train.value = parameter(self._state_initializer, self.num_out * 2, allow_none=False)
      self.state[:] = self.state2train

  def update(self, x):
    h, c = jnp.split(self.state.value, 2, axis=-1)
    gated = x @ self.Wi
    if self.b is not None:
      gated += self.b
    gated += h @ self.Wh
    i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)
    c = bm.sigmoid(f + 1.) * c + bm.sigmoid(i) * self.activation(g)
    h = bm.sigmoid(o) * self.activation(c)
    self.state.value = jnp.concatenate([h, c], axis=-1)
    return h

  @property
  def h(self):
    """Hidden state."""
    return jnp.split(self.state.value, 2, axis=-1)[0]

  @h.setter
  def h(self, value):
    if self.state is None:
      raise ValueError('Cannot set "h" state. Because the state is not initialized.')
    self.state[:self.state.shape[0] // 2, :] = value

  @property
  def c(self):
    """Memory cell."""
    return jnp.split(self.state.value, 2, axis=-1)[1]

  @c.setter
  def c(self, value):
    if self.state is None:
      raise ValueError('Cannot set "c" state. Because the state is not initialized.')
    self.state[self.state.shape[0] // 2:, :] = value


class VanillaRNN(RNNCell):
  """Vanilla RNN.

  .. deprecated:: 2.2.3.4
     Use :py:class:`~.RNNCell` instead. :py:class:`~.VanillaRNN` will be removed since version 2.4.0.

  """

  def __init__(self, *args, **kwargs):
    super(VanillaRNN, self).__init__(*args, **kwargs)
    warnings.warn('Use "brainpy.layers.RNNCell" instead. '
                  '"brainpy.layers.VanillaRNN" is deprecated and will be removed since 2.4.0.',
                  UserWarning)


class GRU(GRUCell):
  """GRU.

  .. deprecated:: 2.2.3.4
     Use :py:class:`~.GRUCell` instead. :py:class:`~.GRU` will be removed since version 2.4.0.

  """

  def __init__(self, *args, **kwargs):
    super(GRU, self).__init__(*args, **kwargs)
    warnings.warn('Use "brainpy.layers.GRUCell" instead. '
                  '"brainpy.layers.GRU" is deprecated and will be removed since 2.4.0.',
                  UserWarning)


class LSTM(LSTMCell):
  """LSTM.

  .. deprecated:: 2.2.3.4
     Use :py:class:`~.LSTMCell` instead. :py:class:`~.LSTM` will be removed since version 2.4.0.

  """

  def __init__(self, *args, **kwargs):
    warnings.warn('Use "brainpy.layers.LSTMCell" instead. '
                  '"brainpy.layers.LSTM" is deprecated and will be removed since 2.4.0.',
                  UserWarning)
    super(LSTM, self).__init__(*args, **kwargs)


class _ConvNDLSTMCell(Layer):
  r"""``num_spatial_dims``-D convolutional LSTM.

  The implementation is based on :cite:`xingjian2015convolutional`.
  Given :math:`x_t` and the previous state :math:`(h_{t-1}, c_{t-1})`
  the core computes

  .. math::

     \begin{array}{ll}
     i_t = \sigma(W_{ii} * x_t + W_{hi} * h_{t-1} + b_i) \\
     f_t = \sigma(W_{if} * x_t + W_{hf} * h_{t-1} + b_f) \\
     g_t = \tanh(W_{ig} * x_t + W_{hg} * h_{t-1} + b_g) \\
     o_t = \sigma(W_{io} * x_t + W_{ho} * h_{t-1} + b_o) \\
     c_t = f_t c_{t-1} + i_t g_t \\
     h_t = o_t \tanh(c_t)
     \end{array}

  where :math:`*` denotes the convolution operator; :math:`i_t`,
  :math:`f_t`, :math:`o_t` are input, forget and output gate activations,
  and :math:`g_t` is a vector of cell updates.

  The output is equal to the new hidden state, :math:`h_t`.

  Notes:
    Forget gate initialization:
      Following :cite:`jozefowicz2015empirical` we add 1.0 to :math:`b_f`
      after initialization in order to reduce the scale of forgetting in
      the beginning of the training.
  """

  def __init__(
      self,
      input_shape: Tuple[int, ...],

      # convolution parameters
      num_spatial_dims: int,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Sequence[int]],
      stride: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      lhs_dilation: Union[int, Tuple[int, ...]] = 1,
      rhs_dilation: Union[int, Tuple[int, ...]] = 1,
      groups: int = 1,
      w_initializer: Union[Callable, ArrayType, Initializer] = XavierNormal(),
      b_initializer: Optional[Union[Callable, ArrayType, Initializer]] = ZeroInit(),

      # recurrent parameters
      state_initializer: Union[ArrayType, Callable, Initializer] = ZeroInit(),
      train_state: bool = False,

      # others
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    """Constructs a convolutional LSTM.

    Args:
      num_spatial_dims: Number of spatial dimensions of the input.
      input_shape: Shape of the inputs excluding batch size.
      out_channels: Number of output channels.
      kernel_size: Sequence of kernel sizes (of length ``num_spatial_dims``),
        or an int. ``kernel_shape`` will be expanded to define a kernel size in
        all dimensions.
      name: Name of the module.
    """
    super().__init__(name=name, mode=mode)

    assert self.mode.is_parent_of(bm.TrainingMode, bm.BatchingMode)

    # parameters
    self._state_initializer = state_initializer
    is_initializer(state_initializer, 'state_initializer', allow_none=False)
    self.train_state = train_state
    self.num_spatial_dims = num_spatial_dims
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.input_shape = tuple(input_shape)
    self.input_to_hidden = _GeneralConv(num_spatial_dims=num_spatial_dims,
                                        in_channels=in_channels,
                                        out_channels=out_channels * 4,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        lhs_dilation=lhs_dilation,
                                        rhs_dilation=rhs_dilation,
                                        groups=groups,
                                        w_initializer=w_initializer,
                                        b_initializer=b_initializer, )
    self.hidden_to_hidden = _GeneralConv(num_spatial_dims=num_spatial_dims,
                                         in_channels=out_channels,
                                         out_channels=out_channels * 4,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         lhs_dilation=lhs_dilation,
                                         rhs_dilation=rhs_dilation,
                                         groups=groups,
                                         w_initializer=w_initializer,
                                         b_initializer=b_initializer, )
    self.reset_state()

  def reset_state(self, batch_size: int = 1):
    shape = self.input_shape + (self.out_channels,)
    h = parameter(self._state_initializer, (batch_size,) + shape, allow_none=False)
    c = parameter(self._state_initializer, (batch_size,) + shape, allow_none=False)
    self.h = bm.Variable(h, batch_axis=0)
    self.c = bm.Variable(c, batch_axis=0)
    if self.mode.is_a(bm.TrainingMode) and self.train_state:
      h_to_train = parameter(self._state_initializer, shape, allow_none=False)
      c_to_train = parameter(self._state_initializer, shape, allow_none=False)
      self.h_to_train = bm.TrainVar(h_to_train)
      self.c_to_train = bm.TrainVar(c_to_train)
      self.h[:] = self.h_to_train
      self.c[:] = self.c_to_train

  def update(self, x):
    gates = self.input_to_hidden(x) + self.hidden_to_hidden(self.h)
    i, g, f, o = bm.split(gates, indices_or_sections=4, axis=-1)
    f = bm.sigmoid(f + 1)
    c = f * self.c + bm.sigmoid(i) * bm.tanh(g)
    h = bm.sigmoid(o) * bm.tanh(c)
    self.h.value = h
    self.c.value = c
    return h


class Conv1dLSTMCell(_ConvNDLSTMCell):  # pylint: disable=empty-docstring
  __doc__ = _ConvNDLSTMCell.__doc__.replace("``num_spatial_dims``", "1")

  def __init__(
      self,
      input_shape: Tuple[int, ...],

      # convolution parameters
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Sequence[int]],
      stride: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      lhs_dilation: Union[int, Tuple[int, ...]] = 1,
      rhs_dilation: Union[int, Tuple[int, ...]] = 1,
      groups: int = 1,
      w_initializer: Union[Callable, ArrayType, Initializer] = XavierNormal(),
      b_initializer: Optional[Union[Callable, ArrayType, Initializer]] = ZeroInit(),

      # recurrent parameters
      state_initializer: Union[ArrayType, Callable, Initializer] = ZeroInit(),
      train_state: bool = False,

      # others
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    """Constructs a 1-D convolutional LSTM.

    Args:
      input_shape: Shape of the inputs excluding batch size.
      out_channels: Number of output channels.
      kernel_size: Sequence of kernel sizes (of length 1), or an int.
        ``kernel_shape`` will be expanded to define a kernel size in all
        dimensions.
      name: Name of the module.
    """
    super().__init__(
      num_spatial_dims=1,
      input_shape=input_shape,
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation,
      groups=groups,
      w_initializer=w_initializer,
      b_initializer=b_initializer,
      state_initializer=state_initializer,
      train_state=train_state,
      mode=mode,
      name=name
    )


class Conv2dLSTMCell(_ConvNDLSTMCell):  # pylint: disable=empty-docstring
  __doc__ = _ConvNDLSTMCell.__doc__.replace("``num_spatial_dims``", "2")

  def __init__(
      self,
      input_shape: Tuple[int, ...],

      # convolution parameters
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Sequence[int]],
      stride: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      lhs_dilation: Union[int, Tuple[int, ...]] = 1,
      rhs_dilation: Union[int, Tuple[int, ...]] = 1,
      groups: int = 1,
      w_initializer: Union[Callable, ArrayType, Initializer] = XavierNormal(),
      b_initializer: Optional[Union[Callable, ArrayType, Initializer]] = ZeroInit(),

      # recurrent parameters
      state_initializer: Union[ArrayType, Callable, Initializer] = ZeroInit(),
      train_state: bool = False,

      # others
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    """Constructs a 2-D convolutional LSTM.

    Args:
      input_shape: Shape of the inputs excluding batch size.
      out_channels: Number of output channels.
      kernel_size: Sequence of kernel sizes (of length 2), or an int.
        ``kernel_shape`` will be expanded to define a kernel size in all
        dimensions.
      name: Name of the module.
    """
    super().__init__(
      num_spatial_dims=2,
      input_shape=input_shape,
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation,
      groups=groups,
      w_initializer=w_initializer,
      b_initializer=b_initializer,
      state_initializer=state_initializer,
      train_state=train_state,
      mode=mode,
      name=name
    )


class Conv3dLSTMCell(_ConvNDLSTMCell):  # pylint: disable=empty-docstring
  __doc__ = _ConvNDLSTMCell.__doc__.replace("``num_spatial_dims``", "3")

  def __init__(
      self,
      input_shape: Tuple[int, ...],

      # convolution parameters
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Sequence[int]],
      stride: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      lhs_dilation: Union[int, Tuple[int, ...]] = 1,
      rhs_dilation: Union[int, Tuple[int, ...]] = 1,
      groups: int = 1,
      w_initializer: Union[Callable, ArrayType, Initializer] = XavierNormal(),
      b_initializer: Optional[Union[Callable, ArrayType, Initializer]] = ZeroInit(),

      # recurrent parameters
      state_initializer: Union[ArrayType, Callable, Initializer] = ZeroInit(),
      train_state: bool = False,

      # others
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    """Constructs a 3-D convolutional LSTM.

    Args:
      input_shape: Shape of the inputs excluding batch size.
      out_channels: Number of output channels.
      kernel_size: Sequence of kernel sizes (of length 3), or an int.
        ``kernel_shape`` will be expanded to define a kernel size in all
        dimensions.
      name: Name of the module.
    """
    super().__init__(
      num_spatial_dims=3,
      input_shape=input_shape,
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation,
      groups=groups,
      w_initializer=w_initializer,
      b_initializer=b_initializer,
      state_initializer=state_initializer,
      train_state=train_state,
      mode=mode,
      name=name
    )
