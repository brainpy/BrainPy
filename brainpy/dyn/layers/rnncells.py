# -*- coding: utf-8 -*-

import warnings
from typing import Union, Callable

import brainpy.math as bm
from brainpy.dyn.base import DynamicalSystem
from brainpy.initialize import (XavierNormal,
                                ZeroInit,
                                Orthogonal,
                                parameter,
                                variable,
                                Initializer)
from brainpy.modes import Mode, TrainingMode, training
from brainpy.tools.checking import (check_integer,
                                    check_initializer)
from brainpy.types import Array

__all__ = [
  'RNNCell', 'GRUCell', 'LSTMCell',

  # deprecated
  'VanillaRNN', 'GRU', 'LSTM',
]


class RecurrentCell(DynamicalSystem):
  def __init__(
      self,
      num_out: int,
      state_initializer: Union[Array, Callable, Initializer] = ZeroInit(),
      mode: Mode = training,
      train_state: bool = False,
      name: str = None
  ):
    super(RecurrentCell, self).__init__(mode=mode, name=name)

    # parameters
    self._state_initializer = state_initializer
    check_initializer(state_initializer, 'state_initializer', allow_none=False)
    self.num_out = num_out
    check_integer(num_out, 'num_out', min_bound=1, allow_none=False)
    self.train_state = train_state


class RNNCell(RecurrentCell):
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
      state_initializer: Union[Array, Callable, Initializer] = ZeroInit(),
      Wi_initializer: Union[Array, Callable, Initializer] = XavierNormal(),
      Wh_initializer: Union[Array, Callable, Initializer] = XavierNormal(),
      b_initializer: Union[Array, Callable, Initializer] = ZeroInit(),
      activation: str = 'relu',
      mode: Mode = training,
      train_state: bool = False,
      name: str = None,
  ):
    super(RNNCell, self).__init__(num_out=num_out,
                                  state_initializer=state_initializer,
                                  train_state=train_state,
                                  mode=mode,
                                  name=name)

    # parameters
    self.num_in = num_in
    check_integer(num_in, 'num_in', min_bound=1, allow_none=False)

    # initializers
    self._Wi_initializer = Wi_initializer
    self._Wh_initializer = Wh_initializer
    self._b_initializer = b_initializer
    check_initializer(Wi_initializer, 'wi_initializer', allow_none=False)
    check_initializer(Wh_initializer, 'wh_initializer', allow_none=False)
    check_initializer(b_initializer, 'b_initializer', allow_none=True)

    # activation function
    self.activation = bm.activations.get(activation)

    # weights
    self.Wi = parameter(self._Wi_initializer, (num_in, self.num_out))
    self.Wh = parameter(self._Wh_initializer, (self.num_out, self.num_out))
    self.b = parameter(self._b_initializer, (self.num_out,))
    if isinstance(self.mode, TrainingMode):
      self.Wi = bm.TrainVar(self.Wi)
      self.Wh = bm.TrainVar(self.Wh)
      self.b = None if (self.b is None) else bm.TrainVar(self.b)

    # state
    self.state = variable(bm.zeros, mode, self.num_out)
    if train_state and isinstance(self.mode, TrainingMode):
      self.state2train = bm.TrainVar(parameter(state_initializer, (self.num_out,), allow_none=False))
      self.state[:] = self.state2train

  def reset_state(self, batch_size=None):
    self.state.value = parameter(self._state_initializer, (batch_size, self.num_out), allow_none=False)
    if self.train_state:
      self.state2train.value = parameter(self._state_initializer, self.num_out, allow_none=False)
      self.state[:] = self.state2train

  def update(self, sha, x):
    h = x @ self.Wi
    h += self.state.value @ self.Wh
    if self.b is not None:
      h += self.b
    self.state.value = self.activation(h)
    return self.state.value


class GRUCell(RecurrentCell):
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
      Wi_initializer: Union[Array, Callable, Initializer] = Orthogonal(),
      Wh_initializer: Union[Array, Callable, Initializer] = Orthogonal(),
      b_initializer: Union[Array, Callable, Initializer] = ZeroInit(),
      state_initializer: Union[Array, Callable, Initializer] = ZeroInit(),
      activation: str = 'tanh',
      mode: Mode = training,
      train_state: bool = False,
      name: str = None,
  ):
    super(GRUCell, self).__init__(num_out=num_out,
                                  state_initializer=state_initializer,
                                  train_state=train_state,
                                  mode=mode,
                                  name=name)
    # parameters
    self.num_in = num_in
    check_integer(num_in, 'num_in', min_bound=1, allow_none=False)

    # initializers
    self._Wi_initializer = Wi_initializer
    self._Wh_initializer = Wh_initializer
    self._b_initializer = b_initializer
    check_initializer(Wi_initializer, 'Wi_initializer', allow_none=False)
    check_initializer(Wh_initializer, 'Wh_initializer', allow_none=False)
    check_initializer(b_initializer, 'b_initializer', allow_none=True)

    # activation function
    self.activation = bm.activations.get(activation)

    # weights
    self.Wi = parameter(self._Wi_initializer, (num_in, self.num_out * 3))
    self.Wh = parameter(self._Wh_initializer, (self.num_out, self.num_out * 3))
    self.b = parameter(self._b_initializer, (self.num_out * 3,))
    if isinstance(self.mode, TrainingMode):
      self.Wi = bm.TrainVar(self.Wi)
      self.Wh = bm.TrainVar(self.Wh)
      self.b = bm.TrainVar(self.b) if (self.b is not None) else None

    # state
    self.state = variable(bm.zeros, mode, self.num_out)
    if train_state and isinstance(self.mode, TrainingMode):
      self.state2train = bm.TrainVar(parameter(state_initializer, (self.num_out,), allow_none=False))
      self.state[:] = self.state2train

  def reset_state(self, batch_size=None):
    self.state.value = parameter(self._state_initializer, (batch_size, self.num_out), allow_none=False)
    if self.train_state:
      self.state2train.value = parameter(self._state_initializer, self.num_out, allow_none=False)
      self.state[:] = self.state2train

  def update(self, sha, x):
    gates_x = bm.matmul(x, self.Wi)
    zr_x, a_x = bm.split(gates_x, indices_or_sections=[2 * self.num_out], axis=-1)
    w_h_z, w_h_a = bm.split(self.Wh, indices_or_sections=[2 * self.num_out], axis=-1)
    zr_h = bm.matmul(self.state, w_h_z)
    zr = zr_x + zr_h
    has_bias = (self.b is not None)
    if has_bias:
      b_z, b_a = bm.split(self.b, indices_or_sections=[2 * self.num_out], axis=0)
      zr += bm.broadcast_to(b_z, zr_h.shape)
    z, r = bm.split(bm.sigmoid(zr), indices_or_sections=2, axis=-1)
    a_h = bm.matmul(r * self.state, w_h_a)
    if has_bias:
      a = self.activation(a_x + a_h + bm.broadcast_to(b_a, a_h.shape))
    else:
      a = self.activation(a_x + a_h)
    next_state = (1 - z) * self.state + z * a
    self.state.value = next_state
    return self.state.value


class LSTMCell(RecurrentCell):
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
      Wi_initializer: Union[Array, Callable, Initializer] = XavierNormal(),
      Wh_initializer: Union[Array, Callable, Initializer] = XavierNormal(),
      b_initializer: Union[Array, Callable, Initializer] = ZeroInit(),
      state_initializer: Union[Array, Callable, Initializer] = ZeroInit(),
      activation: str = 'tanh',
      mode: Mode = training,
      train_state: bool = False,
      name: str = None,
  ):
    super(LSTMCell, self).__init__(num_out=num_out,
                                   state_initializer=state_initializer,
                                   train_state=train_state,
                                   mode=mode,
                                   name=name)
    # parameters
    self.num_in = num_in
    check_integer(num_in, 'num_in', min_bound=1, allow_none=False)

    # initializers
    self._state_initializer = state_initializer
    self._Wi_initializer = Wi_initializer
    self._Wh_initializer = Wh_initializer
    self._b_initializer = b_initializer
    check_initializer(Wi_initializer, 'wi_initializer', allow_none=False)
    check_initializer(Wh_initializer, 'wh_initializer', allow_none=False)
    check_initializer(b_initializer, 'b_initializer', allow_none=True)
    check_initializer(state_initializer, 'state_initializer', allow_none=False)

    # activation function
    self.activation = bm.activations.get(activation)

    # weights
    self.Wi = parameter(self._Wi_initializer, (num_in, self.num_out * 4))
    self.Wh = parameter(self._Wh_initializer, (self.num_out, self.num_out * 4))
    self.b = parameter(self._b_initializer, (self.num_out * 4,))
    if isinstance(self.mode, TrainingMode):
      self.Wi = bm.TrainVar(self.Wi)
      self.Wh = bm.TrainVar(self.Wh)
      self.b = None if (self.b is None) else bm.TrainVar(self.b)

    # state
    self.state = variable(bm.zeros, mode, self.num_out * 2)
    if train_state and isinstance(self.mode, TrainingMode):
      self.state2train = bm.TrainVar(parameter(state_initializer, (self.num_out * 2,), allow_none=False))
      self.state[:] = self.state2train

  def reset_state(self, batch_size=None):
    self.state.value = parameter(self._state_initializer, (batch_size, self.num_out * 2), allow_none=False)
    if self.train_state:
      self.state2train.value = parameter(self._state_initializer, self.num_out * 2, allow_none=False)
      self.state[:] = self.state2train

  def update(self, sha, x):
    h, c = bm.split(self.state, 2, axis=-1)
    gated = x @ self.Wi
    if self.b is not None:
      gated += self.b
    gated += h @ self.Wh
    i, g, f, o = bm.split(gated, indices_or_sections=4, axis=-1)
    c = bm.sigmoid(f + 1.) * c + bm.sigmoid(i) * self.activation(g)
    h = bm.sigmoid(o) * self.activation(c)
    self.state.value = bm.concatenate([h, c], axis=-1)
    return h

  @property
  def h(self):
    """Hidden state."""
    return bm.split(self.state, 2, axis=-1)[0]

  @h.setter
  def h(self, value):
    if self.state is None:
      raise ValueError('Cannot set "h" state. Because the state is not initialized.')
    self.state[:self.state.shape[0] // 2, :] = value

  @property
  def c(self):
    """Memory cell."""
    return bm.split(self.state, 2, axis=-1)[1]

  @c.setter
  def c(self, value):
    if self.state is None:
      raise ValueError('Cannot set "c" state. Because the state is not initialized.')
    self.state[self.state.shape[0] // 2:, :] = value


class VanillaRNN(RNNCell):
  """Vanilla RNN.

  .. deprecated:: 2.2.3.4
     Use `RNNCell` instead. `VanillaRNN` will be removed since version 2.4.0.

  """

  def __init__(self, *args, **kwargs):
    super(VanillaRNN, self).__init__(*args, **kwargs)
    warnings.warn('Use "brainpy.layers.RNNCell" instead. '
                  '"brainpy.layers.VanillaRNN" is deprecated and will be removed since 2.4.0.',
                  UserWarning)


class GRU(GRUCell):
  """GRU.

  .. deprecated:: 2.2.3.4
     Use `GRUCell` instead. `GRU` will be removed since version 2.4.0.

  """

  def __init__(self, *args, **kwargs):
    super(GRU, self).__init__(*args, **kwargs)
    warnings.warn('Use "brainpy.layers.GRUCell" instead. '
                  '"brainpy.layers.GRU" is deprecated and will be removed since 2.4.0.',
                  UserWarning)


class LSTM(LSTMCell):
  """LSTM.

  .. deprecated:: 2.2.3.4
     Use `LSTMCell` instead. `LSTM` will be removed since version 2.4.0.

  """

  def __init__(self, *args, **kwargs):
    super(LSTM, self).__init__(*args, **kwargs)
    warnings.warn('Use "brainpy.layers.LSTMCell" instead. '
                  '"brainpy.layers.LSTM" is deprecated and will be removed since 2.4.0.',
                  UserWarning)


class ConvNDLSTMCell(DynamicalSystem):
  pass


class Conv1DLSTMCell(ConvNDLSTMCell):
  pass


class Conv2DLSTMCell(ConvNDLSTMCell):
  pass


class Conv3DLSTMCell(ConvNDLSTMCell):
  pass
