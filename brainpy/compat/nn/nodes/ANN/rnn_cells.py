# -*- coding: utf-8 -*-


from typing import Union, Callable

import brainpy.math as bm
from brainpy.initialize import (XavierNormal,
                                ZeroInit,
                                Uniform,
                                Orthogonal,
                                init_param,
                                Initializer)
from brainpy.compat.nn.base import RecurrentNode
from brainpy.compat.nn.datatypes import MultipleData
from brainpy.tools.checking import (check_integer,
                                    check_initializer,
                                    check_shape_consistency)
from brainpy.types import Tensor

__all__ = [
  'VanillaRNN',
  'GRU',
  'LSTM',
]


class VanillaRNN(RecurrentNode):
  r"""Basic fully-connected RNN core.

  Given :math:`x_t` and the previous hidden state :math:`h_{t-1}` the
  core computes

  .. math::

     h_t = \mathrm{ReLU}(w_i x_t + b_i + w_h h_{t-1} + b_h)

  The output is equal to the new state, :math:`h_t`.


  Parameters
  ----------
  num_unit: int
    The number of hidden unit in the node.
  state_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The state initializer.
  wi_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The input weight initializer.
  wh_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The hidden weight initializer.
  bias_initializer: optional, callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The bias weight initializer.
  activation: str, callable
    The activation function. It can be a string or a callable function.
    See ``brainpy.math.activations`` for more details.
  trainable: bool
    Whether set the node is trainable.

  """
  data_pass = MultipleData('sequence')

  def __init__(
      self,
      num_unit: int,
      state_initializer: Union[Tensor, Callable, Initializer] = Uniform(),
      wi_initializer: Union[Tensor, Callable, Initializer] = XavierNormal(),
      wh_initializer: Union[Tensor, Callable, Initializer] = XavierNormal(),
      bias_initializer: Union[Tensor, Callable, Initializer] = ZeroInit(),
      activation: str = 'relu',
      **kwargs
  ):
    super(VanillaRNN, self).__init__(**kwargs)

    self.num_unit = num_unit
    check_integer(num_unit, 'num_unit', min_bound=1, allow_none=False)
    self.set_output_shape((None, self.num_unit))

    # initializers
    self._state_initializer = state_initializer
    self._wi_initializer = wi_initializer
    self._wh_initializer = wh_initializer
    self._bias_initializer = bias_initializer
    check_initializer(wi_initializer, 'wi_initializer', allow_none=False)
    check_initializer(wh_initializer, 'wh_initializer', allow_none=False)
    check_initializer(state_initializer, 'state_initializer', allow_none=False)
    check_initializer(bias_initializer, 'bias_initializer', allow_none=True)

    # activation function
    self.activation = bm.activations.get(activation)

  def init_ff_conn(self):
    unique_size, free_sizes = check_shape_consistency(self.feedforward_shapes, -1, True)
    assert len(unique_size) == 1, 'Only support data with or without batch size.'
    # weights
    num_input = sum(free_sizes)
    self.Wff = init_param(self._wi_initializer, (num_input, self.num_unit))
    self.Wrec = init_param(self._wh_initializer, (self.num_unit, self.num_unit))
    self.bias = init_param(self._bias_initializer, (self.num_unit,))
    if self.trainable:
      self.Wff = bm.TrainVar(self.Wff)
      self.Wrec = bm.TrainVar(self.Wrec)
      self.bias = None if (self.bias is None) else bm.TrainVar(self.bias)

  def init_fb_conn(self):
    unique_size, free_sizes = check_shape_consistency(self.feedback_shapes, -1, True)
    assert len(unique_size) == 1, 'Only support data with or without batch size.'
    num_feedback = sum(free_sizes)
    # weights
    self.Wfb = init_param(self._wi_initializer, (num_feedback, self.num_unit))
    if self.trainable:
      self.Wfb = bm.TrainVar(self.Wfb)

  def init_state(self, num_batch=1):
    return init_param(self._state_initializer, (num_batch, self.num_unit))

  def forward(self, ff, fb=None, **shared_kwargs):
    ff = bm.concatenate(ff, axis=-1)
    h = ff @ self.Wff
    h += self.state.value @ self.Wrec
    if self.bias is not None:
      h += self.bias
    if fb is not None:
      fb = bm.concatenate(fb, axis=-1)
      h += fb @ self.Wfb
    self.state.value = self.activation(h)
    return self.state.value


class GRU(RecurrentNode):
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
  num_unit: int
    The number of hidden unit in the node.
  state_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The state initializer.
  wi_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The input weight initializer.
  wh_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The hidden weight initializer.
  bias_initializer: optional, callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The bias weight initializer.
  activation: str, callable
    The activation function. It can be a string or a callable function.
    See ``brainpy.math.activations`` for more details.
  trainable: bool
    Whether set the node is trainable.

  References
  ----------
  .. [1] Chung, J., Gulcehre, C., Cho, K. and Bengio, Y., 2014. Empirical
         evaluation of gated recurrent neural networks on sequence modeling.
         arXiv preprint arXiv:1412.3555.
  """
  data_pass = MultipleData('sequence')

  def __init__(
      self,
      num_unit: int,
      wi_initializer: Union[Tensor, Callable, Initializer] = Orthogonal(),
      wh_initializer: Union[Tensor, Callable, Initializer] = Orthogonal(),
      bias_initializer: Union[Tensor, Callable, Initializer] = ZeroInit(),
      state_initializer: Union[Tensor, Callable, Initializer] = ZeroInit(),
      **kwargs
  ):
    super(GRU, self).__init__(**kwargs)

    self.num_unit = num_unit
    check_integer(num_unit, 'num_unit', min_bound=1, allow_none=False)
    self.set_output_shape((None, self.num_unit))

    self._wi_initializer = wi_initializer
    self._wh_initializer = wh_initializer
    self._bias_initializer = bias_initializer
    self._state_initializer = state_initializer
    check_initializer(wi_initializer, 'wi_initializer', allow_none=False)
    check_initializer(wh_initializer, 'wh_initializer', allow_none=False)
    check_initializer(state_initializer, 'state_initializer', allow_none=False)
    check_initializer(bias_initializer, 'bias_initializer', allow_none=True)

  def init_ff_conn(self):
    # data shape
    unique_size, free_sizes = check_shape_consistency(self.feedforward_shapes, -1, True)
    assert len(unique_size) == 1, 'Only support data with or without batch size.'

    # weights
    num_input = sum(free_sizes)
    self.Wi_ff = init_param(self._wi_initializer, (num_input, self.num_unit * 3))
    self.Wh = init_param(self._wh_initializer, (self.num_unit, self.num_unit * 3))
    self.bias = init_param(self._bias_initializer, (self.num_unit * 3,))
    if self.trainable:
      self.Wi_ff = bm.TrainVar(self.Wi_ff)
      self.Wh = bm.TrainVar(self.Wh)
      self.bias = bm.TrainVar(self.bias) if (self.bias is not None) else None

  def init_fb_conn(self):
    unique_size, free_sizes = check_shape_consistency(self.feedback_shapes, -1, True)
    assert len(unique_size) == 1, 'Only support data with or without batch size.'
    num_feedback = sum(free_sizes)
    # weights
    self.Wi_fb = init_param(self._wi_initializer, (num_feedback, self.num_unit * 3))
    if self.trainable:
      self.Wi_fb = bm.TrainVar(self.Wi_fb)

  def init_state(self, num_batch=1):
    return init_param(self._state_initializer, (num_batch, self.num_unit))

  def forward(self, ff, fb=None, **shared_kwargs):
    gates_x = bm.matmul(bm.concatenate(ff, axis=-1), self.Wi_ff)
    if fb is not None:
      gates_x += bm.matmul(bm.concatenate(fb, axis=-1), self.Wi_fb)
    zr_x, a_x = bm.split(gates_x, indices_or_sections=[2 * self.num_unit], axis=-1)
    w_h_z, w_h_a = bm.split(self.Wh, indices_or_sections=[2 * self.num_unit], axis=-1)
    zr_h = bm.matmul(self.state, w_h_z)
    zr = zr_x + zr_h
    has_bias = (self.bias is not None)
    if has_bias:
      b_z, b_a = bm.split(self.bias, indices_or_sections=[2 * self.num_unit], axis=0)
      zr += bm.broadcast_to(b_z, zr_h.shape)
    z, r = bm.split(bm.sigmoid(zr), indices_or_sections=2, axis=-1)
    a_h = bm.matmul(r * self.state, w_h_a)
    if has_bias:
      a = bm.tanh(a_x + a_h + bm.broadcast_to(b_a, a_h.shape))
    else:
      a = bm.tanh(a_x + a_h)
    next_state = (1 - z) * self.state + z * a
    self.state.value = next_state
    return next_state


class LSTM(RecurrentNode):
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
  num_unit: int
    The number of hidden unit in the node.
  state_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The state initializer.
  wi_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The input weight initializer.
  wh_initializer: callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The hidden weight initializer.
  bias_initializer: optional, callable, Initializer, bm.ndarray, jax.numpy.ndarray
    The bias weight initializer.
  activation: str, callable
    The activation function. It can be a string or a callable function.
    See ``brainpy.math.activations`` for more details.
  trainable: bool
    Whether set the node is trainable.

  References
  ----------

  .. [1] Zaremba, Wojciech, Ilya Sutskever, and Oriol Vinyals. "Recurrent neural
         network regularization." arXiv preprint arXiv:1409.2329 (2014).
  .. [2] Jozefowicz, Rafal, Wojciech Zaremba, and Ilya Sutskever. "An empirical
         exploration of recurrent network architectures." In International conference
         on machine learning, pp. 2342-2350. PMLR, 2015.
  """
  data_pass = MultipleData('sequence')

  def __init__(
      self,
      num_unit: int,
      wi_initializer: Union[Tensor, Callable, Initializer] = XavierNormal(),
      wh_initializer: Union[Tensor, Callable, Initializer] = XavierNormal(),
      bias_initializer: Union[Tensor, Callable, Initializer] = ZeroInit(),
      state_initializer: Union[Tensor, Callable, Initializer] = ZeroInit(),
      **kwargs
  ):
    super(LSTM, self).__init__(**kwargs)

    self.num_unit = num_unit
    check_integer(num_unit, 'num_unit', min_bound=1, allow_none=False)
    self.set_output_shape((None, self.num_unit,))

    self._state_initializer = state_initializer
    self._wi_initializer = wi_initializer
    self._wh_initializer = wh_initializer
    self._bias_initializer = bias_initializer
    check_initializer(wi_initializer, 'wi_initializer', allow_none=False)
    check_initializer(wh_initializer, 'wh_initializer', allow_none=False)
    check_initializer(bias_initializer, 'bias_initializer', allow_none=True)
    check_initializer(state_initializer, 'state_initializer', allow_none=False)

  def init_ff_conn(self):
    # data shape
    unique_size, free_sizes = check_shape_consistency(self.feedforward_shapes, -1, True)
    assert len(unique_size) == 1, 'Only support data with or without batch size.'
    # weights
    num_input = sum(free_sizes)
    self.Wi_ff = init_param(self._wi_initializer, (num_input, self.num_unit * 4))
    self.Wh = init_param(self._wh_initializer, (self.num_unit, self.num_unit * 4))
    self.bias = init_param(self._bias_initializer, (self.num_unit * 4,))
    if self.trainable:
      self.Wi_ff = bm.TrainVar(self.Wi_ff)
      self.Wh = bm.TrainVar(self.Wh)
      self.bias = None if (self.bias is None) else bm.TrainVar(self.bias)

  def init_fb_conn(self):
    unique_size, free_sizes = check_shape_consistency(self.feedback_shapes, -1, True)
    assert len(unique_size) == 1, 'Only support data with or without batch size.'
    num_feedback = sum(free_sizes)
    # weights
    self.Wi_fb = init_param(self._wi_initializer, (num_feedback, self.num_unit * 4))
    if self.trainable:
      self.Wi_fb = bm.TrainVar(self.Wi_fb)

  def init_state(self, num_batch=1):
    return init_param(self._state_initializer, (num_batch * 2, self.num_unit))

  def forward(self, ff, fb=None, **shared_kwargs):
    h, c = bm.split(self.state, 2)
    gated = bm.concatenate(ff, axis=-1) @ self.Wi_ff
    if fb is not None:
      gated += bm.concatenate(fb, axis=-1) @ self.Wi_fb
    if self.bias is not None:
      gated += self.bias
    gated += h @ self.Wh
    i, g, f, o = bm.split(gated, indices_or_sections=4, axis=-1)
    c = bm.sigmoid(f + 1.) * c + bm.sigmoid(i) * bm.tanh(g)
    h = bm.sigmoid(o) * bm.tanh(c)
    self.state.value = bm.vstack([h, c])
    return h

  @property
  def h(self):
    """Hidden state."""
    return bm.split(self.state, 2)[0]

  @h.setter
  def h(self, value):
    if self.state is None:
      raise ValueError('Cannot set "h" state. Because the state is not initialized.')
    self.state[:self.state.shape[0] // 2, :] = value

  @property
  def c(self):
    """Memory cell."""
    return bm.split(self.state, 2)[1]

  @c.setter
  def c(self, value):
    if self.state is None:
      raise ValueError('Cannot set "c" state. Because the state is not initialized.')
    self.state[self.state.shape[0] // 2:, :] = value


class ConvNDLSTM(RecurrentNode):
  data_pass = MultipleData('sequence')


class Conv1DLSTM(ConvNDLSTM):
  data_pass = MultipleData('sequence')


class Conv2DLSTM(ConvNDLSTM):
  data_pass = MultipleData('sequence')


class Conv3DLSTM(ConvNDLSTM):
  data_pass = MultipleData('sequence')
