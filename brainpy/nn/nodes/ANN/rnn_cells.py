# -*- coding: utf-8 -*-


import brainpy.math as bm
from brainpy.initialize import (XavierNormal, ZeroInit,
                                Uniform, Orthogonal)
from brainpy.nn.base import RecurrentNode
from brainpy.nn.utils import init_param
from brainpy.tools.checking import (check_integer,
                                    check_initializer,
                                    check_shape_consistency)

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
  """

  def __init__(
      self,
      num_unit: int,
      state_initializer=Uniform(),
      weight_initializer=XavierNormal(),
      bias_initializer=ZeroInit(),
      activation='relu',
      trainable=True,
      **kwargs
  ):
    super(VanillaRNN, self).__init__(trainable=trainable, **kwargs)

    self.num_unit = num_unit
    check_integer(num_unit, 'num_unit', min_bound=1, allow_none=False)

    self._state_initializer = state_initializer
    self._weight_initializer = weight_initializer
    self._bias_initializer = bias_initializer
    check_initializer(weight_initializer, 'weight_initializer', allow_none=False)
    check_initializer(state_initializer, 'state_initializer', allow_none=False)
    check_initializer(bias_initializer, 'bias_initializer', allow_none=True)

    self.activation = bm.activations.get(activation)

  def init_ff(self):
    unique_size, free_sizes = check_shape_consistency(self.feedforward_shapes, -1, True)
    assert len(unique_size) == 1, 'Only support data with or without batch size.'
    num_input = sum(free_sizes)
    self.set_output_shape(unique_size + (self.num_unit,))
    # weights
    self.weight = init_param(self._weight_initializer, (num_input + self.num_unit, self.num_unit))
    self.bias = init_param(self._bias_initializer, (self.num_unit,))
    if self.trainable:
      self.weight = bm.TrainVar(self.weight)
      self.bias = None if (self.bias is None) else bm.TrainVar(self.bias)

  def init_state(self, num_batch):
    state = init_param(self._state_initializer, (num_batch, self.num_unit))
    self.set_state(state)

  def forward(self, ff, fb=None, **kwargs):
    ff = bm.concatenate(tuple(ff) + (self.state.value,), axis=-1)
    h = ff @ self.weight
    if self.bias is not None:
      h = h + self.bias
    self.state.value = self.activation(h)
    return self.state.value


class GRU(RecurrentNode):
  r"""
  Gated Recurrent Unit.

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

  References
  ----------
  .. [1] Chung, J., Gulcehre, C., Cho, K. and Bengio, Y., 2014. Empirical
         evaluation of gated recurrent neural networks on sequence modeling.
         arXiv preprint arXiv:1412.3555.
  """

  def __init__(
      self,
      num_unit: int,
      wi_initializer=Orthogonal(),
      wh_initializer=Orthogonal(),
      bias_initializer=ZeroInit(),
      state_initializer=ZeroInit(),
      trainable=True,
      **kwargs
  ):
    super(GRU, self).__init__(trainable=trainable, **kwargs)

    self.num_unit = num_unit
    check_integer(num_unit, 'num_unit', min_bound=1, allow_none=False)

    self._wi_initializer = wi_initializer
    self._wh_initializer = wh_initializer
    self._bias_initializer = bias_initializer
    self._state_initializer = state_initializer
    check_initializer(wi_initializer, 'wi_initializer', allow_none=False)
    check_initializer(wh_initializer, 'wh_initializer', allow_none=False)
    check_initializer(state_initializer, 'state_initializer', allow_none=False)
    check_initializer(bias_initializer, 'bias_initializer', allow_none=True)

  def init_ff(self):
    # data shape
    unique_size, free_sizes = check_shape_consistency(self.feedforward_shapes, -1, True)
    assert len(unique_size) == 1, 'Only support data with or without batch size.'
    num_input = sum(free_sizes)
    self.set_output_shape(unique_size + (self.num_unit,))
    # weights
    self.i_weight = init_param(self._wi_initializer, (num_input, self.num_unit * 3))
    self.h_weight = init_param(self._wh_initializer, (self.num_unit, self.num_unit * 3))
    self.bias = init_param(self._bias_initializer, (self.num_unit * 3,))
    if self.trainable:
      self.i_weight = bm.TrainVar(self.i_weight)
      self.h_weight = bm.TrainVar(self.h_weight)
      self.bias = bm.TrainVar(self.bias) if (self.bias is not None) else None

  def init_state(self, num_batch):
    state = init_param(self._state_initializer, (num_batch, self.num_unit))
    self.set_state(state)

  def forward(self, ff, fb=None, **kwargs):
    ff = bm.concatenate(ff, axis=-1)
    gates_x = bm.matmul(ff, self.i_weight)
    zr_x, a_x = bm.split(gates_x, indices_or_sections=[2 * self.num_unit], axis=-1)
    w_h_z, w_h_a = bm.split(self.h_weight, indices_or_sections=[2 * self.num_unit], axis=-1)
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
      num_unit: int,
      weight_initializer=Orthogonal(),
      bias_initializer=ZeroInit(),
      state_initializer=ZeroInit(),
      trainable=True,
      **kwargs
  ):
    super(LSTM, self).__init__(trainable=trainable, **kwargs)

    self.num_unit = num_unit
    check_integer(num_unit, 'num_unit', min_bound=1, allow_none=False)

    self._state_initializer = state_initializer
    self._weight_initializer = weight_initializer
    self._bias_initializer = bias_initializer
    check_initializer(weight_initializer, 'weight_initializer', allow_none=False)
    check_initializer(bias_initializer, 'bias_initializer', allow_none=True)
    check_initializer(state_initializer, 'state_initializer', allow_none=False)

  def init_ff(self):
    # data shape
    unique_size, free_sizes = check_shape_consistency(self.feedforward_shapes, -1, True)
    assert len(unique_size) == 1, 'Only support data with or without batch size.'
    num_input = sum(free_sizes)
    self.set_output_shape(unique_size + (self.num_unit,))
    # weights
    self.weight = init_param(self._weight_initializer, (num_input + self.num_unit, self.num_unit * 4))
    self.bias = init_param(self._bias_initializer, (self.num_unit * 4,))
    if self.trainable:
      self.weight = bm.TrainVar(self.weight)
      self.bias = None if (self.bias is None) else bm.TrainVar(self.bias)

  def init_state(self, num_batch):
    hc = init_param(self._state_initializer, (num_batch * 2, self.num_unit))
    self.set_state(hc)

  def forward(self, ff, fb=None, **kwargs):
    h, c = bm.split(self.state, 2)
    xh = bm.concatenate(tuple(ff) + (h,), axis=-1)
    if self.bias is None:
      gated = xh @ self.weight
    else:
      gated = xh @ self.weight + self.bias
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
      raise ValueError('Cannot set "h" state. Because it is not initialized.')
    self.state[:self.state.shape[0] // 2, :] = value

  @property
  def c(self):
    """Memory cell."""
    return bm.split(self.state, 2)[1]

  @c.setter
  def c(self, value):
    if self.state is None:
      raise ValueError('Cannot set "c" state. Because it is not initialized.')
    self.state[self.state.shape[0] // 2:, :] = value


class ConvNDLSTM(RecurrentNode):
  pass


class Conv1DLSTM(ConvNDLSTM):
  pass


class Conv2DLSTM(ConvNDLSTM):
  pass


class Conv3DLSTM(ConvNDLSTM):
  pass
