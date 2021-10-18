# -*- coding: utf-8 -*-

import abc
import brainpy.math.jax as bm

from brainpy.simulation.brainobjects.neuron import NeuGroup
from brainpy.simulation.initialize import XavierNormal, ZeroInit, Uniform, Orthogonal

__all__ = [
  'RNNCore',
  'VanillaRNN',
  'GRU',
  'LSTM',
]


class RNNCore(NeuGroup):
  def __init__(self, num_hidden, num_input, **kwargs):
    super(RNNCore, self).__init__(size=num_hidden, **kwargs)
    assert isinstance(num_hidden, int)
    assert isinstance(num_input, int)
    self.num_hidden = num_hidden
    self.num_input = num_input

  @abc.abstractmethod
  def init(self, num_batch=1, **kwargs):
    pass

  @abc.abstractmethod
  def update(self, x, **kwargs):
    pass


class VanillaRNN(RNNCore):
  r"""Basic fully-connected RNN core.

  Given :math:`x_t` and the previous hidden state :math:`h_{t-1}` the
  core computes

  .. math::

     h_t = \mathrm{ReLU}(w_i x_t + b_i + w_h h_{t-1} + b_h)

  The output is equal to the new state, :math:`h_t`.
  """
  target_backend = 'jax'

  def __init__(self, num_hidden, num_input, w_init=XavierNormal(), b_init=ZeroInit(),
               h_init=Uniform(), **kwargs):
    super(VanillaRNN, self).__init__(num_hidden, num_input, **kwargs)

    # parameters
    self.h_init = h_init

    # weights
    self.w_ir = bm.TrainVar(w_init((num_input, num_hidden)))
    self.w_rr = bm.TrainVar(w_init((num_hidden, num_hidden)))
    self.b = bm.TrainVar(b_init((num_hidden,)))

  def update(self, x, **kwargs):
    self.h[:] = bm.relu(x @ self.w_ir + self.h @ self.w_rr + self.b)
    return self.h

  def init(self, num_batch=1, **kwargs):
    self.num_batch = num_batch
    self.h = bm.Variable(self.h_init((num_batch, self.num_hidden)))


class GRU(RNNCore):
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

  References
  ----------
  .. [1] Chung, J., Gulcehre, C., Cho, K. and Bengio, Y., 2014. Empirical
         evaluation of gated recurrent neural networks on sequence modeling.
         arXiv preprint arXiv:1412.3555.
  """

  def __init__(self, num_hidden, num_input, wx_init=Orthogonal(),
               wh_init=Orthogonal(), b_init=ZeroInit(), h_init=ZeroInit(), **kwargs):
    super(GRU, self).__init__(num_hidden, num_input, **kwargs)

    # parameters
    self.h_init = h_init

    # weights
    self.w_iz = bm.TrainVar(wx_init((num_input, num_hidden)))
    self.w_ir = bm.TrainVar(wx_init((num_input, num_hidden)))
    self.w_ia = bm.TrainVar(wx_init((num_input, num_hidden)))
    self.w_hz = bm.TrainVar(wh_init((num_hidden, num_hidden)))
    self.w_hr = bm.TrainVar(wh_init((num_hidden, num_hidden)))
    self.w_ha = bm.TrainVar(wh_init((num_hidden, num_hidden)))
    self.bz = bm.TrainVar(b_init((num_hidden,)))
    self.br = bm.TrainVar(b_init((num_hidden,)))
    self.ba = bm.TrainVar(b_init((num_hidden,)))

  def update(self, x, **kwargs):
    z = bm.sigmoid(x @ self.w_iz + self.h @ self.w_hz + self.bz)
    r = bm.sigmoid(x @ self.w_ir + self.h @ self.w_hr + self.br)
    a = bm.tanh(x @ self.w_ia + (r * self.h) @ self.w_ha + self.ba)
    self.h[:] = (1 - z) * self.h + z * a
    return self.h

  def init(self, num_batch=1, **kwargs):
    self.num_batch = num_batch
    self.h = bm.Variable(self.h_init((num_batch, self.num_hidden)))


class LSTM(RNNCore):
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

  def __init__(self, num_hidden, num_input,
               w_init=Orthogonal(), b_init=ZeroInit(),
               h_init=ZeroInit(), **kwargs):
    super(LSTM, self).__init__(num_hidden, num_input, **kwargs)

    # parameters
    self.h_init = h_init

    # weights
    self.w = bm.TrainVar(w_init((num_input + num_hidden, num_hidden * 4)))
    self.b = bm.TrainVar(b_init((num_hidden * 4,)))

  def update(self, x, **kwargs):
    xh = bm.concatenate([x, self.h], axis=-1)
    gated = xh @ self.w + self.b
    i, g, f, o = bm.split(gated, indices_or_sections=4, axis=-1)
    c = bm.sigmoid(f + 1.) * self.c + bm.sigmoid(i) * bm.tanh(g)
    h = bm.sigmoid(o) * bm.tanh(c)
    self.h[:] = h
    self.c[:] = c
    return h

  def init(self, num_batch=1, **kwargs):
    self.num_batch = num_batch
    self.h = bm.Variable(self.h_init((num_batch, self.num_hidden)))
    self.c = bm.Variable(self.h_init((num_batch, self.num_hidden)))


class ConvLSTM(RNNCore):
  pass
