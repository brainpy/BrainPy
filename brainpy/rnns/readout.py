# -*- coding: utf-8 -*-


from brainpy import math
from brainpy.rnns.base import Module
from brainpy.initialize import Initializer, XavierNormal, Uniform, ZeroInit

__all__ = [
  'LinearReadout'
]


class LinearReadout(Module):
  """Neuron group to readout information linearly.

  Parameters
  ----------
  num_hidden : int
    The neuron group size.
  num_input : int
    The input size.
  w_init : Initializer
    Initializer for the weights.
  b_init : Initializer
    Initializer for the bias.
  has_bias : bool
    Whether has the bias to compute.
  s_init : Initializer
    Initializer for variable states.
  train_mask : optional, math.ndarray
    The training mask for the weights.
  """

  def __init__(self, num_hidden, num_input, num_batch=1,
               w_init=XavierNormal(), b_init=ZeroInit(),
               has_bias=True, s_init=Uniform(), train_mask=None, **kwargs):
    super(LinearReadout, self).__init__(**kwargs)

    # parameters
    self.w_init = w_init
    self.b_init = b_init
    self.s_init = s_init
    self.num_input = num_input
    self.has_bias = has_bias

    # weights
    self.w = math.TrainVar(w_init((num_input, self.num)))
    if has_bias: self.b = math.TrainVar(b_init((self.num,)))

    if train_mask is not None:
      assert train_mask.shape == self.w.shape
      self.train_mask = train_mask

    # variables
    self.s = math.Variable(self.s_init((num_batch, self.num)))

  def update(self, x, **kwargs):
    if self.has_bias:
      self.s[:] = x @ self.w + self.b
    else:
      self.s[:] = x @ self.w
    return self.s
