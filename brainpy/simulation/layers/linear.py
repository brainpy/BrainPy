# -*- coding: utf-8 -*-


from brainpy.simulation.module import Module
from brainpy.simulation._imports import mjax
from brainpy.simulation.initialize import XavierNormal, Initializer, ZeroInit

__all__ = [
  'Linear'
]


class Linear(Module):
  """A fully connected layer implemented as the dot product of inputs and
  weights.

  Parameters
  ----------
  n_out : int
      Desired size or shape of layer output
  n_in : int
      The layer input size feeding into this layer
  w_init : Initializer
      Initializer for the weights.
  b_init : Initializer
      Initializer for the bias.
  name : str, optional
  """

  def __init__(self, n_in, n_out, w_init=XavierNormal(), b_init=ZeroInit(), name=None):
    self.n_out = n_out
    self.n_in = n_in

    self.w = mjax.TrainVar(w_init((n_in, n_out)))
    self.b = mjax.TrainVar(b_init(n_out))
    super(Linear, self).__init__(name=name)

  def update(self, x, **kwargs):
    """Returns the results of applying the linear transformation to input x."""
    y = mjax.dot(x, self.w) + self.b
    return y
