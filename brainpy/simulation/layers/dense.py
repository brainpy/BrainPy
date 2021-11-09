# -*- coding: utf-8 -*-


from brainpy import math
from brainpy.simulation.initialize import XavierNormal, Initializer, ZeroInit
from .base import Module

__all__ = [
  'Dense'
]


class Dense(Module):
  """A fully connected layer implemented as the dot product of inputs and weights.

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
  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  monitors : None, list, tuple, datastructures.Monitor
      Variables to monitor.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, num_hidden, num_input, w_init=XavierNormal(),
               b_init=ZeroInit(), has_bias=True, **kwargs):
    super(Dense, self).__init__(**kwargs)

    # parameters
    self.has_bias = has_bias
    self.num_input = num_input
    self.num_hidden = num_hidden

    # variables
    self.w = math.TrainVar(w_init((num_input, num_hidden)))
    if has_bias: self.b = math.TrainVar(b_init((num_hidden,)))

  def update(self, x, **kwargs):
    """Returns the results of applying the linear transformation to input x."""
    if self.has_bias:
      return x @ self.w + self.b
    else:
      return x @ self.w
