# -*- coding: utf-8 -*-


import brainpy.math.jax as bm
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
  w : Initializer, JaxArray, jax.numpy.ndarray
    Initializer for the weights.
  b : Initializer, JaxArray, jax.numpy.ndarray, optional
    Initializer for the bias.
  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  monitors : None, list, tuple, datastructures.Monitor
      Variables to monitor.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, num_hidden, num_input, w=XavierNormal(), b=ZeroInit(), **kwargs):
    super(Dense, self).__init__(**kwargs)

    # parameters
    self.has_bias = True
    self.num_input = num_input
    self.num_hidden = num_hidden

    # variables
    if callable(w):
      self.w = bm.TrainVar(w((num_input, num_hidden)))
    else:
      assert w.shape == (num_input, num_hidden)
      self.w = bm.TrainVar(w)
    if b is None:
      self.has_bias = False
    elif callable(b):
      self.b = bm.TrainVar(b((num_hidden,)))
    else:
      assert b.shape == (num_hidden, )
      self.b = bm.TrainVar(b)

  def update(self, x):
    """Returns the results of applying the linear transformation to input x."""
    if self.has_bias:
      return x @ self.w + self.b
    else:
      return x @ self.w
