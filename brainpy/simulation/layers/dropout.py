# -*- coding: utf-8 -*-

from brainpy.simulation.module import Module
from brainpy.simulation._imports import mjax

__all__ = [
  'Dropout'
]


class Dropout(Module):
  """A layer that stochastically ignores a subset of inputs each training step.

  In training, to compensate for the fraction of input values dropped (`rate`),
  all surviving values are multiplied by `1 / (1 - rate)`.

  The parameter `shared_axes` allows to specify a list of axes on which
  the mask will be shared: we will use size 1 on those axes for dropout mask
  and broadcast it. Sharing reduces randomness, but can save memory.

  This layer is active only during training (`mode='train'`). In other
  circumstances it is a no-op.

  Originally introduced in the paper "Dropout: A Simple Way to Prevent Neural
  Networks from Overfitting" available under the following link:
  https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

  Parameters
  ----------
  prob : float
    Probability to keep element of the tensor.
  """

  def __init__(self, prob, name=None):
    self.prob = prob
    super(Dropout, self).__init__(name=name)

  def update(self, x, config=None, **kwargs):
    if config is None:
      config = dict()
    if config.get('train', True):
      keep_mask = mjax.random.bernoulli(self.prob, x.shape)
      return mjax.where(keep_mask, x / self.prob, 0.)
    else:
      return x
