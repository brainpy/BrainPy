# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.dyn.base import DynamicalSystem
from brainpy.modes import Mode, training

__all__ = [
  'Dropout'
]


class Dropout(DynamicalSystem):
  """A layer that stochastically ignores a subset of inputs each training step.

  In training, to compensate for the fraction of input values dropped (`rate`),
  all surviving values are multiplied by `1 / (1 - rate)`.

  This layer is active only during training (`mode=brainpy.modes.training`). In other
  circumstances it is a no-op.

  Parameters
  ----------
  prob : float
    Probability to keep element of the tensor.
  seed : optional, int
    The random sampling seed.
  mode: Mode
    The computation mode of the object.
  name : str, optional
    The name of the dynamic system.

  References
  ----------
  .. [1] Srivastava, Nitish, et al. "Dropout: a simple way to prevent
         neural networks from overfitting." The journal of machine learning
         research 15.1 (2014): 1929-1958.
  """

  def __init__(
      self,
      prob: float,
      seed: int = None,
      mode: Mode = training,
      name: str = None
  ):
    super(Dropout, self).__init__(mode=mode, name=name)
    self.prob = prob
    self.rng = bm.random.RandomState(seed)

  def update(self, sha, x):
    if sha.get('fit', True):
      keep_mask = self.rng.bernoulli(self.prob, x.shape)
      return bm.where(keep_mask, x / self.prob, 0.)
    else:
      return x

  def reset_state(self, batch_size=None):
    pass
