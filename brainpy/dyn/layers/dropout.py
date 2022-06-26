# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.dyn.training import TrainingSystem

__all__ = [
  'Dropout'
]


class Dropout(TrainingSystem):
  """A layer that stochastically ignores a subset of inputs each training step.

  In training, to compensate for the fraction of input values dropped (`rate`),
  all surviving values are multiplied by `1 / (1 - rate)`.

  The parameter `shared_axes` allows to specify a list of axes on which
  the mask will be shared: we will use size 1 on those axes for dropout mask
  and broadcast it. Sharing reduces randomness, but can save memory.

  This layer is active only during training (`mode='train'`). In other
  circumstances it is a no-op.

  Parameters
  ----------
  prob : float
    Probability to keep element of the tensor.
  seed : optional, int
    The random sampling seed.
  name : str, optional
    The name of the dynamic system.

  References
  ----------
  .. [1] Srivastava, Nitish, et al. "Dropout: a simple way to prevent
         neural networks from overfitting." The journal of machine learning
         research 15.1 (2014): 1929-1958.
  """
  def __init__(self, prob, seed=None, trainable=False, name=None):
    super(Dropout, self).__init__(trainable=trainable, name=name)
    self.prob = prob
    self.rng = bm.random.RandomState(seed=seed)

  def update(self, sha, x):
    if sha.get('fit', True):
      keep_mask = self.rng.bernoulli(self.prob, x.shape)
      return bm.where(keep_mask, x / self.prob, 0.)
    else:
      return x

  def reset(self, batch_size=1):
    pass

  def reset_state(self, batch_size=1):
    pass
