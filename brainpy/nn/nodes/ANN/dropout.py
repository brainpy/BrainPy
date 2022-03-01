# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.nn.base import Node

__all__ = [
  'Dropout'
]


class Dropout(Node):
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
  name : str, optional
    The name of the dynamic system.
  """

  def __init__(self, prob, seed=None, **kwargs):
    super(Dropout, self).__init__(**kwargs)

    # probability
    self.prob = prob
    self.rng = bm.random.RandomState(seed=seed)

  def ff_init(self):
    assert len(self.input_shapes) == 1, 'Only support one feedforward input.'
    self.set_output_shape(self.input_shapes[0])

  def forward(self, ff, **kwargs):
    ff = list(ff.values())[0]
    if kwargs.get('train', True):
      keep_mask = self.rng.bernoulli(self.prob, ff.shape)
      return bm.where(keep_mask, ff / self.prob, 0.)
    else:
      return ff
