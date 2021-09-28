# -*- coding: utf-8 -*-

from brainpy.simulation.module import Module
from brainpy.simulation._imports import mjax


class BatchNorm(Module):
  """Batch Normalization module.

  Normalizes inputs to maintain a mean of ~0 and stddev of ~1 [1]_.

  Parameters
  ----------
  axis : int
  momentum : float
    The value used to compute exponential moving average of batch statistics.
  eps : float
    The small value which is used for numerical stability.

  References
  ----------
  .. [1] Li, Xiang, et al. “Understanding the Disharmony Between Dropout and
         Batch Normalization by Variance Shift.” 2019 IEEE/CVF Conference on
         Computer Vision and Pattern Recognition (CVPR), 2019, pp. 2682–2690.

  """

  def __init__(self, dims, axis, momentum=0.99, eps=1e-6, name=None):
    super(BatchNorm, self).__init__(name=name)

    self.axis = axis
    self.momentum = momentum
    self.eps = eps
    self.running_mean = mjax.zeros(dims)
    self.running_var = mjax.zeros(dims)
    self.shift = mjax.TrainVar(mjax.zeros(dims))
    self.scale = mjax.TrainVar(mjax.ones(dims))

  def update(self, x, config=None, **kwargs):
    if config is None:
      config = dict()
    if config.get('train', True):
      pass
    else:
      pass
