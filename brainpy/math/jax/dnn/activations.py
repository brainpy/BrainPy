# -*- coding: utf-8 -*-

from brainpy.math.jax.ndarray import ndarray
from brainpy.math.jax import math

from jax import numpy as jnp
from jax import nn
from jax import scipy
from jax import lax

__all__ = [
  'celu',
  'elu',
  'leaky_relu',
  'log_sigmoid',
  'log_softmax',
  'logsumexp',
  'selu',
  'sigmoid',
  'softmax',
  'softplus',
  'tanh',
  'relu',
]


def celu(x, alpha=1.0):
  """Continuously-differentiable exponential linear unit activation.

  Computes the element-wise function:

  .. math::
    \mathrm{celu}(x) = \begin{cases}
      x, & x > 0\\
      \alpha \left(\exp(\frac{x}{\alpha}) - 1\right), & x \le 0
    \end{cases}

  For more information, see
  `Continuously Differentiable Exponential Linear Units
  <https://arxiv.org/pdf/1704.07483.pdf>`_.

  Parameters
  ----------
  x : ndarray, jnp.ndarray
    The input array.
  alpha : ndarray, float
    The default is 1.0.
  """
  return math.where(x > 0, x, alpha * math.expm1(x / alpha))


elu = nn.elu
leaky_relu = nn.leaky_relu
log_sigmoid = nn.log_sigmoid
log_softmax = nn.log_softmax
logsumexp = scipy.special.logsumexp
selu = nn.selu
sigmoid = nn.sigmoid
softmax = nn.softmax
softplus = nn.softplus
tanh = lax.tanh
relu = nn.relu
