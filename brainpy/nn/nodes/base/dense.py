# -*- coding: utf-8 -*-


from typing import Sequence, Optional, Dict, Callable, Union

import jax.numpy as jnp

from brainpy import math as bm
from brainpy.errors import UnsupportedError, MathError
from brainpy.initialize import XavierNormal, ZeroInit, Initializer
from brainpy.nn import utils
from brainpy.nn.base import Node
from brainpy.tools.checking import (check_shape_consistency,
                                    check_initializer)
from brainpy.types import Tensor

__all__ = [
  'Dense',
]


class Dense(Node):
  r"""A linear transformation applied over the last dimension of the input.

  Mathematically, this node can be defined as:

  .. math::

     y = W \cdot x + b

  Parameters
  ----------
  num_unit: int
    The number of the output features. A positive integer.
  weight_initializer: optional, Initializer
    The weight initialization.
  bias_initializer: optional, Initializer
    The bias initialization.
  trainable: bool
    Enable training this node or not. (default True)
  """

  def __init__(
      self,
      num_unit: int,
      weight_initializer: Union[Initializer, Callable, Tensor] = XavierNormal(),
      bias_initializer: Optional[Union[Initializer, Callable, Tensor]] = ZeroInit(),
      trainable: bool = True,
      **kwargs
  ):
    super(Dense, self).__init__(trainable=trainable, **kwargs)
    self.num_unit = num_unit
    if num_unit < 0:
      raise ValueError(f'Received an invalid value for `num_unit`, expected '
                       f'a positive integer. Received: num_unit={num_unit}')
    self.weight_initializer = weight_initializer
    self.bias_initializer = bias_initializer
    check_initializer(weight_initializer, 'weight_initializer')
    check_initializer(bias_initializer, 'bias_initializer', allow_none=True)

  def init_ff(self):
    # shapes
    in_sizes = [size[1:] for size in self.feedforward_shapes]  # remove batch size
    unique_shape, free_shapes = check_shape_consistency(in_sizes, -1, True)
    weight_shape = (sum(free_shapes), self.num_unit)
    bias_shape = (self.num_unit,)
    # set output size
    self.set_output_shape((None, ) + unique_shape + (self.num_unit,))
    # initialize feedforward weights
    self.weights = utils.init_param(self.weight_initializer, weight_shape)
    self.bias = utils.init_param(self.bias_initializer, bias_shape)
    if self.trainable:
      self.weights = bm.TrainVar(self.weights)
      if self.bias is not None:
        self.bias = bm.TrainVar(self.bias)

  def forward(self, ff: Sequence[Tensor], **kwargs):
    ff = bm.concatenate(ff, axis=-1)
    if self.bias is None:
      return ff @ self.weights
    else:
      return ff @ self.weights + self.bias

  def __ridge_train__(self,
                      ffs: Sequence[Tensor],
                      targets: Tensor,
                      train_pars: Optional[Dict] = None):
    r"""The ridge training interface for the Dense node.

    This function support the batch training.

    ``targets`` should be a tensor of shape :math:`(num\_time, num\_unit)`.
    Also, the element in ``ffs`` should have the same shape.

    """

    # parameters
    if train_pars is None: train_pars = dict()
    beta = train_pars.get('beta', 0.)
    # checking
    ffs = bm.concatenate(ffs, axis=-1)
    if not isinstance(targets, (bm.ndarray, jnp.ndarray)):
      raise MathError(f'"targets" must be a tensor, but got {type(targets)}')
    assert ffs.ndim == 3, 'Must be a 3D tensor with shape of (num_sample, num_time, num_feature)'
    assert targets.ndim == 3, 'Must be a 3D tensor with shape of (num_sample, num_time, num_feature)'
    assert ffs.shape[0] == targets.shape[0] == 1, (f'Only support training one batch size, '
                                                   f'but got {ffs.shape[0]} (for inputs) and '
                                                   f'{targets.shape[0]} (for targets)')
    ffs, targets = ffs[0], targets[0]
    if ffs.shape[0] != targets.shape[0]:
      raise MathError(f'The time dimension of input and target data should be '
                      f'the same, while we got {ffs.shape[0]} != {targets.shape[0]}')
    # solve weights by ridge regression
    if self.bias is not None:
      ffs = bm.concatenate([ffs, bm.ones(ffs.shape[:-1] + (1,))], axis=-1)  # (..., num_input+1)
    temp = ffs.T @ ffs
    if beta > 0.:
      temp += beta * bm.eye(ffs.shape[-1])
    W = bm.linalg.pinv(temp) @ (ffs.T @ targets)
    # assign trained weights
    if self.bias is None:
      self.weights.value = W
    else:
      self.weights.value = W[:-1]
      self.bias.value = W[-1]

  def __force_init__(self, *args, **kwargs):
    raise UnsupportedError(f'{self.__class__.__name__} node does not support force '
                           f'learning. Please use brainpy.nn.LinearReadout.')

  def __force_train__(self, *args, **kwargs):
    raise UnsupportedError(f'{self.__class__.__name__} node does not support force '
                           f'learning. Please use brainpy.nn.LinearReadout.')
