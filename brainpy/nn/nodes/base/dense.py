# -*- coding: utf-8 -*-


from typing import Sequence, Optional, Dict, Callable, Union

import jax.numpy as jnp

from brainpy import math as bm
from brainpy.errors import UnsupportedError
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
  init_weight: optional, Initializer
    The weight initialization.
  init_bias: optional, Initializer
    The bias initialization.
  trainable: bool
    Enable training this node or not. (default True)
  """

  def __init__(
      self,
      num_unit: int,
      init_weight: Union[Initializer, Callable, Tensor] = XavierNormal(),
      init_bias: Optional[Union[Initializer, Callable, Tensor]] = ZeroInit(),
      trainable: bool = True,
      **kwargs
  ):
    super(Dense, self).__init__(trainable=trainable, **kwargs)
    self.num_unit = num_unit
    if num_unit < 0:
      raise ValueError(f'Received an invalid value for `num_unit`, expected '
                       f'a positive integer. Received: num_unit={num_unit}')
    self.init_weight = init_weight
    self.init_bias = init_bias
    check_initializer(init_weight, 'init_weight')
    check_initializer(init_bias, 'init_bias', allow_none=True)

  def ff_init(self):
    # shapes
    unique_shape, free_shapes = check_shape_consistency(self.input_shapes, -1, True)
    weight_shape = (sum(free_shapes), self.num_unit)
    bias_shape = (self.num_unit,)
    # set output size
    self.set_output_shape(unique_shape + (self.num_unit,))
    # initialize feedforward weights
    self.weights = utils.init_param(self.init_weight, weight_shape)
    self.bias = utils.init_param(self.init_bias, bias_shape)
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
    # parameters
    if train_pars is None: train_pars = dict()
    beta = train_pars.get('beta', 0.)
    # checking
    ffs = bm.concatenate(ffs, axis=-1)
    assert isinstance(targets, (bm.ndarray, jnp.ndarray))
    assert (ffs.ndim == targets.ndim == 2) and (ffs.shape[0] == targets.shape[0])
    # solve weights by ridge regression
    if self.bias is not None:
      ffs = bm.concatenate([ffs, bm.ones((ffs.shape[0], 1))], axis=1)  # (..., num_input+1)
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
