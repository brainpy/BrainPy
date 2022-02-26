# -*- coding: utf-8 -*-


from typing import Sequence, Optional, Dict

import jax.numpy as jnp

from brainpy import math as bm
from brainpy.initialize import XavierNormal, ZeroInit
from brainpy.nn import utils
from brainpy.nn.base import Node
from brainpy.types import Tensor

__all__ = [
  'Dense',
]


class Dense(Node):
  """A linear transformation applied over the last dimension of the input.

  Parameters
  ----------
  num_unit: init
    The number of the output features.
  w_init: optional, Initializer, Tensor
    The weight initialization.
  b_init: optional, Initializer, Tensor
    The bias initialization.
  trainable: bool
    Enable training this node or not. (default True)
  """
  def __init__(self, num_unit: int, w_init=XavierNormal(), b_init=ZeroInit(), trainable=True, **kwargs):
    super(Dense, self).__init__(trainable=trainable, **kwargs)
    self.num_unit = num_unit
    self.w_init = w_init
    self.b_init = b_init

  def ff_init(self):
    # shapes
    unique_shape, free_shapes = utils.check_shape_consistency(self.input_shapes, -1, True)
    weight_shape = (sum(free_shapes), self.num_unit)
    bias_shape = (self.num_unit,)
    # set output size
    self.set_output_shape(unique_shape + (self.num_unit,))
    # initialize feedforward weights
    self.weights = utils.init_param(self.w_init, weight_shape)
    self.bias = utils.init_param(self.b_init, bias_shape)
    if self.trainable:
      self.weights = bm.TrainVar(self.weights)
      if self.bias is not None:
        self.bias = bm.TrainVar(self.bias)

  def call(self, ff: Sequence[Tensor], **kwargs):
    ff = bm.concatenate(ff, axis=-1)
    if self.bias is None:
      return ff @ self.weights
    else:
      return ff @ self.weights + self.bias

  def __ridge_train__(self,
                      ff: Sequence[Tensor], targets: Tensor, fb=None,
                      train_pars: Optional[Dict] = None):
    # parameters
    if train_pars is None: train_pars = dict()
    beta = train_pars.get('beta', 0.)
    # checking
    ff = bm.concatenate(ff, axis=-1)
    assert isinstance(targets, (bm.ndarray, jnp.ndarray))
    assert (ff.ndim == targets.ndim == 2) and (ff.shape[0] == targets.shape[0])
    # solve weights by ridge regression
    if self.bias is not None:
      ff = bm.concatenate([ff, bm.ones((ff.shape[0], 1))], axis=1)  # (..., num_input+1)
    temp = ff.T @ ff
    if beta > 0.:
      temp += beta * bm.eye(ff.shape[-1])
    W = bm.linalg.pinv(temp) @ (ff.T @ targets)
    # assign trained weights
    if self.bias is None:
      self.weights.value = W
    else:
      self.weights.value = W[:-1]
      self.bias.value = W[-1]
