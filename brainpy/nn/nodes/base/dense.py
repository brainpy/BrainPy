# -*- coding: utf-8 -*-


from typing import Sequence, Optional, Dict, Callable, Union

import jax.numpy as jnp

from brainpy import math as bm
from brainpy.errors import MathError
from brainpy.initialize import XavierNormal, ZeroInit, Initializer, init_param
from brainpy.nn.base import Node
from brainpy.tools.checking import (check_shape_consistency,
                                    check_initializer)
from brainpy.types import Tensor

__all__ = [
  'GeneralDense',
  'Dense',
]


class GeneralDense(Node):
  r"""A linear transformation applied over the last dimension of the input.

  Mathematically, this node can be defined as:

  .. math::

     y = x  \cdot W + b

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
    super(GeneralDense, self).__init__(trainable=trainable, **kwargs)

    # shape
    self.num_unit = num_unit
    if num_unit < 0:
      raise ValueError(f'Received an invalid value for `num_unit`, expected '
                       f'a positive integer. Received: num_unit={num_unit}')

    # weight initializer
    self.weight_initializer = weight_initializer
    self.bias_initializer = bias_initializer
    check_initializer(weight_initializer, 'weight_initializer')
    check_initializer(bias_initializer, 'bias_initializer', allow_none=True)

    # weights
    self.Wff = None
    self.bias = None
    self.Wfb = None

  def init_ff_conn(self):
    # shapes
    other_size, free_shapes = check_shape_consistency(self.feedforward_shapes, -1, True)
    # set output size
    self.set_output_shape(other_size + (self.num_unit,))

    # initialize feedforward weights
    self.Wff = init_param(self.weight_initializer, (sum(free_shapes), self.num_unit))
    self.bias = init_param(self.bias_initializer, (self.num_unit,))
    if self.trainable:
      self.Wff = bm.TrainVar(self.Wff)
      self.bias = bm.TrainVar(self.bias) if (self.bias is not None) else None

  def init_fb_conn(self):
    other_size, free_shapes = check_shape_consistency(self.feedback_shapes, -1, True)

    # initialize feedforward weights
    weight_shapes = (sum(free_shapes), self.num_unit)
    if self.trainable:
      self.Wfb = bm.TrainVar(init_param(self.weight_initializer, weight_shapes))
    else:
      self.Wfb = init_param(self.weight_initializer, weight_shapes)

  def forward(self, ff: Sequence[Tensor], fb=None, **shared_kwargs):
    ff = bm.concatenate(ff, axis=-1)
    res = ff @ self.Wff
    if fb is not None:
      fb = bm.concatenate(fb, axis=-1)
      res += fb @ self.Wfb
    if self.bias is not None:
      res += self.bias
    return res


class Dense(GeneralDense):
  r"""A linear transformation.

  Different from :py:class:`GeneralDense`, this class only supports 2D input data.

  Mathematically, this node can be defined as:

  .. math::

     y = x \cdot W+ b

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
    super(Dense, self).__init__(num_unit=num_unit,
                                weight_initializer=weight_initializer,
                                bias_initializer=bias_initializer,
                                trainable=trainable, **kwargs)

    # set output shape
    self.set_output_shape((None, self.num_unit))

  def init_ff_conn(self):
    # shapes
    other_size, free_shapes = check_shape_consistency(self.feedforward_shapes, -1, True)
    assert other_size == (None,), (f'Only support 2D inputs, while we got '
                                   f'{len(other_size) + 1}-D shapes.')
    super(Dense, self).init_ff_conn()

  def init_fb_conn(self):
    other_size, free_shapes = check_shape_consistency(self.feedback_shapes, -1, True)
    assert other_size == (None,), (f'Only support 2D inputs, while we got '
                                   f'{len(other_size) + 1}-D shapes.')
    super(Dense, self).init_fb_conn()

  def __ridge_train__(self,
                      ffs: Sequence[Tensor],
                      targets: Tensor,
                      train_pars: Optional[Dict] = None):
    r"""The ridge training interface for the Dense node.

    This function support the batch training.

    ``targets`` should be a tensor of shape :math:`(num\_time, num\_unit)`.
    Also, the element in ``ffs`` should have the same shape.

    """
    assert self.Wfb is None, 'Currently ridge learning do not support feedback connections.'

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
      self.Wff.value = W
    else:
      self.Wff.value = W[:-1]
      self.bias.value = W[-1]
