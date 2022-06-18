# -*- coding: utf-8 -*-


from typing import Sequence, Optional, Callable, Union

import jax.numpy as jnp

from brainpy import math as bm
from brainpy.errors import MathError
from brainpy.initialize import XavierNormal, ZeroInit, Initializer, parameter
from brainpy.compat.nn.base import Node
from brainpy.compat.nn.datatypes import MultipleData
from brainpy.tools.checking import (check_shape_consistency,
                                    check_initializer)
from brainpy.types import Tensor

__all__ = [
  'DenseMD',
  'Dense',
]


class DenseMD(Node):
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

  data_pass = MultipleData('sequence')

  def __init__(
      self,
      num_unit: int,
      weight_initializer: Union[Initializer, Callable, Tensor] = XavierNormal(),
      bias_initializer: Optional[Union[Initializer, Callable, Tensor]] = ZeroInit(),
      trainable: bool = True,
      **kwargs
  ):
    super(DenseMD, self).__init__(trainable=trainable, **kwargs)

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
    self.Wff = parameter(self.weight_initializer, (sum(free_shapes), self.num_unit))
    self.bias = parameter(self.bias_initializer, (self.num_unit,))
    if self.trainable:
      self.Wff = bm.TrainVar(self.Wff)
      self.bias = None if (self.bias is None) else bm.TrainVar(self.bias)

  def init_fb_conn(self):
    other_size, free_shapes = check_shape_consistency(self.feedback_shapes, -1, True)

    # initialize feedback weights
    weight_shapes = (sum(free_shapes), self.num_unit)
    if self.trainable:
      self.Wfb = bm.TrainVar(parameter(self.weight_initializer, weight_shapes))
    else:
      self.Wfb = parameter(self.weight_initializer, weight_shapes)

  def forward(self, ff: Sequence[Tensor], fb=None, **shared_kwargs):
    ff = bm.concatenate(ff, axis=-1)
    res = ff @ self.Wff
    if fb is not None:
      fb = bm.concatenate(fb, axis=-1)
      res += fb @ self.Wfb
    if self.bias is not None:
      res += self.bias
    return res


class Dense(DenseMD):
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
  data_pass = MultipleData('sequence')

  def __init__(
      self,
      num_unit: int,
      weight_initializer: Union[Initializer, Callable, Tensor] = XavierNormal(),
      bias_initializer: Optional[Union[Initializer, Callable, Tensor]] = ZeroInit(),
      **kwargs
  ):
    super(Dense, self).__init__(num_unit=num_unit,
                                weight_initializer=weight_initializer,
                                bias_initializer=bias_initializer,
                                **kwargs)
    # set output shape
    self.set_output_shape((None, self.num_unit))

  def init_ff_conn(self):
    # shapes
    other_size, free_shapes = check_shape_consistency(self.feedforward_shapes, -1, True)
    if other_size != (None,):
      raise ValueError(f'{self.__class__.__name__} only support 2D inputs, while '
                       f'we got {len(other_size) + 1}-D shapes. For >2D inputs, '
                       f'you should use brainpy.nn.{DenseMD.__name__} instead. ')
    super(Dense, self).init_ff_conn()

  def init_fb_conn(self):
    other_size, free_shapes = check_shape_consistency(self.feedback_shapes, -1, True)
    if other_size != (None,):
      raise ValueError(f'{self.__class__.__name__} only support 2D inputs, while '
                       f'we got {len(other_size) + 1}-D shapes. For >2D inputs, '
                       f'you should use brainpy.nn.{DenseMD.__name__} instead. ')
    super(Dense, self).init_fb_conn()

  def offline_fit(
      self,
      targets: Tensor,
      ffs: Sequence[Tensor],
      fbs: Optional[Sequence[Tensor]] = None,
  ):
    """The offline training interface for the Dense node."""
    # data checking
    ffs = bm.concatenate(ffs, axis=-1)
    if not isinstance(targets, (bm.ndarray, jnp.ndarray)):
      raise MathError(f'"targets" must be a tensor, but got {type(targets)}')
    if ffs.ndim != 3:
      raise ValueError(f'"ffs" must be a 3D tensor with shape of (num_sample, num_time, '
                       f'num_feature), but we got {ffs.shape}')
    if targets.ndim != 3:
      raise ValueError(f'"targets" must be a 3D tensor with shape of (num_sample, num_time, '
                       f'num_feature), but we got {targets.shape}')
    if ffs.shape[0] != targets.shape[0]:
      raise ValueError(f'Batch size of the input and target data should be '
                       f'the same, while we got {ffs.shape[0]} != {targets.shape[0]}.')
    if ffs.shape[1] != targets.shape[1]:
      raise MathError(f'The time dimension of input and target data should be '
                      f'the same, while we got {ffs.shape[1]} != {targets.shape[1]}')
    if fbs is not None:
      fbs = bm.concatenate(fbs, axis=-1)
      if fbs.ndim != 3:
        raise ValueError(f'"fbs" must be a 3D tensor with shape of (num_sample, num_time, '
                         f'num_feature), but we got {fbs.shape}')
      if ffs.shape[0] != fbs.shape[0]:
        raise ValueError(f'Batch size of the feedforward and the feedback inputs should be '
                         f'the same, while we got {ffs.shape[0]} != {fbs.shape[0]}.')
      if ffs.shape[1] != fbs.shape[1]:
        raise MathError(f'The time dimension of feedforward and feedback inputs should be '
                        f'the same, while we got {ffs.shape[1]} != {fbs.shape[1]}')

    # get input and target training data
    inputs = ffs
    num_ff_input = inputs.shape[2]
    if self.bias is not None:
      inputs = bm.concatenate([bm.ones(ffs.shape[:2] + (1,)), inputs], axis=-1)  # (..., 1 + num_ff_input)
    if fbs is not None:
      inputs = bm.concatenate([inputs, fbs], axis=-1)  # (..., 1 + num_ff_input + num_fb_input)

    # solve weights by offline training methods
    weights = self.offline_fit_by(targets, inputs)

    # assign trained weights
    if self.bias is None:
      if fbs is None:
        self.Wff.value = weights
      else:
        self.Wff.value, self.Wfb.value = bm.split(weights, [num_ff_input])
    else:
      if fbs is None:
        bias, Wff = bm.split(weights, [1])
        self.bias.value = bias[0]
        self.Wff.value = Wff
      else:
        bias, Wff, Wfb = bm.split(weights, [1, 1 + num_ff_input])
        self.bias.value = bias[0]
        self.Wff.value = Wff
        self.Wfb.value = Wfb
