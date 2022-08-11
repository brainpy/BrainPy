# -*- coding: utf-8 -*-


from typing import Optional, Callable, Union, Dict

import jax.numpy as jnp

from brainpy import math as bm
from brainpy.dyn.base import DynamicalSystem
from brainpy.errors import MathError
from brainpy.initialize import XavierNormal, ZeroInit, Initializer, parameter
from brainpy.modes import Mode, TrainingMode, training
from brainpy.tools.checking import check_initializer
from brainpy.types import Array

__all__ = [
  'Dense',
]


class Dense(DynamicalSystem):
  r"""A linear transformation applied over the last dimension of the input.

  Mathematically, this node can be defined as:

  .. math::

     y = x  \cdot W + b

  Parameters
  ----------
  num_in: int
    The number of the input feature. A positive integer.
  num_out: int
    The number of the output features. A positive integer.
  W_initializer: optional, Initializer
    The weight initialization.
  b_initializer: optional, Initializer
    The bias initialization.
  mode: Mode
    Enable training this node or not. (default True)
  """

  def __init__(
      self,
      num_in: int,
      num_out: int,
      W_initializer: Union[Initializer, Callable, Array] = XavierNormal(),
      b_initializer: Optional[Union[Initializer, Callable, Array]] = ZeroInit(),
      mode: Mode = training,
      name: str = None,
  ):
    super(Dense, self).__init__(mode=mode, name=name)

    # shape
    self.num_in = num_in
    self.num_out = num_out
    if num_in < 0:
      raise ValueError(f'Received an invalid value for `num_out`, expected '
                       f'a positive integer. Received: num_in={num_in}')
    if num_out < 0:
      raise ValueError(f'Received an invalid value for `num_out`, expected '
                       f'a positive integer. Received: num_out={num_out}')

    # weight initializer
    self.weight_initializer = W_initializer
    self.bias_initializer = b_initializer
    check_initializer(W_initializer, 'weight_initializer')
    check_initializer(b_initializer, 'bias_initializer', allow_none=True)

    # parameter initialization
    self.W = parameter(self.weight_initializer, (num_in, self.num_out))
    self.b = parameter(self.bias_initializer, (self.num_out,))
    if isinstance(self.mode, TrainingMode):
      self.W = bm.TrainVar(self.W)
      self.b = None if (self.b is None) else bm.TrainVar(self.b)

  def __repr__(self):
    return (f'{self.__class__.__name__}(name={self.name}, '
            f'num_in={self.num_in}, '
            f'num_out={self.num_out}, '
            f'mode={self.mode})')

  def reset_state(self, batch_size=None):
    pass

  def update(self, sha, x):
    res = x @ self.W
    if self.b is not None:
      res += self.b

    # online fitting data
    if sha.get('fit', False) and self.online_fit_by is not None:
      self.fit_record['input'] = x
      self.fit_record['output'] = res

    # offline fitting data
    if sha.get('fit', False) and self.offline_fit_by is not None:
      self.fit_record['input'] = x
      self.fit_record['output'] = res
    return res

  def online_init(self):
    if self.b is None:
      num_input = self.num_in
    else:
      num_input = self.num_in + 1
    self.online_fit_by.initialize(feature_in=num_input, feature_out=self.num_out, identifier=self.name)

  def online_fit(self,
                 target: Array,
                 fit_record: Dict[str, Array]):
    if not isinstance(target, (bm.ndarray, jnp.ndarray)):
      raise MathError(f'"target" must be a tensor, but got {type(target)}')
    x = fit_record['input']
    y = fit_record['output']
    if x.ndim != 2:
      raise ValueError(f'"ff" must be a 2D tensor with shape of (num_sample, '
                       f'num_feature), but we got {x.shape}')
    if target.ndim != 2:
      raise ValueError(f'"target" must be a 2D tensor with shape of (num_sample, '
                       f'num_feature), but we got {target.shape}')
    if x.shape[0] != target.shape[0]:
      raise ValueError(f'Batch size of the input and target data should be '
                       f'the same, while we got {x.shape[0]} != {target.shape[0]}.')
    if target.shape[1] != y.shape[1]:
      raise MathError(f'The output dimension of output and target data should be '
                      f'the same, while we got {target.shape[1]} != {y.shape[1]}')

    # data
    if self.b is not None:
      x = bm.concatenate([bm.ones((x.shape[0], 1)), x], axis=-1)

    # fitting
    dW = self.online_fit_by.call(target=target, input=x, output=y, identifier=self.name)

    # assign trained weights
    if self.b is None:
      self.W += dW
    else:
      db, dW = bm.split(dW, [1])
      self.b += db[0]
      self.W += dW

  def offline_init(self):
    if self.b is None:
      num_input = self.num_in + 1
    else:
      num_input = self.num_in
    self.offline_fit_by.initialize(feature_in=num_input, feature_out=self.num_out, identifier=self.name)

  def offline_fit(self,
                  target: Array,
                  fit_record: Dict[str, Array]):
    """The offline training interface for the Dense node."""
    # data checking
    if not isinstance(target, (bm.ndarray, jnp.ndarray)):
      raise MathError(f'"targets" must be a tensor, but got {type(target)}')
    xs = fit_record['input']
    ys = fit_record['output']
    if xs.ndim != 3:
      raise ValueError(f'"ffs" must be a 3D tensor with shape of (num_sample, num_time, '
                       f'num_feature), but we got {xs.shape}')
    if target.ndim != 3:
      raise ValueError(f'"targets" must be a 3D tensor with shape of (num_sample, num_time, '
                       f'num_feature), but we got {target.shape}')
    if ys.shape != target.shape:
      raise ValueError(f'The shapes of output and target data should be '
                       f'the same, while we got {ys.shape} != {target.shape}.')
    if xs.shape[0] != target.shape[0]:
      raise ValueError(f'Batch size of the input and target data should be '
                       f'the same, while we got {xs.shape[0]} != {target.shape[0]}.')
    if xs.shape[1] != target.shape[1]:
      raise MathError(f'The time dimension of input and target data should be '
                      f'the same, while we got {xs.shape[1]} != {target.shape[1]}')

    # get input and target training data
    if self.b is not None:
      xs = bm.concatenate([bm.ones(xs.shape[:2] + (1,)), xs], axis=-1)  # (..., 1 + num_ff_input)

    # solve weights by offline training methods
    weights = self.offline_fit_by(self.name, target, xs, ys)

    # assign trained weights
    if self.b is None:
      self.W.value = weights
    else:
      bias, Wff = bm.split(weights, [1])
      self.W.value = Wff
      self.b.value = bias[0]
