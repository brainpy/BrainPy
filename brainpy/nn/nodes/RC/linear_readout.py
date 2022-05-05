# -*- coding: utf-8 -*-

import jax.numpy as jnp

import brainpy.math as bm
from brainpy.errors import MathError
from brainpy.initialize import Initializer
from brainpy.nn.datatypes import MultipleData
from brainpy.nn.nodes.base.dense import Dense
from brainpy.tools.checking import check_shape_consistency

__all__ = [
  'LinearReadout',
]


class LinearReadout(Dense):
  """Linear readout node. Different from ``Dense``, this node has its own state.

  Parameters
  ----------
  num_unit: int
    The number of output features. A positive integer.
  weight_initializer: Initializer
    The weight initializer.
  bias_initializer: Optional, Initializer
    The bias initializer.
  trainable: bool
    Default is true.
  """
  data_pass = MultipleData('sequence')

  def __init__(self, num_unit: int, **kwargs):
    super(LinearReadout, self).__init__(num_unit=num_unit, **kwargs)

  def init_state(self, num_batch=1):
    return bm.zeros((num_batch,) + self.output_shape[1:], dtype=bm.float_)

  def forward(self, ff, fb=None, **shared_kwargs):
    h = super(LinearReadout, self).forward(ff, fb=fb, **shared_kwargs)
    self.state.value = h
    return h

  def online_init(self):
    _, free_shapes = check_shape_consistency(self.feedforward_shapes, -1, True)
    num_input = sum(free_shapes)
    if self.bias is not None:
      num_input += 1
    if self.feedback_shapes is not None:
      _, free_shapes = check_shape_consistency(self.feedback_shapes, -1, True)
      num_input += sum(free_shapes)
    self.online_fit_by.initialize(feature_in=num_input,
                                  feature_out=self.num_unit,
                                  name=self.name)

  def online_fit(self, target, ff, fb=None):
    if not isinstance(target, (bm.ndarray, jnp.ndarray)):
      raise MathError(f'"target" must be a tensor, but got {type(target)}')
    ff = bm.concatenate(ff, axis=-1)
    if ff.ndim != 2:
      raise ValueError(f'"ff" must be a 2D tensor with shape of (num_sample, '
                       f'num_feature), but we got {ff.shape}')
    if target.ndim != 2:
      raise ValueError(f'"target" must be a 2D tensor with shape of (num_sample, '
                       f'num_feature), but we got {target.shape}')
    if ff.shape[0] != target.shape[0]:
      raise ValueError(f'Batch size of the input and target data should be '
                       f'the same, while we got {ff.shape[0]} != {target.shape[0]}.')
    if target.shape[1] != self.state.shape[1]:
      raise MathError(f'The output dimension of output and target data should be '
                      f'the same, while we got {target.shape[1]} != {self.state.shape[1]}')
    if fb is not None:
      fb = bm.concatenate(fb, axis=-1)
      if fb.ndim != 2:
        raise ValueError(f'"fb" must be a 2D tensor with shape of (num_sample, '
                         f'num_feature), but we got {fb.shape}')
      if ff.shape[0] != fb.shape[0]:
        raise ValueError(f'Batch size of the feedforward and the feedback inputs should be '
                         f'the same, while we got {ff.shape[0]} != {fb.shape[0]}.')

    # data
    inputs = ff
    num_ff_input = ff.shape[1]
    if fb is not None:
      inputs = bm.concatenate([inputs, fb], axis=-1)
    if self.bias is not None:
      inputs = bm.concatenate([bm.ones((inputs.shape[0], 1)), inputs], axis=-1)

    # fitting
    dW = self.online_fit_by.call(target=target, input=inputs, output=self.state, name=self.name)

    # assign trained weights
    if self.bias is None:
      if fb is None:
        self.Wff += dW
      else:
        dWff, dWfb = bm.split(dW, [num_ff_input])
        self.Wff += dWff
        self.Wfb += dWfb
    else:
      if fb is None:
        db, dWff = bm.split(dW, [1])
        self.bias += db[0]
        self.Wff += dWff
      else:
        db, dWff, dWfb = bm.split(dW, [1, 1 + num_ff_input])
        self.bias += db[0]
        self.Wff += dWff
        self.Wfb += dWfb
