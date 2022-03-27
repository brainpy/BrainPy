# -*- coding: utf-8 -*-


from typing import Sequence, Optional, Dict, Callable, Union

import brainpy.math as bm
from brainpy.initialize import XavierNormal, ZeroInit, Initializer
from brainpy.nn.nodes.base.dense import Dense
from brainpy.tools.checking import check_shape_consistency
from brainpy.types import Tensor

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

  def __init__(
      self,
      num_unit: int,
      weight_initializer: Union[Initializer, Callable, Tensor] = XavierNormal(),
      bias_initializer: Optional[Union[Initializer, Callable, Tensor]] = ZeroInit(),
      **kwargs
  ):
    super(LinearReadout, self).__init__(num_unit=num_unit,
                                        weight_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        **kwargs)

  def init_state(self, num_batch=1):
    return bm.zeros((num_batch,) + self.output_shape[1:], dtype=bm.float_)

  def forward(self, ff, fb=None, **shared_kwargs):
    h = super(LinearReadout, self).forward(ff, fb=fb, **shared_kwargs)
    self.state.value = h
    return h

  def __force_init__(self, train_pars: Optional[Dict] = None):
    if train_pars is None: train_pars = dict()
    alpha = train_pars.get('alpha')
    batch_size, free_shapes = check_shape_consistency(self.feedforward_shapes, -1, True)
    assert batch_size == (1,), f'{self.name} only supports batch size is 1. But got {batch_size}'
    num_input = sum(free_shapes)
    if self.bias is not None:
      num_input += 1
    self.P = bm.Variable(bm.eye(num_input) * alpha)  # inverse correlation matrix

  def __force_train__(
      self,
      ffs: Sequence[Tensor],
      target: Tensor,
      train_pars: Optional[Dict] = None
  ):
    for ff in ffs:
      assert ff.ndim == 3
    # note each "ff" has a batch size = 1
    ffs = bm.concatenate(ffs, axis=-1)
    assert ffs.ndim == target.ndim == 1, '"x" and "y" must be a one-dimensional tensors.'
    if self.bias is not None:
      ffs = bm.concatenate([ffs, bm.zeros(1)])
    ffs = bm.expand_dims(ffs, axis=1)
    # update the inverse correlation matrix
    k = bm.expand_dims(bm.dot(self.P, ffs), axis=1)  # (num_hidden, 1)
    hPh = bm.dot(ffs.T, k)  # (1,)
    c = 1.0 / (1.0 + hPh)  # (1,)
    self.P -= bm.dot(k * c, k.T)  # (num_hidden, num_hidden)
    # update the weights
    e = bm.atleast_2d(self.state - target)  # (1, num_output)
    dw = bm.dot(-c * k, e)  # (num_hidden, num_output)
    self.Wff += dw
