# -*- coding: utf-8 -*-


from typing import Sequence, Optional, Dict

import brainpy.math as bm
from brainpy.initialize import XavierNormal, ZeroInit
from brainpy.nn.nodes.base.dense import Dense
from brainpy.nn.utils import check_shape_consistency
from brainpy.types import Tensor

__all__ = [
  'LinearReadout',
]


class LinearReadout(Dense):
  def __init__(self, num_unit, w_init=XavierNormal(), b_init=ZeroInit(), **kwargs):
    super(LinearReadout, self).__init__(num_unit=num_unit, w_init=w_init, b_init=b_init, **kwargs)

  def ff_init(self):
    super(LinearReadout, self).ff_init()
    self.state = bm.Variable(bm.zeros(self.output_shape, dtype=bm.float_))

  def call(self, ff, fb=None, **kwargs):
    self.state.value = super(LinearReadout, self).call(ff, fb=fb, **kwargs)
    return self.state

  def __force_init__(self, train_pars: Optional[Dict] = None):
    if train_pars is None: train_pars = dict()
    alpha = train_pars.get('alpha')
    _, free_shapes = check_shape_consistency(self.input_shapes, -1, True)
    num_input = sum(free_shapes)
    if self.bias is not None:
      num_input += 1
    self.P = bm.Variable(bm.eye(num_input) * alpha)  # inverse correlation matrix

  def __force_train__(self, x: Sequence[Tensor], y: Tensor, train_pars: Optional[Dict] = None):
    x = bm.concatenate(x, axis=-1)
    assert x.ndim == y.ndim == 1
    if self.bias is not None:
      x = bm.concatenate([x, bm.zeros(1)])
    x = bm.expand_dims(x, axis=1)
    # update the inverse correlation matrix
    k = bm.expand_dims(bm.dot(self.P, x), axis=1)  # (num_hidden, 1)
    hPh = bm.dot(x.T, k)  # (1,)
    c = 1.0 / (1.0 + hPh)  # (1,)
    self.P -= bm.dot(k * c, k.T)  # (num_hidden, num_hidden)
    # update the weights
    e = bm.atleast_2d(self.state - y)  # (1, num_output)
    dw = bm.dot(-c * k, e)  # (num_hidden, num_output)
    self.weights += dw
