# -*- coding: utf-8 -*-

from typing import Union, Dict, Callable, Optional

import brainpy.math as bm
from brainpy.connect import TwoEndConnector
from brainpy.dyn.base import NeuGroup, SynapseOutput, SynapsePlasticity, TwoEndConn
from brainpy.initialize import Initializer, init_param
from brainpy.types import Tensor
from ..synouts import CUBA

__all__ = [
  'GapJunction',
]


class GapJunction(TwoEndConn):
  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      conn_type: str = 'dense',
      output: SynapseOutput = None,
      plasticity: Optional[SynapsePlasticity] = None,
      g_max: Union[float, Tensor, Initializer, Callable] = 1.,
      name: str = None,
  ):
    super(GapJunction, self).__init__(pre=pre,
                                      post=post,
                                      conn=conn,
                                      output=CUBA() if output is None else output,
                                      plasticity=plasticity,
                                      name=name)
    # checking
    self.check_pre_attrs('V', 'spike')
    self.check_post_attrs('V', 'input', 'spike')

    # connections
    self.conn_type = conn_type
    if conn_type == 'dense':
      self.conn_mat = self.conn.require('conn_mat')
      self.weights = init_param(g_max, (pre.num, post.num), allow_none=False)
    elif conn_type == 'sparse':
      self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')
      self.weights = init_param(g_max, self.pre_ids.shape, allow_none=False)
    else:
      raise ValueError

  def update(self, t, dt):
    self.output.update(t, dt)
    self.plasticity.update(t, dt, self.pre.spike, self.post.spike)
    if self.conn_type == 'dense':
      # pre -> post
      diff = (self.pre.V.reshape((-1, 1)) - self.post.V) * self.conn_mat * self.weights
      self.post.input += self.output.filter(bm.einsum('ij->j', diff))
      # post -> pre
      self.pre.input += self.output.filter(bm.einsum('ij->i', -diff))
    else:
      diff = (self.pre.V[self.pre_ids] - self.post.V[self.post_ids]) * self.weights
      self.post.input += self.output.filter(bm.syn2post_sum(diff, self.post_ids, self.post.num))
      self.pre.input += self.output.filter(bm.syn2post_sum(-diff, self.pre_ids, self.pre.num))

  def reset(self):
    self.output.reset()
    self.plasticity.reset()

