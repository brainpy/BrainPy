# -*- coding: utf-8 -*-

from typing import Union, Dict, Callable, Optional

import brainpy.math as bm
from brainpy.connect import TwoEndConnector
from brainpy.dyn.base import NeuGroup, SynOut, SynSTP, TwoEndConn
from brainpy.initialize import Initializer, parameter
from brainpy.types import Array
from ..synouts import CUBA

__all__ = [
  'GapJunction',
]


class GapJunction(TwoEndConn):
  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Array, Dict[str, Array]],
      comp_method: str = 'dense',
      g_max: Union[float, Array, Initializer, Callable] = 1.,
      name: str = None,
  ):
    super(GapJunction, self).__init__(pre=pre,
                                      post=post,
                                      conn=conn,
                                      name=name)
    # checking
    self.check_pre_attrs('V')
    self.check_post_attrs('V', 'input')

    # assert isinstance(self.output, _NullSynOut)
    # assert isinstance(self.stp, _NullSynSTP)

    # connections
    self.comp_method = comp_method
    if comp_method == 'dense':
      self.conn_mat = self.conn.require('conn_mat')
      self.weights = parameter(g_max, (pre.num, post.num), allow_none=False)
    elif comp_method == 'sparse':
      self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')
      self.weights = parameter(g_max, self.pre_ids.shape, allow_none=False)
    else:
      raise ValueError

  def update(self, tdi):
    if self.comp_method == 'dense':
      # pre -> post
      diff = (self.pre.V.reshape((-1, 1)) - self.post.V) * self.conn_mat * self.weights
      self.post.input += bm.einsum('ij->j', diff)
      # post -> pre
      self.pre.input += bm.einsum('ij->i', -diff)
    else:
      diff = (self.pre.V[self.pre_ids] - self.post.V[self.post_ids]) * self.weights
      self.post.input += bm.syn2post_sum(diff, self.post_ids, self.post.num)
      self.pre.input += bm.syn2post_sum(-diff, self.pre_ids, self.pre.num)

  def reset_state(self, batch_size=None):
    pass
