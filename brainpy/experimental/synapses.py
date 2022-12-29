# -*- coding: utf-8 -*-

from typing import Union, Optional

import brainpylib as bl
import jax

from brainpy import (math as bm,
                     initialize as init,
                     connect)
from brainpy.dyn.base import DynamicalSystem, SynSTP
from brainpy.integrators.ode import odeint
from brainpy.types import Initializer, ArrayType

__all__ = [
  'Exponential',
]


class Exponential(DynamicalSystem):
  def __init__(
      self,
      conn: connect.TwoEndConnector,
      stp: Optional[SynSTP] = None,
      g_max: Union[float, Initializer] = 1.,
      g_initializer: Union[float, Initializer] = init.ZeroInit(),
      tau: Union[float, ArrayType] = 8.0,
      method: str = 'exp_auto',
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super(Exponential, self).__init__(name=name, mode=mode)

    # component
    self.conn = conn
    self.stp = stp
    self.g_initializer = g_initializer
    assert self.conn.pre_num is not None
    assert self.conn.post_num is not None

    # parameters
    self.tau = tau
    if bm.size(self.tau) != 1:
      raise ValueError(f'"tau" must be a scalar or a tensor with size of 1. But we got {self.tau}')

    # connections and weights
    if isinstance(self.conn, connect.One2One):
      self.g_max = init.parameter(g_max, (self.conn.pre_num,), allow_none=False)

    elif isinstance(self.conn, connect.All2All):
      self.g_max = init.parameter(g_max, (self.conn.pre_num, self.conn.post_num), allow_none=False)

    else:
      self.conn_mask = self.conn.require('pre2post')
      self.g_max = init.parameter(g_max, self.conn_mask[0].shape, allow_none=False)

    # variables
    self.g = init.variable_(g_initializer, self.conn.post_num, self.mode)

    # function
    self.integral = odeint(lambda g, t: -g / self.tau, method=method)

  def reset_state(self, batch_size=None):
    self.g.value = init.variable_(bm.zeros, self.conn.post_num, batch_size)
    if self.stp is not None:
      self.stp.reset_state(batch_size)

  def _syn2post_with_one2one(self, syn_value, syn_weight):
    return syn_value * syn_weight

  def _syn2post_with_all2all(self, syn_value, syn_weight):
    if bm.ndim(syn_weight) == 0:
      if isinstance(self.mode, bm.BatchingMode):
        assert syn_value.ndim == 2
        post_vs = bm.sum(syn_value, keepdims=True, axis=1)
      else:
        post_vs = bm.sum(syn_value)
      if not self.conn.include_self:
        post_vs = post_vs - syn_value
      post_vs = syn_weight * post_vs
    else:
      assert syn_weight.ndim == 2
      if isinstance(self.mode, bm.BatchingMode):
        assert syn_value.ndim == 2
        post_vs = syn_value @ syn_weight
      else:
        post_vs = syn_value @ syn_weight
    return post_vs

  def update(self, tdi, spike):
    t, dt = tdi['t'], tdi.get('dt', bm.dt)

    # update sub-components
    if self.stp is not None:
      self.stp.update(tdi, spike)

    # post values
    if isinstance(self.conn, connect.All2All):
      syn_value = bm.asarray(spike, dtype=bm.float_)
      if self.stp is not None:
        syn_value = self.stp(syn_value)
      post_vs = self._syn2post_with_all2all(syn_value, self.g_max)
    elif isinstance(self.conn, connect.One2One):
      syn_value = bm.asarray(spike, dtype=bm.float_)
      if self.stp is not None:
        syn_value = self.stp(syn_value)
      post_vs = self._syn2post_with_one2one(syn_value, self.g_max)
    else:
      if isinstance(self.mode, bm.BatchingMode):
        f = jax.vmap(bl.event_ops.event_csr_matvec, in_axes=(None, None, None, 0))
        post_vs = f(self.g_max, self.conn_mask[0], self.conn_mask[1], spike,
                    shape=(self.conn.pre_num, self.conn.post_num), transpose=True)
      else:
        post_vs = bl.event_ops.event_csr_matvec(
          self.g_max, self.conn_mask[0], self.conn_mask[1], spike,
          shape=(self.conn.pre_num, self.conn.post_num), transpose=True
        )
    # updates
    self.g.value = self.integral(self.g.value, t, dt) + post_vs

    # output
    return self.g.value
