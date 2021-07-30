# -*- coding: utf-8 -*-

import jax

import brainpy as bp

bp.math.use_backend('jax')


class GapJunction(bp.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=1., **kwargs):
    super(GapJunction, self).__init__(pre=pre, post=post, conn=conn, **kwargs)

    # connections
    self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
    self.num = len(self.pre_ids)

    # parameters
    self.g_max = bp.math.ones(self.num) * g_max

    # checking
    assert hasattr(pre, 'V'), 'Pre-synaptic group must has "V" variable.'
    assert hasattr(post, 'V'), 'Post-synaptic group must has "V" variable.'
    assert hasattr(post, 'input'), 'Post-synaptic group must has "input" variable.'

  def update(self, _t, _i):
    def loop_body(i, val):
      pre_id = val['pre_ids'][i]
      post_id = val['post_ids'][i]
      diff = (val['pre_V'][pre_id] - val['post_V'][post_id])
      val['post_input'][post_id] += val['g_max'][i] * diff
      return val

    val = jax.lax.fori_loop(0, self.num, loop_body,
                            dict(pre_ids=self.pre_ids, post_ids=self.post_ids,
                                 post_input=self.post.input, pre_V=self.pre.V,
                                 post_V=self.post.V, g_max=self.g_max))
    self.post.input.value = val['post_input']


class LifGapJunction(bp.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=0.1, k_spikelet=0.1, **kwargs):
    super(LifGapJunction, self).__init__(pre=pre, post=post, conn=conn, **kwargs)

    # parameters
    self.k_spikelet = k_spikelet

    # connections
    self.conn = conn(pre.size, post.size)
    self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
    self.num = len(self.pre_ids)

    # variables
    self.g_max = bp.math.ones(self.num) * g_max

    # checking
    assert hasattr(pre, 'V'), 'Pre-synaptic group must has "V" variable.'
    assert hasattr(post, 'V'), 'Post-synaptic group must has "V" variable.'
    assert hasattr(post, 'input'), 'Post-synaptic group must has "input" variable.'

  def update(self, _t, _i):
    def loop_body(i, val):
      pre_id = val['pre_ids'][i]
      post_id = val['post_ids'][i]
      spikelet = val['g_max'][i] * val['k_spikelet'] * val['pre_spike'][pre_id]
      diff = (val['pre_V'][pre_id] - val['post_V'][post_id])
      val['post_input'][post_id] += val['g_max'][i] * diff
      val['post_V'][post_id] += spikelet * (1. - val['post_ref'][post_id])
      return val

    val = jax.lax.fori_loop(0, self.num, loop_body,
                            dict(pre_ids=self.pre_ids, pre_spike=self.pre.spike,
                                 post_input=self.post.input, pre_V=self.pre.V,
                                 post_ids=self.post_ids, post_V=self.post.V,
                                 post_ref=self.post.refractory,
                                 g_max=self.g_max, k_spikelet=self.k_spikelet,
                                 ))
    self.post.input.value = val['post_input']
    self.post.V.value = val['post_V']


class LIF(bp.NeuGroup):
  target_backend = 'general'

  @staticmethod
  def derivative(V, t, Iext, V_rest, R, tau):
    dvdt = (-V + V_rest + R * Iext) / tau
    return dvdt

  def __init__(self, size, t_refractory=1., V_rest=0.,
               V_reset=-5., V_th=20., R=1., tau=10., **kwargs):
    super(LIF, self).__init__(size=size, **kwargs)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.R = R
    self.tau = tau
    self.t_refractory = t_refractory

    # variables
    self.t_last_spike = bp.math.ones(self.num) * -1e7
    self.input = bp.math.zeros(self.num)
    self.V = bp.math.ones(self.num) * V_rest
    self.refractory = bp.math.zeros(self.num, dtype=bool)
    self.spike = bp.math.zeros(self.num, dtype=bool)

    # integrator
    self.integral = bp.odeint(self.derivative)

  def update(self, _t, _i):
    refractory = (_t - self.t_last_spike) <= self.t_refractory
    V = self.integral(self.V, _t, self.input, self.V_rest, self.R, self.tau)
    V = bp.math.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike[:] = bp.math.where(spike, _t, self.t_last_spike)
    self.V[:] = bp.math.where(spike, self.V_reset, V)
    self.refractory[:] = refractory | spike
    self.input[:] = 0.
    self.spike[:] = spike


def example():
  lif = LIF(100, monitors=['V'])
  gj = LifGapJunction(lif, lif, bp.connect.FixedProb(0.2))
  net = bp.Network(lif=lif, gj=gj)
  net = bp.math.jit(net)

  net.run(100., report=0.2)


if __name__ == '__main__':
  example()

