# -*- coding: utf-8 -*-


import brainpy as bp

# bp.math.use_backend('jax')


class LIF(bp.NeuGroup):
  def __init__(self, size, V_L=-70., V_reset=-70., V_th=-50.,
               Cm=0.5, gL=0.025, t_refractory=2., **kwargs):
    super(LIF, self).__init__(size=size, **kwargs)

    self.V_L = V_L
    self.V_reset = V_reset
    self.V_th = V_th
    self.Cm = Cm
    self.gL = gL
    self.t_refractory = t_refractory

    self.V = bp.math.Variable(bp.math.ones(self.num) * V_L)
    self.input = bp.math.Variable(bp.math.zeros(self.num))
    self.spike = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
    self.refractory = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
    self.t_last_spike = bp.math.Variable(bp.math.ones(self.num) * -1e7)

  @bp.odeint
  def integral(self, V, t, Iext):
    dVdt = (- self.gL * (V - self.V_L) - Iext) / self.Cm
    return dVdt

  def update(self, _t, _i):
    ref = (_t - self.t_last_spike) <= self.t_refractory
    V = self.integral(self.V, _t, self.input)
    V = bp.math.where(ref, self.V, V)
    spike = (V >= self.V_th)
    self.V[:] = bp.math.where(spike, self.V_reset, V)
    self.spike[:] = spike
    self.t_last_spike[:] = bp.math.where(spike, _t, self.t_last_spike)
    self.refractory[:] = bp.math.logical_or(spike, ref)
    self.input[:] = 0.


if __name__ == '__main__':
  group = bp.math.jit(LIF(100, monitors=['V']))

  group.run(duration=200., inputs=('input', -1.), report=0.1)
  bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

  group.run(duration=(200, 400.), report=0.1)
  bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)
