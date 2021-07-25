# -*- coding: utf-8 -*-
import jax.lax

import brainpy as bp

bp.math.use_backend('jax')


class LIF(bp.NeuGroup):
  def __init__(self, size, t_refractory=1., V_rest=0.,
               V_reset=-5., V_th=20., R=1., tau=10., **kwargs):
    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.R = R
    self.tau = tau
    self.t_refractory = t_refractory

    # variables
    self.V = bp.math.ones(size) * V_reset
    self.input = bp.math.zeros(size)
    self.t_last_spike = bp.math.ones(size) * -1e7
    self.spike = bp.math.zeros(size, dtype=bool)
    self.refractory = bp.math.zeros(size, dtype=bool)

    super(LIF, self).__init__(size=size, **kwargs)

  @bp.odeint
  def int_V(self, V, t, Iext):
    return (- (V - self.V_rest) + self.R * Iext) / self.tau

  @bp.math.control_transform
  def update(self, _t, _i):
    for i in range(self.num):
      if _t - self.t_last_spike[i] <= self.t_refractory:
        self.refractory[i] = True
      else:
        V = self.int_V(self.V[i], _t, self.input[i])
        if V >= self.V_th:
          self.V[i] = self.V_reset
          self.spike[i] = 1.
          self.t_last_spike[i] = _t
          self.refractory[i] = True
        else:
          self.spike[i] = 0.
          self.V[i] = V
          self.refractory[i] = False
      self.input[i] = 0.


class LIF2(bp.NeuGroup):
  def __init__(self, size, t_refractory=1., V_rest=0.,
               V_reset=-5., V_th=20., R=1., tau=10., **kwargs):
    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.R = R
    self.tau = tau
    self.t_refractory = t_refractory

    # variables
    self.V = bp.math.ones(size) * V_reset
    self.input = bp.math.zeros(size)
    self.t_last_spike = bp.math.ones(size) * -1e7
    self.spike = bp.math.zeros(size, dtype=bool)
    self.refractory = bp.math.zeros(size, dtype=bool)

    super(LIF2, self).__init__(size=size, **kwargs)

  @bp.odeint
  def int_V(self, V, t, Iext):
    return (- (V - self.V_rest) + self.R * Iext) / self.tau

  @bp.math.control_transform
  def update(self, _t, _i):

    def true_func2(val2):
      i, V, _V = val2
      _V['V'][i] = _V['V_reset']
      _V['spike'][i] = True
      _V['t_last_spike'][i] = _V['_t']
      _V['refractory'][i] = True
      return _V

    def false_func2(val2):
      i, V, _V = val2
      _V['spike'][i] = False
      _V['V'][i] = V
      _V['refractory'][i] = False
      return _V

    def false_func1(val2):
      i, _V = val2
      V = self.int_V(_V['V'][i], _V['_t'], _V['input'][i])
      _V = jax.lax.cond(V >= _V['V_th'], true_func2, false_func2, (i, V, _V))
      return _V

    def true_fun1(val2):
      i, _V = val2
      _V['refractory'][i] = True
      return _V

    def loop_func(i, val):
      val = jax.lax.cond(val['_t'] - val['t_last_spike'][i] <= val['t_refractory'],
                         true_fun1,
                         false_func1,
                         (i, val))
      val['input'][i] = 0.
      return val

    loop_val = dict(t_last_spike=self.t_last_spike,
                    t_refractory=self.t_refractory,
                    refractory=self.refractory,
                    V_th=self.V_th,
                    V=self.V,
                    input=self.input,
                    V_reset=self.V_reset,
                    spike=self.spike,
                    _t=_t,
                    _i=_i)
    loop_val = jax.lax.fori_loop(0, self.num, loop_func, loop_val)
    self.t_last_spike.value = loop_val['t_last_spike']
    self.refractory.value = loop_val['refractory']
    self.V.value = loop_val['V']
    self.input.value = loop_val['input']
    self.spike.value = loop_val['spike']
    # Problems
    # 1. where value is the ndarray
    # 2.


if __name__ == '__main__':
  # group = LIF(100, monitors=['V'])
  group = LIF2(100, monitors=['V'])
  group = bp.math.jit(group)

  group.run(duration=200., inputs=('input', 26.), report=0.1)
  bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

  group.run(duration=(200, 400.), report=0.1)
  bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)
