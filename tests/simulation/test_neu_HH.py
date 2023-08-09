import brainpy as bp
import brainpy.math as bm
import unittest

show = False


class HH(bp.dyn.CondNeuGroup):
  def __init__(self, size):
    super(HH, self).__init__(size)
    self.INa = bp.channels.INa_HH1952(size, )
    self.IK = bp.channels.IK_HH1952(size, )
    self.IL = bp.channels.IL(size, E=-54.387, g_max=0.03)


class HHv2(bp.dyn.NeuDyn):
  def __init__(self, size, ENa=50., gNa=120., EK=-77., gK=36., EL=-54.387, gL=0.03, V_th=20., C=1.0):
    super().__init__(size=size)

    # initialize parameters
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.gNa = gNa
    self.gK = gK
    self.gL = gL
    self.C = C
    self.V_th = V_th

    # initialize variables
    self.V = bm.Variable(bm.random.randn(self.num) - 70.)
    self.m = bm.Variable(0.5 * bm.ones(self.num))
    self.h = bm.Variable(0.6 * bm.ones(self.num))
    self.n = bm.Variable(0.32 * bm.ones(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

    # integral functions
    self.int_V = bp.odeint(f=self.dV, method='exp_auto')
    self.int_m = bp.odeint(f=self.dm, method='exp_auto')
    self.int_h = bp.odeint(f=self.dh, method='exp_auto')
    self.int_n = bp.odeint(f=self.dn, method='exp_auto')

  def dV(self, V, t, m, h, n, Iext):
    I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
    I_K = (self.gK * n ** 4.0) * (V - self.EK)
    I_leak = self.gL * (V - self.EL)
    dVdt = (- I_Na - I_K - I_leak + Iext) / self.C
    return dVdt

  def dm(self, m, t, V):
    alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
    beta = 4.0 * bm.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m
    return dmdt

  def dh(self, h, t, V):
    alpha = 0.07 * bm.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bm.exp(-(V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h
    return dhdt

  def dn(self, n, t, V):
    alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
    beta = 0.125 * bm.exp(-(V + 65) / 80)
    dndt = alpha * (1 - n) - beta * n
    return dndt

  def update(self, x=None):
    x = 0. if x is None else x
    _t = bp.share.load('t')
    _dt = bp.share.load('dt')

    dV_grad = bm.vector_grad(self.dV, argnums=0)(self.V.value, _t, self.m.value, self.h.value, self.n.value, x)

    # compute V, m, h, n
    V = self.int_V(self.V, _t, self.m, self.h, self.n, x, dt=_dt)
    self.h.value = self.int_h(self.h, _t, self.V, dt=_dt)
    self.m.value = self.int_m(self.m, _t, self.V, dt=_dt)
    self.n.value = self.int_n(self.n, _t, self.V, dt=_dt)

    # update the spiking state and the last spiking time
    self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    self.t_last_spike.value = bm.where(self.spike, _t, self.t_last_spike)

    # update V
    self.V.value = V

    return dV_grad


class TestHH(unittest.TestCase):
  def test1(self):
    bm.random.seed()
    hh = HH(1)
    I, length = bp.inputs.section_input(values=[0, 5, 0], durations=[10, 100, 10], return_length=True)
    runner = bp.DSRunner(
      hh, monitors=['V', 'INa.p', 'INa.q', 'IK.p'],
      inputs=[hh.input, I, 'iter'],
    )
    runner.run(length)

    if show:
      bp.visualize.line_plot(runner.mon.ts, runner.mon.V, show=True)

  def test2(self):
    bm.random.seed()
    with bp.math.environment(dt=0.1):
      hh = bp.neurons.HH(1)
      looper = bp.LoopOverTime(hh, out_vars=(hh.V, hh.m, hh.n, hh.h))
      grads, (vs, ms, ns, hs) = looper(bm.ones(1000) * 5)

      if show:
        ts = bm.as_numpy(bm.arange(1000) * bm.dt)
        fig, gs = bp.visualize.get_figure(4, 1, 3, 10)
        fig.add_subplot(gs[0, 0])
        bp.visualize.line_plot(ts, vs, legend='v')
        fig.add_subplot(gs[1, 0])
        bp.visualize.line_plot(ts, ms, legend='m')
        bp.visualize.line_plot(ts, hs, legend='h')
        bp.visualize.line_plot(ts, ns, legend='n')
        fig.add_subplot(gs[2, 0])
        bp.visualize.line_plot(ts, grads, legend='grad')
        fig.add_subplot(gs[3, 0])
        bp.visualize.line_plot(ts, bm.exp(grads * bm.dt), show=True)

