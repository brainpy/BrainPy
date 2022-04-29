# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

bp.math.set_platform('cpu')

Cm = 200  # Membrane Capacitance [pF]
gl = 10.  # Leak Conductance   [nS]
g_Na = 20. * 1000
g_Kd = 6. * 1000  # K Conductance      [nS]
El = -60.  # Resting Potential [mV]
ENa = 50.  # reversal potential (Sodium) [mV]
EK = -90.  # reversal potential (Potassium) [mV]
VT = -63.
V_th = -20.
taue = 5.  # Excitatory synaptic time constant [ms]
taui = 10.  # Inhibitory synaptic time constant [ms]
Ee = 0.  # Excitatory reversal potential (mV)
Ei = -80.  # Inhibitory reversal potential (Potassium) [mV]
we = 6.  # excitatory synaptic conductance [nS]
wi = 67.  # inhibitory synaptic conductance [nS]


class HH(bp.dyn.NeuGroup):
  def __init__(self, size, method='exp_auto'):
    super(HH, self).__init__(size)

    # variables
    self.V = bm.Variable(El + (bm.random.randn(self.num) * 5 - 5))
    self.m = bm.Variable(bm.zeros(self.num))
    self.n = bm.Variable(bm.zeros(self.num))
    self.h = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.input = bm.Variable(bm.zeros(size))

    def dV(V, t, m, h, n, Isyn):
      gna = g_Na * (m * m * m) * h
      gkd = g_Kd * (n * n * n * n)
      dVdt = (-gl * (V - El) - gna * (V - ENa) - gkd * (V - EK) + Isyn) / Cm
      return dVdt

    def dm(m, t, V, ):
      m_alpha = 0.32 * (13 - V + VT) / (bm.exp((13 - V + VT) / 4) - 1.)
      m_beta = 0.28 * (V - VT - 40) / (bm.exp((V - VT - 40) / 5) - 1)
      dmdt = (m_alpha * (1 - m) - m_beta * m)
      return dmdt

    def dh(h, t, V):
      h_alpha = 0.128 * bm.exp((17 - V + VT) / 18)
      h_beta = 4. / (1 + bm.exp(-(V - VT - 40) / 5))
      dhdt = (h_alpha * (1 - h) - h_beta * h)
      return dhdt

    def dn(n, t, V):
      c = 15 - V + VT
      n_alpha = 0.032 * c / (bm.exp(c / 5) - 1.)
      n_beta = .5 * bm.exp((10 - V + VT) / 40)
      dndt = (n_alpha * (1 - n) - n_beta * n)
      return dndt

    # functions
    self.integral = bp.odeint(bp.JointEq([dV, dm, dh, dn]), method=method)

  def update(self, t, dt):
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, t, Isyn=self.input, dt=dt)
    self.spike.value = bm.logical_and(self.V < V_th, V >= V_th)
    self.m.value = m
    self.h.value = h
    self.n.value = n
    self.V.value = V
    self.input[:] = 0.


class COBAHH(bp.dyn.Network):
  def __init__(self, scale=1., method='exp_auto'):
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)
    E = HH(num_exc, method=method)
    I = HH(num_inh, method=method)
    E2E = bp.dyn.ExpCOBA(pre=E, post=E, conn=bp.conn.FixedProb(prob=0.02),
                         E=Ee, g_max=we / scale, tau=taue, method=method)
    E2I = bp.dyn.ExpCOBA(pre=E, post=I, conn=bp.conn.FixedProb(prob=0.02),
                         E=Ee, g_max=we / scale, tau=taue, method=method)
    I2E = bp.dyn.ExpCOBA(pre=I, post=E, conn=bp.conn.FixedProb(prob=0.02),
                         E=Ei, g_max=wi / scale, tau=taui, method=method)
    I2I = bp.dyn.ExpCOBA(pre=I, post=I, conn=bp.conn.FixedProb(prob=0.02),
                         E=Ei, g_max=wi / scale, tau=taui, method=method)

    super(COBAHH, self).__init__(E2E, E2I, I2I, I2E, E=E, I=I)


net = COBAHH(scale=1)
runner = bp.dyn.DSRunner(net, monitors=['E.spike'])
t = runner.run(100.)
print(t)
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)
