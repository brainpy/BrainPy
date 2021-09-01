# -*- coding: utf-8 -*-


import sys

sys.path.append(r'/mnt/d/codes/Projects/BrainPy')

import brainpy as bp

bp.math.use_backend('numpy')
bp.math.set_dt(0.01)
bp.integrators.set_default_odeint('rk4')


class HH(bp.NeuGroup):
  def __init__(self, size, ENa=50., EK=-77., EL=-54.387,
               C=1.0, gNa=120., gK=36., gL=0.03, V_th=20.,
               **kwargs):
    super(HH, self).__init__(size=size, **kwargs)

    # parameters
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.C = C
    self.gNa = gNa
    self.gK = gK
    self.gL = gL
    self.V_th = V_th

    # variables
    self.V = bp.math.Variable(bp.math.ones(self.num) * -65.)
    self.m = bp.math.Variable(bp.math.ones(self.num) * 0.5)
    self.h = bp.math.Variable(bp.math.ones(self.num) * 0.6)
    self.n = bp.math.Variable(bp.math.ones(self.num) * 0.32)
    self.spike = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
    self.input = bp.math.Variable(bp.math.zeros(self.num))

  # @bp.odeint(method='exponential_euler')
  @bp.odeint(method='rk4')
  def integral(self, V, m, h, n, t, Iext):
    alpha = 0.1 * (V + 40) / (1 - bp.math.exp(-(V + 40) / 10))
    beta = 4.0 * bp.math.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m

    alpha = 0.07 * bp.math.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bp.math.exp(-(V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h

    alpha = 0.01 * (V + 55) / (1 - bp.math.exp(-(V + 55) / 10))
    beta = 0.125 * bp.math.exp(-(V + 65) / 80)
    dndt = alpha * (1 - n) - beta * n

    I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
    I_K = (self.gK * n ** 4.0) * (V - self.EK)
    I_leak = self.gL * (V - self.EL)
    dVdt = (- I_Na - I_K - I_leak + Iext) / self.C

    return dVdt, dmdt, dhdt, dndt

  def update(self, _t, _i):
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t, self.input)
    self.spike[:] = bp.math.logical_and(self.V < self.V_th, V >= self.V_th)
    self.V[:] = V
    self.m[:] = m
    self.h[:] = h
    self.n[:] = n
    self.input[:] = 0.


class AMPA_vec(bp.TwoEndConn):
  def __init__(self, pre, post, conn, delay=0., g_max=0.10, E=0., tau=2.0, **kwargs):
    super(AMPA_vec, self).__init__(pre=pre, post=post, **kwargs)

    # parameters
    self.g_max = g_max
    self.E = E
    self.tau = tau
    self.delay = delay

    # connections
    self.conn = conn(pre.size, post.size)
    self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
    self.size = len(self.pre_ids)

    # data
    self.s = bp.math.Variable(bp.math.zeros(self.size))
    self.g = self.register_constant_delay('g', size=self.size, delay=delay)

  @bp.odeint(method='euler')
  def int_s(self, s, t):
    return - s / self.tau

  def update(self, _t, _i):
    g = self.g.pull()
    for i in range(self.size):
      pre_id = self.pre_ids[i]
      self.s[i] = self.int_s(self.s[i], _t)
      self.s[i] += self.pre.spike[pre_id]
      post_id = self.post_ids[i]
      self.post.input[post_id] -= g[i] * (self.post.V[post_id] - self.E)
    self.g.push(self.g_max * self.s)


class AMPA_mat(bp.TwoEndConn):
  def __init__(self, pre, post, conn, delay=0., g_max=0.10, E=0., tau=2.0, **kwargs):
    super(AMPA_mat, self).__init__(pre=pre, post=post, **kwargs)

    # parameters
    self.g_max = g_max
    self.E = E
    self.tau = tau
    self.delay = delay

    # connections
    self.conn = conn(pre.size, post.size)
    self.conn_mat = conn.requires('conn_mat')
    self.size = bp.math.shape(self.conn_mat)

    # variables
    self.s = bp.math.Variable(bp.math.zeros(self.size))
    self.g = self.register_constant_delay('g', size=self.size, delay=delay)

  @bp.odeint
  def int_s(self, s, t):
    return - s / self.tau

  def update(self, _t, _i):
    self.s[:] = self.int_s(self.s, _t)
    for i in range(self.pre.size[0]):
      if self.pre.spike[i] > 0:
        self.s[i] += self.conn_mat[i]
    self.g.push(self.g_max * self.s)
    g = self.g.pull()
    self.post.input[:] -= bp.math.sum(g, axis=0) * (self.post.V - self.E)


if __name__ == '__main__':
  hh = HH(100, monitors=['V'], name='X')
  ampa = AMPA_vec(pre=hh, post=hh, conn=bp.connect.All2All(), delay=10., monitors=['s'])
  # ampa = AMPA_mat(pre=hh, post=hh, conn=bp.connect.All2All(), delay=10., monitors=['s'])
  net = bp.Network(hh, ampa)
  net = bp.math.jit(net, show_code=True)
  net.run(100., inputs=('X.input', 10.), report=0.1)

  fig, gs = bp.visualize.get_figure(row_num=2, col_num=1, )
  fig.add_subplot(gs[0, 0])
  bp.visualize.line_plot(hh.mon.ts, hh.mon.V)
  fig.add_subplot(gs[1, 0])
  bp.visualize.line_plot(ampa.mon.ts, ampa.mon.s, show=True)
