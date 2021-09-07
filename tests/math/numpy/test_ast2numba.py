# -*- coding: utf-8 -*-

import ast
import re
from pprint import pprint

import brainpy as bp
from brainpy.math.numpy.ast2numba import FuncTransformer
from brainpy.math.numpy.ast2numba import _jit_cls_func
from brainpy.tools import ast2code

bp.math.use_backend('numpy')


def test_find_self_data1():
  code = '''
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
  '''

  arg = 'self'
  print(re.findall('\\b' + arg + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b', code))


def test_transformer():
  code = '''
def update(self, _t, _dt):
  V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t, self.input)
  self.spike[:] = (self.V < self.V_th) * (V >= self.V_th)
  self.V[:] = V
  self.m[:] = m
  self.h[:] = h
  self.n[:] = n
  self.input[:] = 0.
  '''
  tree = ast.parse(code)

  transformer = FuncTransformer(func_name='self.integral', arg_to_append={'A_gNa': 'A.gNa'})
  new_tree = transformer.visit(tree)

  new_code = ast2code(new_tree)
  print(new_code)


def test_cls_func_hh1():
  class HH(bp.NeuGroup):
    target_backend = 'general'

    def __init__(self, size, ENa=50., EK=-77., EL=-54.387,
                 C=1.0, gNa=120., gK=36., gL=0.03, V_th=20.,
                 **kwargs):
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
      self.V = bp.math.Variable(bp.math.ones(size) * -65.)
      self.m = bp.math.Variable(bp.math.ones(size) * 0.5)
      self.h = bp.math.Variable(bp.math.ones(size) * 0.6)
      self.n = bp.math.Variable(bp.math.ones(size) * 0.32)
      self.spike = bp.math.Variable(bp.math.zeros(size, dtype=bool))
      self.input = bp.math.Variable(bp.math.zeros(size))

      super(HH, self).__init__(size=size, **kwargs)

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
      self.spike[:] = (self.V < self.V_th) * (V >= self.V_th)
      self.V[:] = V
      self.m[:] = m
      self.h[:] = h
      self.n[:] = n
      self.input[:] = 0.

  hh = HH(10)

  r = _jit_cls_func(hh.update, show_code=True)
  pprint(r['func'])
  pprint('arguments:')
  pprint(r['arguments'])
  pprint('arg2call:')
  pprint(r['arg2call'])
  pprint('nodes:')
  pprint(r['nodes'])


def test_cls_func_hh2():
  class HH(bp.NeuGroup):
    def __init__(self, size, ENa=50., EK=-77., EL=-54.387,
                 C=1.0, gNa=120., gK=36., gL=0.03, V_th=20.,
                 **kwargs):
      # parameters
      self.ENa = ENa
      self.EK = EK
      self.EL = EL
      self.C = C
      self.gNa = bp.math.Variable(gNa)
      self.gK = gK
      self.gL = gL
      self.V_th = bp.math.Variable(V_th)

      # variables
      self.V = bp.math.Variable(bp.math.ones(size) * -65.)
      self.m = bp.math.Variable(bp.math.ones(size) * 0.5)
      self.h = bp.math.Variable(bp.math.ones(size) * 0.6)
      self.n = bp.math.Variable(bp.math.ones(size) * 0.32)
      self.spike = bp.math.Variable(bp.math.zeros(size, dtype=bool))
      self.input = bp.math.Variable(bp.math.zeros(size))

      super(HH, self).__init__(size=size, **kwargs)

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
      self.spike[:] = (self.V < self.V_th) * (V >= self.V_th)
      self.V[:] = V
      self.m[:] = m
      self.h[:] = h
      self.n[:] = n
      self.input[:] = 0.

  hh = HH(10)

  r = _jit_cls_func(hh.update, show_code=True)
  pprint(r['func'])
  pprint('arguments:')
  pprint(r['arguments'])
  pprint('arg2call:')
  pprint(r['arg2call'])
  pprint('nodes:')
  pprint(r['nodes'])


def test_cls_func_ampa1():
  class HH(bp.NeuGroup):
    def __init__(self, size, ENa=50., EK=-77., EL=-54.387,
                 C=1.0, gNa=120., gK=36., gL=0.03, V_th=20.,
                 **kwargs):
      # parameters
      self.ENa = ENa
      self.EK = EK
      self.EL = EL
      self.C = C
      self.gNa = bp.math.Variable(gNa)
      self.gK = gK
      self.gL = gL
      self.V_th = bp.math.Variable(V_th)

      # variables
      self.V = bp.math.Variable(bp.math.ones(size) * -65.)
      self.m = bp.math.Variable(bp.math.ones(size) * 0.5)
      self.h = bp.math.Variable(bp.math.ones(size) * 0.6)
      self.n = bp.math.Variable(bp.math.ones(size) * 0.32)
      self.spike = bp.math.Variable(bp.math.zeros(size, dtype=bool))
      self.input = bp.math.Variable(bp.math.zeros(size))

      super(HH, self).__init__(size=size, **kwargs)

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
      self.spike[:] = (self.V < self.V_th) * (V >= self.V_th)
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

    @bp.odeint
    def int_s(self, s, t):
      return - s / self.tau

    def update(self, _t, _i):
      for i in range(self.size):
        pre_id = self.pre_ids[i]
        self.s[i] = self.int_s(self.s[i], _t)
        self.s[i] += self.pre.spike[pre_id]
        self.g.push(i, self.g_max * self.s[i])
        post_id = self.post_ids[i]
        self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)

  hh = HH(10)
  ampa = AMPA_vec(pre=hh, post=hh, conn=bp.connect.All2All(), delay=10.)

  r = _jit_cls_func(ampa.update, show_code=True)
  pprint(r['func'])
  pprint('arguments:')
  pprint(r['arguments'])
  pprint('arg2call:')
  pprint(r['arg2call'])
  pprint('nodes:')
  pprint(r['nodes'])



def test_hh1():
  class HH(bp.NeuGroup):
    def __init__(self, size, ENa=50., EK=-77., EL=-54.387,
                 C=1.0, gNa=120., gK=36., gL=0.03, V_th=20.,
                 **kwargs):
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
      self.V = bp.math.Variable(bp.math.ones(size) * -65.)
      self.m = bp.math.Variable(bp.math.ones(size) * 0.5)
      self.h = bp.math.Variable(bp.math.ones(size) * 0.6)
      self.n = bp.math.Variable(bp.math.ones(size) * 0.32)
      self.spike = bp.math.Variable(bp.math.zeros(size, dtype=bool))
      self.input = bp.math.Variable(bp.math.zeros(size))

      super(HH, self).__init__(size=size, **kwargs)

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
      self.spike[:] = (self.V < self.V_th) * (V >= self.V_th)
      self.V[:] = V
      self.m[:] = m
      self.h[:] = h
      self.n[:] = n
      self.input[:] = 0.

  hh = HH(10)
  bp.math.jit(hh, show_code=True)


def test_hh2():
  class HH(bp.NeuGroup):
    def __init__(self, size, ENa=50., EK=-77., EL=-54.387,
                 C=1.0, gNa=120., gK=36., gL=0.03, V_th=20.,
                 **kwargs):
      # parameters
      self.ENa = ENa
      self.EK = EK
      self.EL = EL
      self.C = C
      self.gNa = bp.math.Variable(gNa)
      self.gK = gK
      self.gL = gL
      self.V_th = bp.math.Variable(V_th)

      # variables
      self.V = bp.math.Variable(bp.math.ones(size) * -65.)
      self.m = bp.math.Variable(bp.math.ones(size) * 0.5)
      self.h = bp.math.Variable(bp.math.ones(size) * 0.6)
      self.n = bp.math.Variable(bp.math.ones(size) * 0.32)
      self.spike = bp.math.Variable(bp.math.zeros(size, dtype=bool))
      self.input = bp.math.Variable(bp.math.zeros(size))

      super(HH, self).__init__(size=size, **kwargs)

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
      self.spike[:] = (self.V < self.V_th) * (V >= self.V_th)
      self.V[:] = V
      self.m[:] = m
      self.h[:] = h
      self.n[:] = n
      self.input[:] = 0.

  hh = HH(10)
  bp.math.jit(hh, show_code=True)


def test_hh_ampa_net1():
  class HH(bp.NeuGroup):
    def __init__(self, size, ENa=50., EK=-77., EL=-54.387,
                 C=1.0, gNa=120., gK=36., gL=0.03, V_th=20.,
                 **kwargs):
      # parameters
      self.ENa = ENa
      self.EK = EK
      self.EL = EL
      self.C = C
      self.gNa = bp.math.Variable(gNa)
      self.gK = gK
      self.gL = gL
      self.V_th = bp.math.Variable(V_th)

      # variables
      self.V = bp.math.Variable(bp.math.ones(size) * -65.)
      self.m = bp.math.Variable(bp.math.ones(size) * 0.5)
      self.h = bp.math.Variable(bp.math.ones(size) * 0.6)
      self.n = bp.math.Variable(bp.math.ones(size) * 0.32)
      self.spike = bp.math.Variable(bp.math.zeros(size, dtype=bool))
      self.input = bp.math.Variable(bp.math.zeros(size))

      super(HH, self).__init__(size=size, **kwargs)

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
      self.spike[:] = (self.V < self.V_th) * (V >= self.V_th)
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

    @bp.odeint
    def int_s(self, s, t):
      return - s / self.tau

    def update(self, _t, _i):
      for i in range(self.size):
        pre_id = self.pre_ids[i]
        self.s[i] = self.int_s(self.s[i], _t)
        self.s[i] += self.pre.spike[pre_id]
        self.g.push(i, self.g_max * self.s[i])
        post_id = self.post_ids[i]
        self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)

  hh = HH(10)
  ampa = AMPA_vec(pre=hh, post=hh, conn=bp.connect.All2All(), delay=10.)
  net = bp.Network(hh, ampa)

  bp.math.jit(net, show_code=True)

