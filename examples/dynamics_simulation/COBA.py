import brainpy as bp
import brainpy.math as bm

neu_pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                V_initializer=bp.init.Normal(-55., 2.))


class EICOBA_PreAlign(bp.DynamicalSystem):
  def __init__(self, num_exc, num_inh, inp=20.):
    super().__init__()

    self.inp = inp
    self.E = bp.dyn.LifRefLTC(num_exc, **neu_pars)
    self.I = bp.dyn.LifRefLTC(num_inh, **neu_pars)

    self.E2I = bp.dyn.FullProjAlignPreSDMg(
      pre=self.E,
      syn=bp.dyn.Expon.desc(self.E.varshape, tau=5.),
      delay=None,
      comm=bp.dnn.CSRLinear(bp.conn.FixedProb(0.02, pre=self.E.num, post=self.I.num), 0.6),
      out=bp.dyn.COBA(E=0.),
      post=self.I,
    )
    self.E2E = bp.dyn.FullProjAlignPreSDMg(
      pre=self.E,
      syn=bp.dyn.Expon.desc(self.E.varshape, tau=5.),
      delay=None,
      comm=bp.dnn.CSRLinear(bp.conn.FixedProb(0.02, pre=self.E.num, post=self.E.num), 0.6),
      out=bp.dyn.COBA(E=0.),
      post=self.E,
    )
    self.I2E = bp.dyn.FullProjAlignPreSDMg(
      pre=self.I,
      syn=bp.dyn.Expon.desc(self.I.varshape, tau=10.),
      delay=None,
      comm=bp.dnn.CSRLinear(bp.conn.FixedProb(0.02, pre=self.I.num, post=self.E.num), 6.7),
      out=bp.dyn.COBA(E=-80.),
      post=self.E,
    )
    self.I2I = bp.dyn.FullProjAlignPreSDMg(
      pre=self.I,
      syn=bp.dyn.Expon.desc(self.I.varshape, tau=10.),
      delay=0.,
      comm=bp.dnn.CSRLinear(bp.conn.FixedProb(0.02, pre=self.I.num, post=self.I.num), 6.7),
      out=bp.dyn.COBA(E=-80.),
      post=self.I,
    )

  def update(self):
    self.E2I()
    self.I2I()
    self.I2E()
    self.E2E()
    self.E(self.inp)
    self.I(self.inp)


class EICOBA_PostAlign(bp.DynamicalSystem):
  def __init__(self, num_exc, num_inh, inp=20., ltc=True):
    super().__init__()
    self.inp = inp

    if ltc:
      self.E = bp.dyn.LifRefLTC(num_exc, **neu_pars)
      self.I = bp.dyn.LifRefLTC(num_inh, **neu_pars)
    else:
      self.E = bp.dyn.LifRef(num_exc, **neu_pars)
      self.I = bp.dyn.LifRef(num_inh, **neu_pars)

    self.E2E = bp.dyn.FullProjAlignPostMg(
      pre=self.E,
      delay=None,
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(0.02, pre=self.E.num, post=self.E.num), 0.6),
      syn=bp.dyn.Expon.desc(self.E.varshape, tau=5.),
      out=bp.dyn.COBA.desc(E=0.),
      post=self.E,
    )
    self.E2I = bp.dyn.FullProjAlignPostMg(
      pre=self.E,
      delay=None,
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(0.02, pre=self.E.num, post=self.I.num), 0.6),
      syn=bp.dyn.Expon.desc(self.I.varshape, tau=5.),
      out=bp.dyn.COBA.desc(E=0.),
      post=self.I,
    )
    self.I2E = bp.dyn.FullProjAlignPostMg(
      pre=self.I,
      delay=None,
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(0.02, pre=self.I.num, post=self.E.num), 6.7),
      syn=bp.dyn.Expon.desc(self.E.varshape, tau=10.),
      out=bp.dyn.COBA.desc(E=-80.),
      post=self.E,
    )
    self.I2I = bp.dyn.FullProjAlignPostMg(
      pre=self.I,
      delay=None,
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(0.02, pre=self.I.num, post=self.I.num), 6.7),
      syn=bp.dyn.Expon.desc(self.I.varshape, tau=10.),
      out=bp.dyn.COBA.desc(E=-80.),
      post=self.I,
    )

  def update(self):
    self.E2I()
    self.I2I()
    self.I2E()
    self.E2E()
    self.E(self.inp)
    self.I(self.inp)


class EINet(bp.Network):
  def __init__(self, scale=1.0, method='exp_auto'):
    # network size
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)

    # neurons
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                V_initializer=bp.init.Normal(-55., 2.))
    E = bp.neurons.LIF(num_exc, **pars, method=method)
    I = bp.neurons.LIF(num_inh, **pars, method=method)

    # synapses
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight
    E2E = bp.synapses.Exponential(E, E, bp.conn.FixedProb(prob=0.02),
                                  g_max=we, tau=5., method=method,
                                  output=bp.synouts.COBA(E=0.))
    E2I = bp.synapses.Exponential(E, I, bp.conn.FixedProb(prob=0.02),
                                  g_max=we, tau=5., method=method,
                                  output=bp.synouts.COBA(E=0.))
    I2E = bp.synapses.Exponential(I, E, bp.conn.FixedProb(prob=0.02),
                                  g_max=wi, tau=10., method=method,
                                  output=bp.synouts.COBA(E=-80.))
    I2I = bp.synapses.Exponential(I, I, bp.conn.FixedProb(prob=0.02),
                                  g_max=wi, tau=10., method=method,
                                  output=bp.synouts.COBA(E=-80.))

    super(EINet, self).__init__(E2E, E2I, I2E, I2I, E=E, I=I)


# num_device = 8
# bm.set_host_device_count(num_device)
# bm.sharding.set(mesh_axes=(bp.dyn.PNEU_AXIS,), mesh_shape=(num_device, ))


def run1():
  with bm.environment(mode=bm.BatchingMode(10)):
    net = EICOBA_PostAlign(3200, 800)
    runner = bp.DSRunner(net, monitors={'E.spike': net.E.spike})
    runner.run(100.)
    bp.visualize.raster_plot(runner.mon['ts'], runner.mon['E.spike'][0], show=True)
    print(runner.run(100., eval_time=True))
    print(runner.mon['E.spike'].shape)
    print(runner.mon['ts'].shape)


def run2():
  net = EINet()
  runner = bp.DSRunner(net,
                       monitors=['E.spike'],
                       inputs=[('E.input', 20.), ('I.input', 20.)])
  r = runner.run(100., eval_time=True)
  print(r)
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)


def run3():
  net = EICOBA_PreAlign(3200, 800)
  runner = bp.DSRunner(net, monitors={'E.spike': net.E.spike})
  print(runner.run(100., eval_time=True))
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)


def run4():
  bm.set(dt=0.5, x64=True)
  net = EICOBA_PostAlign(3200, 800, ltc=True)
  runner = bp.DSRunner(net, monitors={'E.spike': net.E.spike})
  print(runner.run(100., eval_time=True))
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)






if __name__ == '__main__':
  # run1()
  # run2()
  # run3()
  run4()
