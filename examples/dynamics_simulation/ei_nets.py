import brainpy as bp
import brainpy.math as bm


def model1():
  class EINet(bp.DynSysGroup):
    def __init__(self):
      super().__init__()
      self.N = bp.dyn.LifRefLTC(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                V_initializer=bp.init.Normal(-55., 2.))
      self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
      self.E = bp.dyn.HalfProjAlignPost(comm=bp.dnn.EventJitFPHomoLinear(3200, 4000, prob=0.02, weight=0.6),
                                        syn=bp.dyn.Expon(size=4000, tau=5.),
                                        out=bp.dyn.COBA(E=0.),
                                        post=self.N)
      self.I = bp.dyn.HalfProjAlignPost(comm=bp.dnn.EventJitFPHomoLinear(800, 4000, prob=0.02, weight=6.7),
                                        syn=bp.dyn.Expon(size=4000, tau=10.),
                                        out=bp.dyn.COBA(E=-80.),
                                        post=self.N)

    def update(self, input):
      spk = self.delay.at('I')
      self.E(spk[:3200])
      self.I(spk[3200:])
      self.delay(self.N(input))
      return self.N.spike.value

  model = EINet()
  indices = bm.arange(1000)
  spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
  bp.visualize.raster_plot(indices, spks, show=True)


def model2():
  class EINet(bp.DynSysGroup):
    def __init__(self):
      super().__init__()
      ne, ni = 3200, 800
      self.E = bp.dyn.LifRefLTC(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                V_initializer=bp.init.Normal(-55., 2.))
      self.I = bp.dyn.LifRefLTC(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                V_initializer=bp.init.Normal(-55., 2.))
      self.E2E = bp.dyn.FullProjAlignPost(pre=self.E,
                                          delay=0.1,
                                          comm=bp.dnn.EventJitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                          syn=bp.dyn.Expon(size=ne, tau=5.),
                                          out=bp.dyn.COBA(E=0.),
                                          post=self.E)
      self.E2I = bp.dyn.FullProjAlignPost(pre=self.E,
                                          delay=0.1,
                                          comm=bp.dnn.EventJitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                          syn=bp.dyn.Expon(size=ni, tau=5.),
                                          out=bp.dyn.COBA(E=0.),
                                          post=self.I)
      self.I2E = bp.dyn.FullProjAlignPost(pre=self.I,
                                          delay=0.1,
                                          comm=bp.dnn.EventJitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                          syn=bp.dyn.Expon(size=ne, tau=10.),
                                          out=bp.dyn.COBA(E=-80.),
                                          post=self.E)
      self.I2I = bp.dyn.FullProjAlignPost(pre=self.I,
                                          delay=0.1,
                                          comm=bp.dnn.EventJitFPHomoLinear(ni, ni, prob=0.02, weight=6.7),
                                          syn=bp.dyn.Expon(size=ni, tau=10.),
                                          out=bp.dyn.COBA(E=-80.),
                                          post=self.I)

    def update(self, inp):
      self.E2E()
      self.E2I()
      self.I2E()
      self.I2I()
      self.E(inp)
      self.I(inp)
      return self.E.spike

  model = EINet()
  indices = bm.arange(1000)
  spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
  bp.visualize.raster_plot(indices, spks, show=True)


def model3():
  class EINet(bp.DynSysGroup):
    def __init__(self):
      super().__init__()
      self.N = bp.dyn.LifRefLTC(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                V_initializer=bp.init.Normal(-55., 2.))
      self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
      self.syn1 = bp.dyn.Expon(size=3200, tau=5.)
      self.syn2 = bp.dyn.Expon(size=800, tau=10.)
      self.E = bp.dyn.VanillaProj(comm=bp.dnn.JitFPHomoLinear(3200, 4000, prob=0.02, weight=0.6),
                                  out=bp.dyn.COBA(E=0.),
                                  post=self.N)
      self.I = bp.dyn.VanillaProj(comm=bp.dnn.JitFPHomoLinear(800, 4000, prob=0.02, weight=6.7),
                                  out=bp.dyn.COBA(E=-80.),
                                  post=self.N)

    def update(self, input):
      spk = self.delay.at('I')
      self.E(self.syn1(spk[:3200]))
      self.I(self.syn2(spk[3200:]))
      self.delay(self.N(input))
      return self.N.spike.value

  model = EINet()
  indices = bm.arange(1000)
  spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
  bp.visualize.raster_plot(indices, spks, show=True)


def model4():
  class EINet(bp.DynSysGroup):
    def __init__(self):
      super().__init__()
      ne, ni = 3200, 800
      self.E = bp.dyn.LifRefLTC(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                V_initializer=bp.init.Normal(-55., 2.))
      self.I = bp.dyn.LifRefLTC(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                V_initializer=bp.init.Normal(-55., 2.))
      self.E2E = bp.dyn.FullProjAlignPreSDMg(pre=self.E,
                                             syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                             delay=0.1,
                                             comm=bp.dnn.JitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                             out=bp.dyn.COBA(E=0.),
                                             post=self.E)
      self.E2I = bp.dyn.FullProjAlignPreSDMg(pre=self.E,
                                             syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                             delay=0.1,
                                             comm=bp.dnn.JitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                             out=bp.dyn.COBA(E=0.),
                                             post=self.I)
      self.I2E = bp.dyn.FullProjAlignPreSDMg(pre=self.I,
                                             syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                             delay=0.1,
                                             comm=bp.dnn.JitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                             out=bp.dyn.COBA(E=-80.),
                                             post=self.E)
      self.I2I = bp.dyn.FullProjAlignPreSDMg(pre=self.I,
                                             syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                             delay=0.1,
                                             comm=bp.dnn.JitFPHomoLinear(ni, ni, prob=0.02, weight=6.7),
                                             out=bp.dyn.COBA(E=-80.),
                                             post=self.I)

    def update(self, inp):
      self.E2E()
      self.E2I()
      self.I2E()
      self.I2I()
      self.E(inp)
      self.I(inp)
      return self.E.spike

  model = EINet()
  indices = bm.arange(1000)
  spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
  bp.visualize.raster_plot(indices, spks, show=True)


def model5():
  class EINet(bp.DynSysGroup):
    def __init__(self):
      super().__init__()
      ne, ni = 3200, 800
      self.E = bp.dyn.LifRefLTC(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                V_initializer=bp.init.Normal(-55., 2.))
      self.I = bp.dyn.LifRefLTC(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                V_initializer=bp.init.Normal(-55., 2.))
      self.E2E = bp.dyn.FullProjAlignPreDSMg(pre=self.E,
                                             delay=0.1,
                                             syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                             comm=bp.dnn.JitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                             out=bp.dyn.COBA(E=0.),
                                             post=self.E)
      self.E2I = bp.dyn.FullProjAlignPreDSMg(pre=self.E,
                                             delay=0.1,
                                             syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                             comm=bp.dnn.JitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                             out=bp.dyn.COBA(E=0.),
                                             post=self.I)
      self.I2E = bp.dyn.FullProjAlignPreDSMg(pre=self.I,
                                             delay=0.1,
                                             syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                             comm=bp.dnn.JitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                             out=bp.dyn.COBA(E=-80.),
                                             post=self.E)
      self.I2I = bp.dyn.FullProjAlignPreDSMg(pre=self.I,
                                             delay=0.1,
                                             syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                             comm=bp.dnn.JitFPHomoLinear(ni, ni, prob=0.02, weight=6.7),
                                             out=bp.dyn.COBA(E=-80.),
                                             post=self.I)

    def update(self, inp):
      self.E2E()
      self.E2I()
      self.I2E()
      self.I2I()
      self.E(inp)
      self.I(inp)
      return self.E.spike

  model = EINet()
  indices = bm.arange(1000)
  spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
  bp.visualize.raster_plot(indices, spks, show=True)


def vanalla_proj():
  class EINet(bp.DynSysGroup):
    def __init__(self):
      super().__init__()
      self.N = bp.dyn.LifRefLTC(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                V_initializer=bp.init.Normal(-55., 1.))
      self.delay = bp.VarDelay(self.N.spike, entries={'delay': 2})
      self.syn1 = bp.dyn.Expon(size=3200, tau=5.)
      self.syn2 = bp.dyn.Expon(size=800, tau=10.)
      self.E = bp.dyn.VanillaProj(
        comm=bp.dnn.CSRLinear(bp.conn.FixedProb(0.02, pre=3200, post=4000), weight=0.6),
        out=bp.dyn.COBA(E=0.),
        post=self.N
      )
      self.I = bp.dyn.VanillaProj(
        comm=bp.dnn.CSRLinear(bp.conn.FixedProb(0.02, pre=800, post=4000), weight=6.7),
        out=bp.dyn.COBA(E=-80.),
        post=self.N
      )

    def update(self, input):
      spk = self.delay.at('I')
      self.E(self.syn1(spk[:3200]))
      self.I(self.syn2(spk[3200:]))
      self.delay(self.N(input))
      return self.N.spike.value

  model = EINet()
  indices = bm.arange(10000)
  spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices, progress_bar=True)
  bp.visualize.raster_plot(indices, spks, show=True)


if __name__ == '__main__':
  # model1()
  # model2()
  # model3()
  # model4()
  # model5()
  vanalla_proj()
