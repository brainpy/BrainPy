import jax

import brainpy as bp
import brainpy.math as bm

bm.set_host_device_count(4)


class EINet1(bp.DynSysGroup):
  def __init__(self):
    super().__init__()
    self.N = bp.dyn.LifRefLTC(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                              V_initializer=bp.init.Normal(-55., 2.),
                              sharding=[bm.sharding.NEU_AXIS])
    self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
    self.E = bp.dyn.ProjAlignPostMg1(
      comm=bp.dnn.EventJitFPHomoLinear(3200, 4000, prob=0.02, weight=0.6),
      syn=bp.dyn.Expon.desc(size=4000, tau=5., sharding=[bm.sharding.NEU_AXIS]),
      out=bp.dyn.COBA.desc(E=0.),
      post=self.N
    )
    self.I = bp.dyn.ProjAlignPostMg1(
      comm=bp.dnn.EventJitFPHomoLinear(800, 4000, prob=0.02, weight=6.7),
      syn=bp.dyn.Expon.desc(size=4000, tau=10., sharding=[bm.sharding.NEU_AXIS]),
      out=bp.dyn.COBA.desc(E=-80.),
      post=self.N
    )

  def update(self, input):
    spk = self.delay.at('I')
    self.E(spk[:3200])
    self.I(spk[3200:])
    self.delay(self.N(input))
    return self.N.spike.value


class EINet2(bp.DynSysGroup):
  def __init__(self):
    super().__init__()
    self.N = bp.dyn.LifRefLTC(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                              V_initializer=bp.init.Normal(-55., 2.),
                              sharding=[bm.sharding.NEU_AXIS])
    self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
    self.E = bp.dyn.ProjAlignPostMg1(
      comm=bp.dnn.MaskedLinear(bp.conn.FixedProb(0.02, pre=3200, post=4000), weight=0.6,
                               sharding=[None, bm.sharding.NEU_AXIS]),
      syn=bp.dyn.Expon.desc(size=4000, tau=5., sharding=[bm.sharding.NEU_AXIS]),
      out=bp.dyn.COBA.desc(E=0.),
      post=self.N
    )
    self.I = bp.dyn.ProjAlignPostMg1(
      comm=bp.dnn.MaskedLinear(bp.conn.FixedProb(0.02, pre=800, post=4000), weight=0.6,
                               sharding=[None, bm.sharding.NEU_AXIS]),
      syn=bp.dyn.Expon.desc(size=4000, tau=10., sharding=[bm.sharding.NEU_AXIS]),
      out=bp.dyn.COBA.desc(E=-80.),
      post=self.N
    )

  def update(self, input):
    spk = self.delay.at('I')
    self.E(spk[:3200])
    self.I(spk[3200:])
    self.delay(self.N(input))
    return self.N.spike.value


@bm.jit
def run(indexes):
  return bm.for_loop(lambda i: model.step_run(i, 20.), indexes)


with bm.sharding.device_mesh(jax.devices(), [bm.sharding.NEU_AXIS]):
  # model = EINet1()
  model = EINet2()
  indices = bm.arange(1000)
  spks = run(indices)
bp.visualize.raster_plot(indices, spks, show=True)
