import jax

import brainpy as bp
import brainpy.math as bm
from jax.experimental.maps import xmap


# bm.set_host_device_count(4)


class ExpJIT(bp.Projection):
  def __init__(self, pre_num, post, prob, g_max, tau=5., E=0.):
    super().__init__()
    self.proj = bp.dyn.HalfProjAlignPostMg(
      comm=bp.dnn.EventJitFPHomoLinear(pre_num, post.num, prob=prob, weight=g_max),
      syn=bp.dyn.Expon.desc(size=post.num, tau=tau, sharding=[bm.sharding.NEU_AXIS]),
      out=bp.dyn.COBA.desc(E=E),
      post=post
    )


class EINet1(bp.DynSysGroup):
  def __init__(self):
    super().__init__()
    self.N = bp.dyn.LifRefLTC(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                              V_initializer=bp.init.Normal(-55., 2.),
                              sharding=[bm.sharding.NEU_AXIS])
    self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
    self.E = ExpJIT(3200, self.N, 0.02, 0.6)
    self.I = ExpJIT(800, self.N, 0.02, 6.7, E=-80., tau=10.)

  def update(self, input):
    spk = self.delay.at('I')
    self.E(spk[:3200])
    self.I(spk[3200:])
    self.delay(self.N(input))
    return self.N.spike.value


class ExpMasked(bp.Projection):
  def __init__(self, pre_num, post, prob, g_max, tau=5., E=0.):
    super().__init__()
    self.proj = bp.dyn.HalfProjAlignPostMg(
      comm=bp.dnn.MaskedLinear(bp.conn.FixedProb(prob, pre=pre_num, post=post.num), weight=g_max,
                               sharding=[None, bm.sharding.NEU_AXIS]),
      syn=bp.dyn.Expon.desc(size=post.num, tau=tau, sharding=[bm.sharding.NEU_AXIS]),
      out=bp.dyn.COBA.desc(E=E),
      post=post
    )


class EINet2(bp.DynSysGroup):
  def __init__(self):
    super().__init__()
    self.N = bp.dyn.LifRefLTC(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                              V_initializer=bp.init.Normal(-55., 2.),
                              sharding=[bm.sharding.NEU_AXIS])
    self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
    self.E = ExpMasked(3200, self.N, 0.02, 0.6)
    self.I = ExpMasked(800, self.N, 0.02, 6.7, E=-80., tau=10.)

  def update(self, input):
    spk = self.delay.at('I')
    self.E(spk[:3200])
    self.I(spk[3200:])
    self.delay(self.N(input))
    return self.N.spike.value


class PCSR(bp.dnn.Layer):
  def __init__(self, conn, weight, num_shard, transpose=True):
    super().__init__()

    self.conn = conn
    self.transpose = transpose
    self.num_shard = num_shard

    # connection
    self.indices = []
    self.inptr = []
    for _ in range(num_shard):
      indices, inptr = self.conn.require('csr')
      self.indices.append(indices)
      self.inptr.append(inptr)
    self.indices = bm.asarray(self.indices)
    self.inptr = bm.asarray(self.inptr)

    # weight
    weight = bp.init.parameter(weight, (self.indices.size,))
    if isinstance(self.mode, bm.TrainingMode):
      weight = bm.TrainVar(weight)
    self.weight = weight

  def update(self, v):
    # ax1 = None if bm.size(self.weight) > 1 else (None, bm.sharding.NEU_AXIS)
    mapped = xmap(
      self._f,
      in_axes=((bm.sharding.NEU_AXIS, None), (bm.sharding.NEU_AXIS, None), (None, )),
      out_axes=(bm.sharding.NEU_AXIS, None),
      # axis_resources={bm.sharding.NEU_AXIS: bm.sharding.NEU_AXIS},
    )
    r = mapped(self.indices, self.inptr, v)
    return r.flatten()

  def _f(self, indices, indptr, x):
    return bm.event.csrmv(self.weight, indices, indptr, x,
                          shape=(self.conn.pre_num, self.conn.post_num // self.num_shard),
                          transpose=self.transpose)


class ExpMasked2(bp.Projection):
  def __init__(self, pre_num, post, prob, g_max, tau=5., E=0.):
    super().__init__()
    self.proj = bp.dyn.HalfProjAlignPostMg(
      comm=PCSR(bp.conn.FixedProb(prob, pre=pre_num, post=post.num), weight=g_max, num_shard=4),
      syn=bp.dyn.Expon.desc(size=post.num, tau=tau, sharding=[bm.sharding.NEU_AXIS]),
      out=bp.dyn.COBA.desc(E=E),
      post=post
    )


class EINet3(bp.DynSysGroup):
  def __init__(self):
    super().__init__()
    self.N = bp.dyn.LifRefLTC(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                              V_initializer=bp.init.Normal(-55., 2.),
                              sharding=[bm.sharding.NEU_AXIS])
    self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
    self.E = ExpMasked2(3200, self.N, 0.02, 0.6)
    self.I = ExpMasked2(800, self.N, 0.02, 6.7, E=-80., tau=10.)

  def update(self, input):
    spk = self.delay.at('I')
    self.E(spk[:3200])
    self.I(spk[3200:])
    self.delay(self.N(input))
    return self.N.spike.value


def try_ei_net1():
  @bm.jit
  def run(indexes):
    return bm.for_loop(lambda i: model.step_run(i, 20.), indexes)

  with bm.sharding.device_mesh(jax.devices(), [bm.sharding.NEU_AXIS]):
    model = EINet1()
    indices = bm.arange(1000)
    spks = run(indices)
  bp.visualize.raster_plot(indices, spks, show=True)


def try_ei_net2():
  @bm.jit
  def run(indexes):
    return bm.for_loop(lambda i: model.step_run(i, 20.), indexes)

  with bm.sharding.device_mesh(jax.devices(), [bm.sharding.NEU_AXIS]):
    model = EINet2()
    indices = bm.arange(1000)
    spks = run(indices)
  bp.visualize.raster_plot(indices, spks, show=True)



def try_ei_net3():
  @bm.jit
  def run(indexes):
    return bm.for_loop(lambda i: model.step_run(i, 20.), indexes)

  with bm.sharding.device_mesh(jax.devices(), [bm.sharding.NEU_AXIS]):
    model = EINet3()
    indices = bm.arange(1000)
    spks = run(indices)
  bp.visualize.raster_plot(indices, spks, show=True)


if __name__ == '__main__':
  # try_ei_net1()
  # try_ei_net2()
  try_ei_net3()
