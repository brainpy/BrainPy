import brainpy as bp
import brainpy.math as bm
from jax import pmap

bm.set_host_device_count(20)


class EINet(bp.DynamicalSystemNS):
  def __init__(self, scale=1.0, e_input=20., i_input=20., delay=None):
    super().__init__()

    self.bg_exc = e_input
    self.bg_inh = i_input

    # network size
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)

    # neurons
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                V_initializer=bp.init.Normal(-55., 2.), input_var=False)
    self.E = bp.neurons.LIF(num_exc, **pars)
    self.I = bp.neurons.LIF(num_inh, **pars)

    # synapses
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight
    self.E2E = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.E.size, post=self.E.size),
      g_max=we, tau=5., out=bp.experimental.COBA(E=0.), comp_method='dense'
    )
    self.E2I = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.E.size, post=self.I.size, ),
      g_max=we, tau=5., out=bp.experimental.COBA(E=0.), comp_method='dense'
    )
    self.I2E = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.I.size, post=self.E.size),
      g_max=wi, tau=10., out=bp.experimental.COBA(E=-80.), comp_method='dense'
    )
    self.I2I = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.I.size, post=self.I.size),
      g_max=wi, tau=10., out=bp.experimental.COBA(E=-80.), comp_method='dense'
    )
    self.delayE = bp.Delay(self.E.spike, entries={'E': delay})
    self.delayI = bp.Delay(self.I.spike, entries={'I': delay})

  def update(self):
    e_spike = self.delayE.at('E')
    i_spike = self.delayI.at('I')
    e_inp = self.E2E(e_spike, self.E.V) + self.I2E(i_spike, self.E.V) + self.bg_exc
    i_inp = self.I2I(i_spike, self.I.V) + self.E2I(e_spike, self.I.V) + self.bg_inh
    self.delayE(self.E(e_inp))
    self.delayI(self.I(i_inp))


class EINetv2(bp.DynamicalSystemNS):
  def __init__(self, scale=1.0, e_input=20., i_input=20., delay=None):
    super().__init__()

    self.bg_exc = e_input
    self.bg_inh = i_input

    # network size
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)

    # neurons
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                V_initializer=bp.init.Normal(-55., 2.), input_var=False)
    self.E = bp.neurons.LIF(num_exc, **pars)
    self.I = bp.neurons.LIF(num_inh, **pars)

    # synapses
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight
    self.E2E = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.E.size, post=self.E.size),
      g_max=we, tau=5., out=bp.experimental.COBA(E=0.)
    )
    self.E2I = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.E.size, post=self.I.size, ),
      g_max=we, tau=5., out=bp.experimental.COBA(E=0.)
    )
    self.I2E = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.I.size, post=self.E.size),
      g_max=wi, tau=10., out=bp.experimental.COBA(E=-80.)
    )
    self.I2I = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.I.size, post=self.I.size),
      g_max=wi, tau=10., out=bp.experimental.COBA(E=-80.)
    )
    bp.share.save('E-spike', bp.Delay(self.E.spike, entries={'E': delay}))
    bp.share.save('I-spike', bp.Delay(self.I.spike, entries={'I': delay}))

  def update(self):
    e_spike = bp.share.load('E-spike').at('E')
    i_spike = bp.share.load('I-spike').at('I')
    e_inp = self.E2E(e_spike, self.E.V) + self.I2E(i_spike, self.E.V) + self.bg_exc
    i_inp = self.I2I(i_spike, self.I.V) + self.E2I(e_spike, self.I.V) + self.bg_inh
    self.E(e_inp)
    self.I(i_inp)


# simulation
net = EINet(delay=0., scale=1.)
runner = bp.DSRunner(net, monitors={'E.spike': net.E.spike})
runner.run(100.)
# print(r)
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)

# @pmap
# def f2(I):
#   net = EINet(delay=0., scale=5., e_input=I, i_input=I)
#   # net = EINetv2(delay=0., scale=2.)
#   runner = bp.DSRunner(net, monitors={'E.spike': net.E.spike}, numpy_mon_after_run=False)
#   runner.run(10000.)
#   return runner.mon
#   # print(r)
#   # bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)
#
#
# print(f2(bm.ones(20) * 20.))







