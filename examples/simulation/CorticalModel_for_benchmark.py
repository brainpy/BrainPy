# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm
import numpy as np

bp.math.set_platform('cpu')


class LIF(bp.dynsim.NeuGroup):
  def __init__(self, size, tau_neu=10., tau_syn=0.5, tau_ref=2.,
               V_reset=-65., V_th=-50., Cm=0.25, ):
    super(LIF, self).__init__(size=size)

    # parameters
    self.tau_neu = tau_neu  # membrane time constant [ms]
    self.tau_syn = tau_syn  # Post-synaptic current time constant [ms]
    self.tau_ref = tau_ref  # absolute refractory period [ms]
    self.Cm = Cm  # membrane capacity [nF]
    self.V_reset = V_reset  # reset potential [mV]
    self.V_th = V_th  # fixed firing threshold [mV]
    self.Iext = 0.  # constant external current [nA]

    # variables
    self.V = bm.Variable(-65. + 5.0 * bm.random.randn(self.num))  # [mV]
    self.I = bm.Variable(bm.zeros(self.num))  # synaptic currents [nA]
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

    # function
    self.integral = bp.odeint(bp.JointEq([self.dV, self.dI]), method='exp_auto')

  def dV(self, V, t, I):
    return (-V + self.V_reset) / self.tau_neu + (I + self.Iext) / self.Cm

  def dI(self, I, t):
    return -I / self.tau_syn

  def update(self, _t, _dt):
    ref = (_t - self.t_last_spike) <= self.tau_ref
    V, I = self.integral(self.V, self.I, _t, _dt)
    V = bm.where(ref, self.V, V)
    spike = (V >= self.V_th)
    self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
    self.V.value = bm.where(spike, self.V_reset, V)
    self.spike.value = spike
    self.I.value = I


class ExpSyn(bp.dynsim.TwoEndConn):
  # Synapses parameters
  exc_delay = (1.5, 0.75)  # Excitatory/Std. delay [ms]
  inh_delay = (0.80, 0.4)  # Inhibitory/Std. delay [ms]
  exc_weight = (0.0878, 0.0088)  # excitatory/Std. synaptic weight [nA]
  inh_weight_scale = -4.  # Relative inhibitory synaptic strength

  def __init__(self, pre, post, prob, syn_type='e', conn_type=0):
    super(ExpSyn, self).__init__(pre=pre, post=post, conn=None)
    self.check_pre_attrs('spike')
    self.check_post_attrs('I')
    assert syn_type in ['e', 'i']
    # assert conn_type in [0, 1, 2, 3]
    assert 0. < prob < 1.

    # parameters
    self.syn_type = syn_type
    self.conn_type = conn_type

    # connection
    if conn_type == 0:
      # number of synapses calculated with equation 3 from the article
      num = int(np.log(1.0 - prob) / np.log(1.0 - (1.0 / float(pre.num * post.num))))
      self.pre2post = bp.conn.ij2csr(pre_ids=np.random.randint(0, pre.num, num),
                                     post_ids=np.random.randint(0, post.num, num),
                                     num_pre=pre.num)
      self.num = self.pre2post[0].size
    elif conn_type == 1:
      # number of synapses calculated with equation 5 from the article
      self.pre2post = bp.conn.FixedProb(prob)(pre.size, post.size).require('pre2post')
      self.num = self.pre2post[0].size
    elif conn_type == 2:
      self.num = int(prob * pre.num * post.num)
      self.pre_ids = bm.random.randint(0, pre.num, size=self.num, dtype=bm.uint32)
      self.post_ids = bm.random.randint(0, post.num, size=self.num, dtype=bm.uint32)
    elif conn_type in [3, 4]:
      self.pre2post = bp.conn.FixedProb(prob)(pre.size, post.size).require('pre2post')
      self.num = self.pre2post[0].size
      self.max_post_conn = bm.diff(self.pre2post[1]).max()
    else:
      raise ValueError

    # delay
    if syn_type == 'e':
      self.delay = bm.random.normal(*self.exc_delay, size=pre.num)
    elif syn_type == 'i':
      self.delay = bm.random.normal(*self.inh_delay, size=pre.num)
    else:
      raise ValueError
    self.delay = bm.where(self.delay < bm.get_dt(), bm.get_dt(), self.delay)

    # weights
    self.weights = bm.random.normal(*self.exc_weight, size=self.num)
    self.weights = bm.where(self.weights < 0, 0., self.weights)
    if syn_type == 'i':
      self.weights *= self.inh_weight_scale

    # variables
    self.pre_sps = bp.ConstantDelay(pre.num, self.delay, bool)

  def update(self, _t, _dt):
    self.pre_sps.push(self.pre.spike)
    delayed_sps = self.pre_sps.pull()
    if self.conn_type in [0, 1]:
      post_vs = bm.pre2post_event_sum(delayed_sps, self.pre2post, self.post.num, self.weights)
    elif self.conn_type == 2:
      post_vs = bm.pre2post_event_sum2(delayed_sps, self.pre_ids, self.post_ids, self.post.num, self.weights)
      # post_vs = bm.zeros(self.post.num)
      # post_vs = post_vs.value.at[self.post_ids.value].add(delayed_sps[self.pre_ids.value])
    elif self.conn_type == 3:
      post_vs = bm.pre2post_event_sum3(delayed_sps, self.pre2post, self.post.num, self.weights,
                                       self.max_post_conn)
    elif self.conn_type == 4:
      post_vs = bm.pre2post_event_sum4(delayed_sps, self.pre2post, self.post.num, self.weights,
                                       self.max_post_conn)
    else:
      raise ValueError
    self.post.I += post_vs


# class PoissonInput(bp.NeuGroup):
#   def __init__(self, post, freq=8.):
#     base = 20
#     super(PoissonInput, self).__init__(size=(post.num, base))
#
#     # parameters
#     freq = post.num * freq / base
#     self.prob = freq * bm.get_dt() / 1000.
#     self.weight = ExpSyn.exc_weight[0]
#     self.post = post
#     assert hasattr(post, 'I')
#
#     # variables
#     self.rng = bm.random.RandomState()
#
#   def update(self, _t, _dt):
#     self.post.I += self.weight * self.rng.random(self.size).sum(axis=1)


class PoissonInput(bp.dynsim.NeuGroup):
  def __init__(self, post, freq=8.):
    super(PoissonInput, self).__init__(size=(post.num,))

    # parameters
    self.prob = freq * bm.get_dt() / 1000.
    self.loc = post.num * self.prob
    self.scale = np.sqrt(post.num * self.prob * (1 - self.prob))
    self.weight = ExpSyn.exc_weight[0]
    self.post = post
    assert hasattr(post, 'I')

    # variables
    self.rng = bm.random.RandomState()

  def update(self, _t, _dt):
    self.post.I += self.weight * self.rng.normal(self.loc, self.scale, self.num)


class PoissonInput2(bp.dynsim.NeuGroup):
  def __init__(self, pops, freq=8.):
    super(PoissonInput2, self).__init__(size=sum([p.num for p in pops]))

    # parameters
    self.pops = pops
    prob = freq * bm.get_dt() / 1000.
    assert (prob * self.num > 5.) and (self.num * (1 - prob) > 5)
    self.loc = self.num * prob
    self.scale = np.sqrt(self.num * prob * (1 - prob))
    self.weight = ExpSyn.exc_weight[0]

    # variables
    self.rng = bm.random.RandomState()

  def update(self, _t, _dt):
    sample_weights = self.rng.normal(self.loc, self.scale, self.num) * self.weight
    size = 0
    for p in self.pops:
      p.I += sample_weights[size: size + p.num]
      size += p.num


class ThalamusInput(bp.dynsim.TwoEndConn):
  def __init__(self, pre, post, conn_prob=0.1):
    super(ThalamusInput, self).__init__(pre=pre, post=post, conn=bp.conn.FixedProb(conn_prob))
    self.check_pre_attrs('spike')
    self.check_post_attrs('I')

    # connection and weights
    self.pre2post = self.conn.require('pre2post')
    self.syn_num = self.pre2post[0].size
    self.weights = bm.random.normal(*ExpSyn.exc_weight, size=self.syn_num)
    self.weights = bm.where(self.weights < 0., 0., self.weights)

    # variables
    self.turn_on = bm.Variable(bm.asarray([False]))

  def update(self, _t, _dt):
    def true_fn(x):
      post_vs = bm.pre2post_event_sum(self.pre.spike, self.pre2post, self.post.num, self.weights)
      self.post.I += post_vs

    bm.make_cond(true_fn, lambda _: None, dyn_vars=(self.post.I, self.pre.spike))(self.turn_on[0])


class CorticalMicrocircuit(bp.dynsim.Network):
  # Names for each layer:
  layer_name = ['L23e', 'L23i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i', 'Th']

  # Population size per layer:
  #            2/3e   2/3i   4e    4i    5e    5i    6e     6i    Th
  layer_num = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948, 902]

  # Layer-specific background input [nA]:
  #                             2/3e  2/3i  4e    4i    5e    5i    6e    6i
  layer_specific_bg = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100]) / 1000

  # Layer-independent background input [nA]:
  #                                2/3e  2/3i  4e    4i    5e    5i    6e    6i
  layer_independent_bg = np.array([2000, 1850, 2000, 1850, 2000, 1850, 2000, 1850]) / 1000

  # Prob. connection table
  conn_table = np.array([[0.101, 0.169, 0.044, 0.082, 0.032, 0.0000, 0.008, 0.000, 0.0000],
                         [0.135, 0.137, 0.032, 0.052, 0.075, 0.0000, 0.004, 0.000, 0.0000],
                         [0.008, 0.006, 0.050, 0.135, 0.007, 0.0003, 0.045, 0.000, 0.0983],
                         [0.069, 0.003, 0.079, 0.160, 0.003, 0.0000, 0.106, 0.000, 0.0619],
                         [0.100, 0.062, 0.051, 0.006, 0.083, 0.3730, 0.020, 0.000, 0.0000],
                         [0.055, 0.027, 0.026, 0.002, 0.060, 0.3160, 0.009, 0.000, 0.0000],
                         [0.016, 0.007, 0.021, 0.017, 0.057, 0.0200, 0.040, 0.225, 0.0512],
                         [0.036, 0.001, 0.003, 0.001, 0.028, 0.0080, 0.066, 0.144, 0.0196]])

  def __init__(self, bg_type=0, stim_type=0, conn_type=0, poisson_freq=8., has_thalamus=False):
    super(CorticalMicrocircuit, self).__init__()

    # parameters
    self.bg_type = bg_type
    self.stim_type = stim_type
    self.conn_type = conn_type
    self.poisson_freq = poisson_freq
    self.has_thalamus = has_thalamus

    # NEURON: populations
    self.populations = bp.Collector()
    for i in range(8):
      l_name = self.layer_name[i]
      print(f'Creating {l_name} ...')
      self.populations[l_name] = LIF(self.layer_num[i])

    # SYNAPSE: synapses
    self.synapses = bp.Collector()
    for c in range(8):  # from
      for r in range(8):  # to
        if self.conn_table[r, c] > 0.:
          print(f'Creating Synapses from {self.layer_name[c]} to {self.layer_name[r]} ...')
          syn = ExpSyn(pre=self.populations[self.layer_name[c]],
                       post=self.populations[self.layer_name[r]],
                       prob=self.conn_table[r, c],
                       syn_type=self.layer_name[c][-1],
                       conn_type=conn_type)
          self.synapses[f'{self.layer_name[c]}_to_{self.layer_name[r]}'] = syn
    # Synaptic weight from L4e to L2/3e is doubled
    self.synapses['L4e_to_L23e'].weights *= 2.

    # NEURON & SYNAPSE: poisson inputs
    if stim_type == 0:
      # print(f'Creating Poisson noise group ...')
      # self.populations['Poisson'] = PoissonInput2(
      #   freq=poisson_freq, pops=[self.populations[k] for k in self.layer_name[:-1]])
      for r in range(0, 8):
        l_name = self.layer_name[r]
        print(f'Creating Poisson group of {l_name} ...')
        N = PoissonInput(freq=poisson_freq, post=self.populations[l_name])
        self.populations[f'Poisson_to_{l_name}'] = N
    elif stim_type == 1:
      bg_inputs = self._get_bg_inputs(bg_type)
      assert bg_inputs is not None
      for i, current in enumerate(bg_inputs):
        self.populations[self.layer_name[i]].Iext = 0.3512 * current

    # NEURON & SYNAPSE: thalamus inputs
    if has_thalamus:
      thalamus = bp.dynsim.PoissonInput(self.layer_num[-1], freqs=15.)
      self.populations[self.layer_name[-1]] = thalamus
      for r in range(0, 8):
        l_name = self.layer_name[r]
        print(f'Creating Thalamus projection of {l_name} ...')
        S = ThalamusInput(pre=thalamus,
                          post=self.populations[l_name],
                          conn_prob=self.conn_table[r, 8])
        self.synapses[f'{self.layer_name[-1]}_to_{l_name}'] = S

    # finally, compose them as a network
    self.register_implicit_nodes(self.populations)
    self.register_implicit_nodes(self.synapses)

  def _get_bg_inputs(self, bg_type):
    if bg_type == 0:  # layer-specific
      bg_layer = self.layer_specific_bg
    elif bg_type == 1:  # layer-independent
      bg_layer = self.layer_independent_bg
    elif bg_type == 2:  # layer-independent-random
      bg_layer = np.zeros(8)
      for i in range(0, 8, 2):
        # randomly choosing a number for the external input to an excitatory population:
        exc_bound = [self.layer_specific_bg[i], self.layer_independent_bg[i]]
        exc_input = np.random.uniform(min(exc_bound), max(exc_bound))
        # randomly choosing a number for the external input to an inhibitory population:
        T = 0.1 if i != 6 else 0.2
        inh_bound = ((1 - T) / (1 + T)) * exc_input  # eq. 4 from the article
        inh_input = np.random.uniform(inh_bound, exc_input)
        # array created to save the values:
        bg_layer[i] = int(exc_input)
        bg_layer[i + 1] = int(inh_input)
    else:
      bg_layer = None
    return bg_layer


bm.random.seed()
net = CorticalMicrocircuit(conn_type=2, poisson_freq=8., stim_type=1, bg_type=0)
sps_monitors = [f'{n}.spike' for n in net.layer_name[:-1]]
runner = bp.dynsim.StructRunner(net, monitors=sps_monitors)
runner.run(1000.)

spikes = np.hstack([runner.mon[name] for name in sps_monitors])
bp.visualize.raster_plot(runner.mon.ts, spikes, show=True)


# bp.visualize.line_plot(runner.mon.ts, runner.mon['L4e.V'], plot_ids=[0, 1, 2], show=True)

# def run1():
# fig, gs = bp.visualize.get_figure(8, 1, col_len=8, row_len=1)
# for i in range(8):
#   fig.add_subplot(gs[i, 0])
#   name = net.layer_name[i]
#   bp.visualize.raster_plot(runner.mon.ts, runner.mon[f'{name}.spike'],
#                            xlabel='Time [ms]' if i == 7 else None,
#                            ylabel=name, show=i == 7)


# if __name__ == '__main__':
#     run1()
