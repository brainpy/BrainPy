# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: brainpy
#     language: python
#     name: brainpy
# ---

# %% [markdown]
# # *(Wang, 2002)*: Decision making spiking model

# %% [markdown]
# Implementation of the paper: *Wang, Xiao-Jing. "Probabilistic decision making by slow reverberation in cortical circuits." Neuron 36.5 (2002): 955-968.*
#
# - Author : Chaoming Wang (chao.brain@qq.com), Xinyu Liu (adaliu1998@163.com)

# %%
# import os
# os.environ['JAX_CHECK_TRACER_LEAKS'] = '1'

import sys

sys.path.append('../../')

# %%
import matplotlib.pyplot as plt
import numpy as np
import brainpy as bp

# %%
bp.backend.set('numba', dt=0.1)


# %% [markdown]
# ## Neuron model

# %% [markdown]
# ### LIF neurons

# %% [markdown]
# Both pyramidal cells and interneurons are described by leaky integrate-and-fire neurons. 
#
# $$
# C_{m} \frac{d V(t)}{d t}=-g_{L}\left(V(t)-V_{L}\right)-I_{s y n}(t)
# $$
# where 
# - $I_{syn}(t)$ represents the total synaptic current flowing into the cell
# - resting potential $V_L$ = -70 mV
# - firing threshold $V_{th}$ = -50 mV
# - reset potential $V_{rest}$ = -55 mV
# - membrane capacitance $C_m$ = 0.5 nF for pyramidal cells and 0.2 nF for interneurons
# - membrane leak conductance $g_L$ = 25 nS for pyramidal cells and 20 nS for interneurons
# - refractory period $\tau_{ref}$ = 2 ms for pyramidal cells and 1 ms for interneurons

# %%
class LIF(bp.NeuGroup):
  target_backend = 'general'

  def __init__(self, size, V_L=-70., V_reset=-55., V_th=-50.,
               Cm=0.5, gL=0.025, t_refractory=2., **kwargs):
    super(LIF, self).__init__(size=size, **kwargs)

    self.V_L = V_L
    self.V_reset = V_reset
    self.V_th = V_th
    self.Cm = Cm
    self.gL = gL
    self.t_refractory = t_refractory

    self.V = np.zeros(self.num) * V_L
    self.input = np.zeros(self.num)
    self.spike = np.zeros(self.num, dtype=np.bool_)
    self.refractory = np.zeros(self.num, dtype=np.bool_)
    self.t_last_spike = np.ones(self.num) * -1e7

  @staticmethod
  @bp.odeint(method='exponential_euler')
  def integral(V, t, Iext, gL, V_L, Cm):
    dVdt = (-gL * (V - V_L) - Iext) / Cm
    return dVdt

  def update(self, _t, _i):
    ref = (_t - self.t_last_spike) <= self.t_refractory
    V = self.integral(self.V, _t, self.input, self.gL, self.V_L, self.Cm)
    V = np.where(ref, self.V, V)
    spike = (V >= self.V_th)
    self.V = np.where(spike, self.V_reset, V)
    self.spike = spike
    self.t_last_spike = np.where(spike, _t, self.t_last_spike)
    self.refractory = np.logical_or(spike, ref)
    self.input[:] = 0.


# %% [markdown]
# ### Poisson neurons


# %%
class PoissonNoise(bp.NeuGroup):
  target_backend = 'general'

  def __init__(self, size, freqs, **kwargs):
    super(PoissonNoise, self).__init__(size=size, **kwargs)

    self.freqs = freqs
    self.dt = bp.backend.get_dt() / 1000.
    self.spike = np.zeros(self.num, dtype=bool)

  def update(self, _t, _i):
    self.spike[:] = np.random.random(self.num) < self.freqs * self.dt


# %%
class PoissonStimulus(bp.NeuGroup):
  """
  from time <t_start> to <t_end> during the simulation, the neuron
  generates a possion spike with frequency <self.freq>. however,
  the value of <self.freq> changes every <t_interval> ms and obey
  a Gaussian distribution defined by <mean_freq> and <var_freq>.
  """
  target_backend = 'general'

  def __init__(self, size, t_start=0., t_end=0., t_interval=0.,
               freq_mean=0., freq_var=20., **kwargs):
    super(PoissonStimulus, self).__init__(size=size, **kwargs)

    self.dt = bp.backend.get_dt() / 1000
    self.t_start = t_start
    self.t_end = t_end
    self.t_interval = t_interval
    self.freq_mean = freq_mean
    self.freq_var = freq_var
    self.freq = 0.
    self.t_last_change = -1e7
    self.spike = np.zeros(size, dtype=bool)

  def update(self, _t, _i):
    if self.t_start < _t < self.t_end:
      if _t - self.t_last_change >= self.t_interval:
        self.freq = np.random.normal(self.freq_mean, self.freq_var)
        self.freq = max(self.freq, 0)
        self.t_last_change = _t
      self.spike = np.random.random(self.num) < (self.freq * self.dt)
    else:
      self.freq = 0.
      self.spike[:] = False


# %% [markdown]
# ## Synapse models

# %% [markdown]
# The total synaptic currents are given by
#
# $$
# I_{s y n}(t)=I_{e x t, A M P A}(t)+I_{r e c, A M P A}(t)+I_{r e c, N M D A}(t)+I_{r e c, G A B A}(t)
# $$
#
# in which
#
# $$
# \begin{gathered}
# I_{\mathrm{ext}, \mathrm{AMPA}}(t)=g_{\mathrm{ext}, \mathrm{AMPA}}\left(V(t)-V_{E}\right) \mathrm{s}^{\mathrm{ext}, \mathrm{AMPA}}(t) \\
# I_{\mathrm{rec}, \mathrm{AMPA}}(t)=g_{\mathrm{rec}, \mathrm{AMPA}}\left(V(t)-V_{E}\right) \sum_{\mathrm{j}=1}^{\mathrm{C}_{\mathrm{E}}} W_{j} S_{j}^{\mathrm{AMPA}}(t) \\
# I_{\mathrm{rec}, \mathrm{NMDA}}(t)=\frac{g_{\mathrm{NMDA}}\left(V(t)-V_{E}\right)}{\left(1+\left[\mathrm{Mg}^{2+}\right] \exp (-0.062 V(t)) / 3.57\right)} \sum_{j=1}^{\mathrm{c}_{E}} w_{j} \mathrm{~s}_{j}^{\mathrm{NMDA}}(t) \\
# I_{\mathrm{rec}, \mathrm{GABA}}(t)=g_{\mathrm{GABA}}\left(V(t)-V_{l}\right) \sum_{j=1}^{c_{1}} s_{j}^{\mathrm{GABA}}(t)
# \end{gathered}
# $$

# %% [markdown]
# where 
#
# - $V_E$ = 0 mV
# - $V_I$ = -70 mV 
# - $\left[\mathrm{Mg}^{2+}\right]$ = 1 mM
# - The dimensionless weights $w_j$ represent the structured excitatory recurrent connections
# - the sum over $j$ represents a sum over the synapses formed by presynaptic neurons $j$

# %% [markdown]
# ### AMPA
#
# The AMPA (external and recurrent) channels are described by
#
# $$
# \frac{d s_{j}^{A M P A}(t)}{d t}=-\frac{s_{j}^{A M P A}(t)}{\tau_{A M P A}}+\sum_{k} \delta\left(t-t_{j}^{k}\right)
# $$
#
# where 
#
# - the decay time of AMPA currents $\tau_{A M P A}$ = 2 ms
# - for the external AMPA currents, the spikes are emitted according to a Poisson process with rate $V_{ext}$ = 2400 Hz independently from cell to cell

# %%
class AMPA_One(bp.TwoEndConn):
  target_backend = 'general'

  def __init__(self, pre, post, delay=0.5, g_max=0.10, E=0., tau=2.0, **kwargs):
    # parameters
    self.g_max = g_max
    self.E = E
    self.tau = tau
    self.delay = delay

    # variables
    self.pre_spike = self.register_constant_delay('ps', size=pre.num, delay=delay)
    self.s = np.zeros(pre.num)
    super(AMPA_One, self).__init__(pre=pre, post=post, **kwargs)

  @staticmethod
  @bp.odeint(method='exponential_euler')
  def int_s(s, t, tau):
    ds = - s / tau
    return ds

  def update(self, _t, _i):
    self.pre_spike.push(self.pre.spike)
    pre_spike = self.pre_spike.pull()
    self.s = self.int_s(self.s, _t, self.tau)
    self.s += pre_spike * self.g_max
    self.post.input += self.s * (self.post.V - self.E)


# %%
class AMPA(bp.TwoEndConn):
  target_backend = 'general'

  def __init__(self, pre, post, delay=0.5, g_max=0.10, E=0., tau=2.0, **kwargs):
    # parameters
    self.g_max = g_max
    self.E = E
    self.tau = tau
    self.delay = delay
    self.size = (pre.num, post.num)

    # variables
    self.pre_spike = self.register_constant_delay('ps', size=pre.num, delay=delay)
    self.pre_one = np.ones(pre.num)
    self.s = np.zeros(self.size)
    super(AMPA, self).__init__(pre=pre, post=post, **kwargs)

  @staticmethod
  @bp.odeint(method='exponential_euler')
  def int_s(s, t, tau):
    ds = - s / tau
    return ds

  def update(self, _t, _i):
    self.pre_spike.push(self.pre.spike)
    pre_spike = self.pre_spike.pull()
    self.s = self.int_s(self.s, _t, self.tau)
    self.s += np.reshape(pre_spike * self.g_max, (-1, 1))
    self.post.input += np.dot(self.pre_one, self.s) * (self.post.V - self.E)


# %% [markdown]
# ### NMDA

# %% [markdown]
# NMDA channels are described by:
#
# $$
# \begin{gathered}
# \frac{d s_{j}^{\mathrm{NMDA}}(t)}{d t}=-\frac{s_{j}^{\mathrm{NMDA}}(t)}{\tau_{\mathrm{NMDA}, \text { decay }}}+\alpha x_{j}(t)\left(1-s_{j}^{\mathrm{NMDA}}(t)\right) \\
# \frac{d x_{j}(t)}{d t}=-\frac{x_{j}(t)}{\tau_{\mathrm{NMDA}, \text { rise }}}+\sum_{k} \delta\left(t-t_{j}^{k}\right)
# \end{gathered}
# $$
#
# where
#
# - the decay time $\tau_{\mathrm{NMDA}, \text { decay }}$ = 100 ms
# - $\alpha$ = 0.5 $\mathrm{ms}^{-1}$
# - the rise time $\tau_{\mathrm{NMDA}, \text { rise }}$ = 2 ms

# %%
class NMDA(bp.TwoEndConn):
  target_backend = 'general'

  def __init__(self, pre, post, delay=0.5, tau_decay=100, tau_rise=2.,
               g_max=0.15, E=0., cc_Mg=1., alpha=0.5, **kwargs):
    # parameters
    self.g_max = g_max
    self.E = E
    self.cc_Mg = cc_Mg
    self.alpha = alpha
    self.tau_decay = tau_decay
    self.tau_rise = tau_rise
    self.delay = delay
    self.size = (pre.num, post.num)

    # variables
    self.pre_spike = self.register_constant_delay('ps', size=pre.num, delay=delay)
    self.pre_one = np.ones(pre.num)
    self.s = np.zeros(self.size)
    self.x = np.zeros(self.size)
    super(NMDA, self).__init__(pre=pre, post=post, **kwargs)

  @staticmethod
  @bp.odeint(method='exponential_euler')
  def integral(s, x, t, tau_decay, tau_rise, alpha):
    dsdt = -s / tau_decay + alpha * x * (1 - s)
    dxdt = -x / tau_rise
    return dsdt, dxdt

  def update(self, _t, _i):
    self.pre_spike.push(self.pre.spike)
    pre_spike = self.pre_spike.pull()
    self.s, self.x = self.integral(self.s, self.x, _t, self.tau_decay, self.tau_rise, self.alpha)
    self.x += np.reshape(pre_spike, (-1, 1))

    g_inf = 1 / (1 + self.cc_Mg * np.exp(-0.062 * self.post.V) / 3.57)
    Iext = np.dot(self.pre_one, self.s) * (self.post.V - self.E) * g_inf
    self.post.input += Iext * self.g_max


# %% [markdown]
# ### GABAA
#
# The GABA synaptic variable obeys
#
# $$
# \frac{d s_{j}^{G A B A}(t)}{d t}=-\frac{s_{j}^{G A B A}(t)}{\tau_{G A B A}}+\sum_{k} \delta\left(t-t_{j}^{k}\right)
# $$
#
# where
# - the decay time of AMPA currents $\tau_{GABA}$ = 5 ms

# %%
class GABAa(AMPA):
  target_backend = 'general'

  def __init__(self, pre, post, delay=0.5, g_max=0.10, E=-70., tau=5.0, **kwargs):
    super(GABAa, self).__init__(pre=pre, post=post, E=E, tau=tau, delay=delay, g_max=g_max, **kwargs)


# %% [markdown]
# ## Parameters

# %%
scale = 1.
num_exc = int(1600 * scale)
num_inh = int(400 * scale)
f = 0.15
num_A = int(f * num_exc)
num_B = int(f * num_exc)
num_N = num_exc - num_A - num_B
print(f"N_E = {num_exc} = {num_A} + {num_B} + {num_N}, N_I = {num_inh}")

# %%
mu0 = 40.
coherence = 25.6

# %%
# times
pre_period = 100.
stim_period = 1000.
delay_period = 500.
total_period = pre_period + stim_period + delay_period

# %%
poisson_freq = 2400.  # Hz
w_pos = 1.7
w_neg = 1. - f * (w_pos - 1.) / (1. - f)
g_max_ext2E_AMPA = 2.1 * 1e-3  # uS
g_max_ext2I_AMPA = 1.62 * 1e-3  # uS
g_max_E2E_AMPA = 0.05 * 1e-3 / scale  # uS
g_max_E2E_NMDA = 0.165 * 1e-3 / scale  # uS
g_max_E2I_AMPA = 0.04 * 1e-3 / scale  # uS
g_max_E2I_NMDA = 0.13 * 1e-3 / scale  # uS
g_max_I2E_GABAa = 1.3 * 1e-3 / scale  # uS
g_max_I2I_GABAa = 1.0 * 1e-3 / scale  # uS

# %% [markdown]
# ## Build the network


# %%
# E neurons/pyramid neurons
A = LIF(num_A, Cm=0.5, gL=0.025, t_refractory=2., monitors=['spike'])
B = LIF(num_B, Cm=0.5, gL=0.025, t_refractory=2., monitors=['spike'])
N = LIF(num_N, Cm=0.5, gL=0.025, t_refractory=2., monitors=['spike'])
# I neurons/interneurons
I = LIF(num_inh, Cm=0.2, gL=0.020, t_refractory=1.)

# %%
IA = PoissonStimulus(num_A,
                     t_start=pre_period,
                     t_end=pre_period + stim_period,
                     t_interval=50.,
                     freq_mean=mu0 + mu0 / 100. * coherence,
                     freq_var=10.,
                     monitors=['freq'])
IB = PoissonStimulus(num_B,
                     t_start=pre_period,
                     t_end=pre_period + stim_period,
                     t_interval=50.,
                     freq_mean=mu0 - mu0 / 100. * coherence,
                     freq_var=10.,
                     monitors=['freq'])

# %%
noise_A = PoissonNoise(num_A, freqs=poisson_freq)
noise_B = PoissonNoise(num_B, freqs=poisson_freq)
noise_N = PoissonNoise(num_N, freqs=poisson_freq)
noise_I = PoissonNoise(num_inh, freqs=poisson_freq)

# %%
IA2A = AMPA_One(pre=IA, post=A, g_max=g_max_ext2E_AMPA)
IB2B = AMPA_One(pre=IB, post=B, g_max=g_max_ext2E_AMPA)

# %%
## define E2E conn
A2A_AMPA = AMPA(pre=A, post=A, g_max=g_max_E2E_AMPA * w_pos)
A2A_NMDA = NMDA(pre=A, post=A, g_max=g_max_E2E_NMDA * w_pos)

A2B_AMPA = AMPA(pre=A, post=B, g_max=g_max_E2E_AMPA * w_neg)
A2B_NMDA = NMDA(pre=A, post=B, g_max=g_max_E2E_NMDA * w_neg)

A2N_AMPA = AMPA(pre=A, post=N, g_max=g_max_E2E_AMPA)
A2N_NMDA = NMDA(pre=A, post=N, g_max=g_max_E2E_NMDA)

B2A_AMPA = AMPA(pre=B, post=A, g_max=g_max_E2E_AMPA * w_neg)
B2A_NMDA = NMDA(pre=B, post=A, g_max=g_max_E2E_NMDA * w_neg)

B2B_AMPA = AMPA(pre=B, post=B, g_max=g_max_E2E_AMPA * w_pos)
B2B_NMDA = NMDA(pre=B, post=B, g_max=g_max_E2E_NMDA * w_pos)

B2N_AMPA = AMPA(pre=B, post=N, g_max=g_max_E2E_AMPA)
B2N_NMDA = NMDA(pre=B, post=N, g_max=g_max_E2E_NMDA)

N2A_AMPA = AMPA(pre=N, post=A, g_max=g_max_E2E_AMPA * w_neg)
N2A_NMDA = NMDA(pre=N, post=A, g_max=g_max_E2E_NMDA * w_neg)

N2B_AMPA = AMPA(pre=N, post=B, g_max=g_max_E2E_AMPA * w_neg)
N2B_NMDA = NMDA(pre=N, post=B, g_max=g_max_E2E_NMDA * w_neg)

N2N_AMPA = AMPA(pre=N, post=N, g_max=g_max_E2E_AMPA)
N2N_NMDA = NMDA(pre=N, post=N, g_max=g_max_E2E_NMDA)

## define E2I conn
A2I_AMPA = AMPA(pre=A, post=I, g_max=g_max_E2I_AMPA)
A2I_NMDA = NMDA(pre=A, post=I, g_max=g_max_E2I_NMDA)

B2I_AMPA = AMPA(pre=B, post=I, g_max=g_max_E2I_AMPA)
B2I_NMDA = NMDA(pre=B, post=I, g_max=g_max_E2I_NMDA)

N2I_AMPA = AMPA(pre=N, post=I, g_max=g_max_E2I_AMPA)
N2I_NMDA = NMDA(pre=N, post=I, g_max=g_max_E2I_NMDA)

I2A_GABAa = GABAa(pre=I, post=A, g_max=g_max_I2E_GABAa)
I2B_GABAa = GABAa(pre=I, post=B, g_max=g_max_I2E_GABAa)
I2N_GABAa = GABAa(pre=I, post=N, g_max=g_max_I2E_GABAa)

## define I2I conn
I2I_GABAa = GABAa(pre=I, post=I, g_max=g_max_I2I_GABAa)

## define external projections
noise2A_AMPA = AMPA_One(pre=noise_A, post=A, g_max=g_max_ext2E_AMPA)
noise2B_AMPA = AMPA_One(pre=noise_B, post=B, g_max=g_max_ext2E_AMPA)
noise2N_AMPA = AMPA_One(pre=noise_N, post=N, g_max=g_max_ext2E_AMPA)
noise2I_AMPA = AMPA_One(pre=noise_I, post=I, g_max=g_max_ext2I_AMPA)

# %%
# build & simulate network
net = bp.Network(
  A, B, N, I, noise_A, noise_B, noise_N, noise_I, IA, IB,
  noise2A_AMPA, noise2B_AMPA, noise2N_AMPA, noise2I_AMPA,
  IA2A, IB2B, N2N_AMPA, N2N_NMDA,
  A2A_AMPA, A2A_NMDA, A2B_AMPA, A2B_NMDA, A2N_AMPA, A2N_NMDA, B2A_AMPA, B2A_NMDA,
  B2B_AMPA, B2B_NMDA, B2N_AMPA, B2N_NMDA, N2A_AMPA, N2A_NMDA, N2B_AMPA, N2B_NMDA,
  A2I_AMPA, A2I_NMDA, B2I_AMPA, B2I_NMDA, N2I_AMPA, N2I_NMDA,
  I2A_GABAa, I2B_GABAa, I2N_GABAa, I2I_GABAa
)

net.run(duration=total_period, report=True, report_percent=0.01)


# %% [markdown]
# ## Visualization

# %%
def firing_rate(sp_matrix, width):
  rate = np.sum(sp_matrix, axis=1) / sp_matrix.shape[1]
  dt = bp.backend.get_dt()
  width1 = int(width / 2 / dt) * 2 + 1
  window = np.ones(width1) * 1000 / width
  return np.convolve(rate, window, mode='same')


# %%
t_start = 10.

fig, gs = bp.visualize.get_figure(4, 1, 3, 10)

fig.add_subplot(gs[0, 0])
bp.visualize.raster_plot(A.mon.ts, A.mon.spike, markersize=1)
plt.title("Spiking activity of group A")
plt.ylabel("Neuron Index")
plt.xlim(t_start, total_period + 1)
plt.axvline(pre_period, linestyle='dashed')
plt.axvline(pre_period + stim_period, linestyle='dashed')
plt.axvline(pre_period + stim_period + delay_period, linestyle='dashed')

fig.add_subplot(gs[1, 0])
bp.visualize.raster_plot(B.mon.ts, B.mon.spike, markersize=1)
plt.title("Spiking activity of group B")
plt.ylabel("Neuron Index")
plt.xlim(t_start, total_period + 1)
plt.axvline(pre_period, linestyle='dashed')
plt.axvline(pre_period + stim_period, linestyle='dashed')
plt.axvline(pre_period + stim_period + delay_period, linestyle='dashed')

fig.add_subplot(gs[2, 0])
plt.plot(IA.mon.ts, IA.mon.freq, label="group A")
plt.plot(IB.mon.ts, IB.mon.freq, label="group B")
plt.title("Input activity")
plt.ylabel("Firing rate [Hz]")
plt.xlim(t_start, total_period + 1)
plt.axvline(pre_period, linestyle='dashed')
plt.axvline(pre_period + stim_period, linestyle='dashed')
plt.axvline(pre_period + stim_period + delay_period, linestyle='dashed')
plt.legend()

fig.add_subplot(gs[3, 0])
rateA = firing_rate(A.mon.spike, width=10.)
rateB = firing_rate(B.mon.spike, width=10.)
plt.plot(A.mon.ts, rateA, label="Group A")
plt.plot(B.mon.ts, rateB, label="Group B")
plt.ylabel('Firing rate [Hz]')
plt.title("Population activity")
plt.xlim(t_start, total_period + 1)
plt.axvline(pre_period, linestyle='dashed')
plt.axvline(pre_period + stim_period, linestyle='dashed')
plt.axvline(pre_period + stim_period + delay_period, linestyle='dashed')
plt.legend()

plt.xlabel("Time [ms]")
plt.show()
