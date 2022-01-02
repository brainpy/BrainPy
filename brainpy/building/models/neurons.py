# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.errors import ModelBuildError
from brainpy.integrators.ode import odeint
from brainpy.building.brainobjects import NeuGroup

__all__ = [
  'LIF',
  'Izhikevich',
  'AdExIF',
  'SpikeTimeInput',
  'PoissonInput',
]


class LIF(NeuGroup):
  r"""Leaky integrate-and-fire neuron model.

  **Model Descriptions**

  The formal equations of a LIF model [1]_ is given by:

  .. math::

      \tau \frac{dV}{dt} = - (V(t) - V_{rest}) + I(t) \\
      \text{after} \quad V(t) \gt V_{th}, V(t) = V_{reset} \quad
      \text{last} \quad \tau_{ref} \quad  \text{ms}

  where :math:`V` is the membrane potential, :math:`V_{rest}` is the resting
  membrane potential, :math:`V_{reset}` is the reset membrane potential,
  :math:`V_{th}` is the spike threshold, :math:`\tau` is the time constant,
  :math:`\tau_{ref}` is the refractory time period,
  and :math:`I` is the time-variant synaptic inputs.

  **Model Examples**

  - `(Brette, Romain. 2004) LIF phase locking <https://brainpy-examples.readthedocs.io/en/latest/neurons/Romain_2004_LIF_phase_locking.html>`_

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>>
    >>> group = bp.math.jit(bp.models.LIF(1))
    >>>
    >>> runner = bp.StructRunner(group, monitors=['V'], inputs=('input', 26.))
    >>> runner.run(200.)
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.V, show=True)


  **Model Parameters**

  ============= ============== ======== =========================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------
  V_rest         0              mV       Resting membrane potential.
  V_reset        -5             mV       Reset potential after spike.
  V_th           20             mV       Threshold potential of spike.
  tau            10             ms       Membrane time constant. Compute by R * C.
  tau_ref       5              ms       Refractory period length.(ms)
  ============= ============== ======== =========================================

  **Neuron Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                    0                Membrane potential.
  input                0                External and synaptic input current.
  spike                False             Flag to mark whether the neuron is spiking.
  refractory           False             Flag to mark whether the neuron is in refractory period.
  t_last_spike         -1e7              Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] Abbott, Larry F. "Lapicque’s introduction of the integrate-and-fire model
         neuron (1907)." Brain research bulletin 50, no. 5-6 (1999): 303-304.
  """

  def __init__(self, size, V_rest=0., V_reset=-5., V_th=20., tau=10.,
               tau_ref=1., method='exp_auto', **kwargs):
    # initialization
    super(LIF, self).__init__(size=size, **kwargs)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.tau = tau
    self.tau_ref = tau_ref

    # variables
    self.V = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))

    # integral
    self.int_V = odeint(method=method, f=self.dV)

  def dV(self, V, t, Iext):
    dvdt = (-V + self.V_rest + Iext) / self.tau
    return dvdt

  def update(self, _t, _dt):
    refractory = (_t - self.t_last_spike) <= self.tau_ref
    V = self.int_V(self.V, _t, self.input, dt=_dt)
    V = bm.where(refractory, self.V, V)
    spike = V >= self.V_th
    self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
    self.V.value = bm.where(spike, self.V_reset, V)
    self.refractory.value = bm.logical_or(refractory, spike)
    self.spike.value = spike
    self.input[:] = 0.


class Izhikevich(NeuGroup):
  r"""The Izhikevich neuron model.

  **Model Descriptions**

  The dynamics of the Izhikevich neuron model [1]_ [2]_ is given by:

  .. math ::

      \frac{d V}{d t} &= 0.04 V^{2}+5 V+140-u+I

      \frac{d u}{d t} &=a(b V-u)

  .. math ::

      \text{if}  v \geq 30  \text{mV}, \text{then}
      \begin{cases} v \leftarrow c \\
      u \leftarrow u+d \end{cases}

  **Model Examples**

  - `Detailed examples to reproduce different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Izhikevich_2003_Izhikevich_model.html>`_

  **Model Parameters**

  ============= ============== ======== ================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- --------------------------------------------------------------------------------
  a             0.02           \        It determines the time scale of
                                        the recovery variable :math:`u`.
  b             0.2            \        It describes the sensitivity of the
                                        recovery variable :math:`u` to
                                        the sub-threshold fluctuations of the
                                        membrane potential :math:`v`.
  c             -65            \        It describes the after-spike reset value
                                        of the membrane potential :math:`v` caused by
                                        the fast high-threshold :math:`K^{+}`
                                        conductance.
  d             8              \        It describes after-spike reset of the
                                        recovery variable :math:`u`
                                        caused by slow high-threshold
                                        :math:`Na^{+}` and :math:`K^{+}` conductance.
  tau_ref       0              ms       Refractory period length. [ms]
  V_th          30             mV       The membrane potential threshold.
  ============= ============== ======== ================================================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                          -65        Membrane potential.
  u                          1          Recovery variable.
  input                      0          External and synaptic input current.
  spike                      False      Flag to mark whether the neuron is spiking.
  refractory                False       Flag to mark whether the neuron is in refractory period.
  t_last_spike               -1e7       Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] Izhikevich, Eugene M. "Simple model of spiking neurons." IEEE
         Transactions on neural networks 14.6 (2003): 1569-1572.

  .. [2] Izhikevich, Eugene M. "Which model to use for cortical spiking neurons?."
         IEEE transactions on neural networks 15.5 (2004): 1063-1070.
  """

  def __init__(self, size, a=0.02, b=0.20, c=-65., d=8., tau_ref=0.,
               V_th=30., method='exp_auto', **kwargs):
    # initialization
    super(Izhikevich, self).__init__(size=size, **kwargs)

    # params
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.V_th = V_th
    self.tau_ref = tau_ref

    # variables
    self.u = bm.Variable(bm.ones(self.num))
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.V = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

    # functions
    self.int_V = odeint(method=method, f=self.dV)
    self.int_u = odeint(method=method, f=self.du)

  def dV(self, V, t, u, Iext):
    dVdt = 0.04 * V * V + 5 * V + 140 - u + Iext
    return dVdt

  def du(self, u, t, V):
    dudt = self.a * (self.b * V - u)
    return dudt

  def update(self, _t, _dt):
    V = self.int_V(self.V, _t, self.u, self.input, dt=_dt)
    u = self.int_u(self.u, _t, self.V, dt=_dt)
    refractory = (_t - self.t_last_spike) <= self.tau_ref
    V = bm.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
    self.V.value = bm.where(spike, self.c, V)
    self.u.value = bm.where(spike, u + self.d, u)
    self.refractory.value = bm.logical_or(refractory, spike)
    self.spike.value = spike
    self.input[:] = 0.


class AdExIF(NeuGroup):
  r"""Adaptive exponential integrate-and-fire neuron model.

  **Model Descriptions**

  The **adaptive exponential integrate-and-fire model**, also called AdEx, is a
  spiking neuron model with two variables [1]_ [2]_.

  .. math::

      \begin{aligned}
      \tau_m\frac{d V}{d t} &= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} - Rw + RI(t), \\
      \tau_w \frac{d w}{d t} &=a(V-V_{rest}) - w
      \end{aligned}

  once the membrane potential reaches the spike threshold,

  .. math::

      V \rightarrow V_{reset}, \\
      w \rightarrow w+b.

  The first equation describes the dynamics of the membrane potential and includes
  an activation term with an exponential voltage dependence. Voltage is coupled to
  a second equation which describes adaptation. Both variables are reset if an action
  potential has been triggered. The combination of adaptation and exponential voltage
  dependence gives rise to the name Adaptive Exponential Integrate-and-Fire model.

  The adaptive exponential integrate-and-fire model is capable of describing known
  neuronal firing patterns, e.g., adapting, bursting, delayed spike initiation,
  initial bursting, fast spiking, and regular spiking.

  **Model Examples**

  - `Examples for different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/AdExIF_model.html>`_

  **Model Parameters**

  ============= ============== ======== ========================================================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
  V_rest        -65            mV       Resting potential.
  V_reset       -68            mV       Reset potential after spike.
  V_th          -30            mV       Threshold potential of spike and reset.
  V_T           -59.9          mV       Threshold potential of generating action potential.
  delta_T       3.48           \        Spike slope factor.
  a             1              \        The sensitivity of the recovery variable :math:`u` to the sub-threshold fluctuations of the membrane potential :math:`v`
  b             1              \        The increment of :math:`w` produced by a spike.
  R             1              \        Membrane resistance.
  tau           10             ms       Membrane time constant. Compute by R * C.
  tau_w         30             ms       Time constant of the adaptation current.
  ============= ============== ======== ========================================================================================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                   0                 Membrane potential.
  w                   0                 Adaptation current.
  input               0                 External and synaptic input current.
  spike               False             Flag to mark whether the neuron is spiking.
  t_last_spike        -1e7              Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
         mechanisms determine the neuronal response to fluctuating
         inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
  .. [2] http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model
  """

  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=-30., V_T=-59.9, delta_T=3.48, a=1.,
               b=1., tau=10., tau_w=30., R=1., method='exp_auto', **kwargs):
    super(AdExIF, self).__init__(size=size, **kwargs)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_T = V_T
    self.delta_T = delta_T
    self.a = a
    self.b = b
    self.tau = tau
    self.tau_w = tau_w
    self.R = R

    # variables
    self.w = bm.Variable(bm.zeros(self.num))
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.V = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

    # functions
    self.int_V = odeint(method=method, f=self.dV)
    self.int_w = odeint(method=method, f=self.dw)

  def dV(self, V, t, w, Iext):
    dVdt = (- V + self.V_rest + self.delta_T * bm.exp((V - self.V_T) / self.delta_T) -
            self.R * w + self.R * Iext) / self.tau
    return dVdt

  def dw(self, w, t, V):
    dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
    return dwdt

  def update(self, _t, _dt):
    V = self.int_V(self.V, _t, self.w, self.input, dt=_dt)
    w = self.int_w(self.w, _t, self.V, dt=_dt)
    spike = V >= self.V_th
    self.t_last_spike[:] = bm.where(spike, _t, self.t_last_spike)
    self.V.value = bm.where(spike, self.V_reset, V)
    self.w.value = bm.where(spike, w + self.b, w)
    self.spike.value = spike
    self.input[:] = 0.


class SpikeTimeInput(NeuGroup):
  """The input neuron group characterized by spikes emitting at given times.

  >>> # Get 2 neurons, firing spikes at 10 ms and 20 ms.
  >>> SpikeTimeInput(2, times=[10, 20])
  >>> # or
  >>> # Get 2 neurons, the neuron 0 fires spikes at 10 ms and 20 ms.
  >>> SpikeTimeInput(2, times=[10, 20], indices=[0, 0])
  >>> # or
  >>> # Get 2 neurons, neuron 0 fires at 10 ms and 30 ms, neuron 1 fires at 20 ms.
  >>> SpikeTimeInput(2, times=[10, 20, 30], indices=[0, 1, 0])
  >>> # or
  >>> # Get 2 neurons; at 10 ms, neuron 0 fires; at 20 ms, neuron 0 and 1 fire;
  >>> # at 30 ms, neuron 1 fires.
  >>> SpikeTimeInput(2, times=[10, 20, 20, 30], indices=[0, 0, 1, 1])

  Parameters
  ----------
  size : int, tuple, list
      The neuron group geometry.
  indices : int, list, tuple
      The neuron indices at each time point to emit spikes.
  times : list, np.ndarray
      The time points which generate the spikes.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, size, times, indices, need_sort=True, name=None):
    super(SpikeTimeInput, self).__init__(size=size, name=name)

    # parameters
    if len(indices) != len(times):
      raise ModelBuildError(f'The length of "indices" and "times" must be the same. '
                            f'However, we got {len(indices)} != {len(times)}.')
    self.num_times = len(times)

    # data about times and indices
    self.i = bm.Variable(bm.zeros(1, dtype=bm.int_))
    self.times = bm.Variable(bm.asarray(times, dtype=bm.float_))
    self.indices = bm.Variable(bm.asarray(indices, dtype=bm.int_))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    if need_sort:
      sort_idx = bm.argsort(times)
      self.indices.value = self.indices[sort_idx]
      self.times.value = self.times[sort_idx]

    # functions
    def cond_fun(t):
      return bm.logical_and(self.i[0] < self.num_times, t >= self.times[self.i[0]])

    def body_fun(t):
      self.spike[self.indices[self.i[0]]] = True
      self.i[0] += 1

    self._run = bm.make_while(cond_fun, body_fun, dyn_vars=self.vars())

  def update(self, _t, _i, **kwargs):
    self.spike[:] = False
    self._run(_t)


class PoissonInput(NeuGroup):
  """Poisson Neuron Group.

  Parameters
  ----------
  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, size, freqs, seed=None, **kwargs):
    super(PoissonInput, self).__init__(size=size, **kwargs)

    self.freqs = freqs
    self.dt = bm.get_dt() / 1000.
    self.size = (size,) if isinstance(size, int) else tuple(size)
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)
    self.rng = bm.random.RandomState(seed=seed)

  def update(self, _t, _i):
    self.spike.update(self.rng.random(self.num) <= self.freqs * self.dt)
    self.t_last_spike.update(bm.where(self.spike, _t, self.t_last_spike))

