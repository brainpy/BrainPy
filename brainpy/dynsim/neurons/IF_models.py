# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.integrators.joint_eq import JointEq
from brainpy.integrators.ode import odeint
from brainpy.dynsim.base import NeuGroup

__all__ = [
  'LIF',
  'ExpIF',
  'AdExIF',
  'QuaIF',
  'AdQuaIF',
  'GIF',
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
               tau_ref=1., method='exp_auto', name=None):
    # initialization
    super(LIF, self).__init__(size=size, name=name)

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
    self.integral = odeint(method=method, f=self.derivative)

  def derivative(self, V, t):
    dvdt = (-V + self.V_rest + self.input) / self.tau
    return dvdt

  def update(self, _t, _dt):
    refractory = (_t - self.t_last_spike) <= self.tau_ref
    V = self.integral(self.V, _t, dt=_dt)
    V = bm.where(refractory, self.V, V)
    spike = V >= self.V_th
    self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
    self.V.value = bm.where(spike, self.V_reset, V)
    self.refractory.value = bm.logical_or(refractory, spike)
    self.spike.value = spike
    self.input[:] = 0.


class ExpIF(NeuGroup):
  r"""Exponential integrate-and-fire neuron model.

  **Model Descriptions**

  In the exponential integrate-and-fire model [1]_, the differential
  equation for the membrane potential is given by

  .. math::

      \tau\frac{d V}{d t}= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} + RI(t), \\
      \text{after} \, V(t) \gt V_{th}, V(t) = V_{reset} \, \text{last} \, \tau_{ref} \, \text{ms}

  This equation has an exponential nonlinearity with "sharpness" parameter :math:`\Delta_{T}`
  and "threshold" :math:`\vartheta_{rh}`.

  The moment when the membrane potential reaches the numerical threshold :math:`V_{th}`
  defines the firing time :math:`t^{(f)}`. After firing, the membrane potential is reset to
  :math:`V_{rest}` and integration restarts at time :math:`t^{(f)}+\tau_{\rm ref}`,
  where :math:`\tau_{\rm ref}` is an absolute refractory time.
  If the numerical threshold is chosen sufficiently high, :math:`V_{th}\gg v+\Delta_T`,
  its exact value does not play any role. The reason is that the upswing of the action
  potential for :math:`v\gg v +\Delta_{T}` is so rapid, that it goes to infinity in
  an incredibly short time. The threshold :math:`V_{th}` is introduced mainly for numerical
  convenience. For a formal mathematical analysis of the model, the threshold can be pushed
  to infinity.

  The model was first introduced by Nicolas Fourcaud-Trocmé, David Hansel, Carl van Vreeswijk
  and Nicolas Brunel [1]_. The exponential nonlinearity was later confirmed by Badel et al. [3]_.
  It is one of the prominent examples of a precise theoretical prediction in computational
  neuroscience that was later confirmed by experimental neuroscience.

  Two important remarks:

  - (i) The right-hand side of the above equation contains a nonlinearity
    that can be directly extracted from experimental data [3]_. In this sense the exponential
    nonlinearity is not an arbitrary choice but directly supported by experimental evidence.
  - (ii) Even though it is a nonlinear model, it is simple enough to calculate the firing
    rate for constant input, and the linear response to fluctuations, even in the presence
    of input noise [4]_.

  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> group = bp.neurons.ExpIF(1)
    >>> runner = bp.DSRunner(group, monitors=['V'], inputs=('input', 10.))
    >>> runner.run(300., )
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='V', show=True)


  **Model Parameters**

  ============= ============== ======== ===================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ---------------------------------------------------
  V_rest        -65            mV       Resting potential.
  V_reset       -68            mV       Reset potential after spike.
  V_th          -30            mV       Threshold potential of spike.
  V_T           -59.9          mV       Threshold potential of generating action potential.
  delta_T       3.48           \        Spike slope factor.
  R             1              \        Membrane resistance.
  tau           10             \        Membrane time constant. Compute by R * C.
  tau_ref       1.7            \        Refractory period length.
  ============= ============== ======== ===================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                  0                 Membrane potential.
  input              0                 External and synaptic input current.
  spike              False             Flag to mark whether the neuron is spiking.
  refractory         False             Flag to mark whether the neuron is in refractory period.
  t_last_spike       -1e7              Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
         mechanisms determine the neuronal response to fluctuating
         inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
  .. [2] Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014).
         Neuronal dynamics: From single neurons to networks and models
         of cognition. Cambridge University Press.
  .. [3] Badel, Laurent, Sandrine Lefort, Romain Brette, Carl CH Petersen,
         Wulfram Gerstner, and Magnus JE Richardson. "Dynamic IV curves
         are reliable predictors of naturalistic pyramidal-neuron voltage
         traces." Journal of Neurophysiology 99, no. 2 (2008): 656-666.
  .. [4] Richardson, Magnus JE. "Firing-rate response of linear and nonlinear
         integrate-and-fire neurons to modulated current-based and
         conductance-based synaptic drive." Physical Review E 76, no. 2 (2007): 021919.
  .. [5] https://en.wikipedia.org/wiki/Exponential_integrate-and-fire
  """

  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=-30., V_T=-59.9, delta_T=3.48,
               R=1., tau=10., tau_ref=1.7, method='exp_auto', name=None):
    # initialize
    super(ExpIF, self).__init__(size=size, name=name)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_T = V_T
    self.delta_T = delta_T
    self.R = R
    self.tau = tau
    self.tau_ref = tau_ref

    # variables
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))
    # variables
    self.V = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

    # integral
    self.integral = odeint(method=method, f=self.derivative)

  def derivative(self, V, t):
    exp_v = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
    dvdt = (- (V - self.V_rest) + exp_v + self.R * self.input) / self.tau
    return dvdt

  def update(self, _t, _dt):
    refractory = (_t - self.t_last_spike) <= self.tau_ref
    V = self.integral(self.V, _t, dt=_dt)
    V = bm.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
    self.V.value = bm.where(spike, self.V_reset, V)
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
               b=1., tau=10., tau_w=30., R=1., method='exp_auto', name=None):
    super(AdExIF, self).__init__(size=size, name=name)

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
    self.integral = odeint(method=method, f=self.derivative)

  def dV(self, V, t, w):
    dVdt = (- V + self.V_rest + self.delta_T * bm.exp((V - self.V_T) / self.delta_T) -
            self.R * w + self.R * self.input) / self.tau
    return dVdt

  def dw(self, w, t, V):
    dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
    return dwdt

  @property
  def derivative(self):
    return JointEq([self.dV, self.dw])

  def update(self, _t, _dt):
    V, w = self.integral(self.V, self.w, _t, dt=_dt)
    spike = V >= self.V_th
    self.t_last_spike[:] = bm.where(spike, _t, self.t_last_spike)
    self.V.value = bm.where(spike, self.V_reset, V)
    self.w.value = bm.where(spike, w + self.b, w)
    self.spike.value = spike
    self.input[:] = 0.


class QuaIF(NeuGroup):
  r"""Quadratic Integrate-and-Fire neuron model.

  **Model Descriptions**

  In contrast to physiologically accurate but computationally expensive
  neuron models like the Hodgkin–Huxley model, the QIF model [1]_ seeks only
  to produce **action potential-like patterns** and ignores subtleties
  like gating variables, which play an important role in generating action
  potentials in a real neuron. However, the QIF model is incredibly easy
  to implement and compute, and relatively straightforward to study and
  understand, thus has found ubiquitous use in computational neuroscience.

  .. math::

      \tau \frac{d V}{d t}=c(V-V_{rest})(V-V_c) + RI(t)

  where the parameters are taken to be :math:`c` =0.07, and :math:`V_c = -50 mV` (Latham et al., 2000).

  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>>
    >>> group = bp.neurons.QuaIF(1,)
    >>>
    >>> runner = bp.DSRunner(group, monitors=['V'], inputs=('input', 20.))
    >>> runner.run(duration=200.)
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.V, show=True)


  **Model Parameters**

  ============= ============== ======== ========================================================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
  V_rest        -65            mV       Resting potential.
  V_reset       -68            mV       Reset potential after spike.
  V_th          -30            mV       Threshold potential of spike and reset.
  V_c           -50            mV       Critical voltage for spike initiation. Must be larger than V_rest.
  c             .07            \        Coefficient describes membrane potential update. Larger than 0.
  R             1              \        Membrane resistance.
  tau           10             ms       Membrane time constant. Compute by R * C.
  tau_ref       0              ms       Refractory period length.
  ============= ============== ======== ========================================================================================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                   0                 Membrane potential.
  input               0                 External and synaptic input current.
  spike               False             Flag to mark whether the neuron is spiking.
  refractory          False             Flag to mark whether the neuron is in refractory period.
  t_last_spike       -1e7               Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1]  P. E. Latham, B.J. Richmond, P. Nelson and S. Nirenberg
          (2000) Intrinsic dynamics in neuronal networks. I. Theory.
          J. Neurophysiology 83, pp. 808–827.
  """

  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=-30., V_c=-50.0, c=.07,
               R=1., tau=10., tau_ref=0., method='exp_auto', name=None):
    # initialization
    super(QuaIF, self).__init__(size=size, name=name)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_c = V_c
    self.c = c
    self.R = R
    self.tau = tau
    self.tau_ref = tau_ref

    # variables
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))
    # variables
    self.V = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

    # integral
    self.integral = odeint(method=method, f=self.derivative)

  def derivative(self, V, t):
    dVdt = (self.c * (V - self.V_rest) * (V - self.V_c) + self.R * self.input) / self.tau
    return dVdt

  def update(self, _t, _dt, **kwargs):
    refractory = (_t - self.t_last_spike) <= self.tau_ref
    V = self.integral(self.V, _t, dt=_dt)
    V = bm.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
    self.V.value = bm.where(spike, self.V_reset, V)
    self.refractory.value = bm.logical_or(refractory, spike)
    self.spike.value = spike
    self.input[:] = 0.


class AdQuaIF(NeuGroup):
  r"""Adaptive quadratic integrate-and-fire neuron model.

  **Model Descriptions**

  The adaptive quadratic integrate-and-fire neuron model [1]_ is given by:

  .. math::

      \begin{aligned}
      \tau_m \frac{d V}{d t}&=c(V-V_{rest})(V-V_c) - w + I(t), \\
      \tau_w \frac{d w}{d t}&=a(V-V_{rest}) - w,
      \end{aligned}

  once the membrane potential reaches the spike threshold,

  .. math::

      V \rightarrow V_{reset}, \\
      w \rightarrow w+b.

  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> group = bp.neurons.AdQuaIF(1, )
    >>> runner = bp.DSRunner(group, monitors=['V', 'w'], inputs=('input', 30.))
    >>> runner.run(300)
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
    >>> fig.add_subplot(gs[0, 0])
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='V')
    >>> fig.add_subplot(gs[1, 0])
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.w, ylabel='w', show=True)

  **Model Parameters**

  ============= ============== ======== =======================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -------------------------------------------------------
  V_rest         -65            mV       Resting potential.
  V_reset        -68            mV       Reset potential after spike.
  V_th           -30            mV       Threshold potential of spike and reset.
  V_c            -50            mV       Critical voltage for spike initiation. Must be larger
                                         than :math:`V_{rest}`.
  a               1              \       The sensitivity of the recovery variable :math:`u` to
                                         the sub-threshold fluctuations of the membrane
                                         potential :math:`v`
  b              .1             \        The increment of :math:`w` produced by a spike.
  c              .07             \       Coefficient describes membrane potential update.
                                         Larger than 0.
  tau            10             ms       Membrane time constant.
  tau_w          10             ms       Time constant of the adaptation current.
  ============= ============== ======== =======================================================

  **Model Variables**

  ================== ================= ==========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ----------------------------------------------------------
  V                   0                 Membrane potential.
  w                   0                 Adaptation current.
  input               0                 External and synaptic input current.
  spike               False             Flag to mark whether the neuron is spiking.
  t_last_spike        -1e7              Last spike time stamp.
  ================== ================= ==========================================================

  **References**

  .. [1] Izhikevich, E. M. (2004). Which model to use for cortical spiking
         neurons?. IEEE transactions on neural networks, 15(5), 1063-1070.
  .. [2] Touboul, Jonathan. "Bifurcation analysis of a general class of
         nonlinear integrate-and-fire neurons." SIAM Journal on Applied
         Mathematics 68, no. 4 (2008): 1045-1079.
  """

  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=-30., V_c=-50.0, a=1., b=.1,
               c=.07, tau=10., tau_w=10., method='exp_auto', name=None):
    super(AdQuaIF, self).__init__(size=size, name=name)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_c = V_c
    self.c = c
    self.a = a
    self.b = b
    self.tau = tau
    self.tau_w = tau_w

    # variables
    self.V = bm.Variable(bm.zeros(self.num))
    self.w = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))

    # integral
    self.integral = odeint(method=method, f=self.derivative)

  def dV(self, V, t, w):
    dVdt = (self.c * (V - self.V_rest) * (V - self.V_c) - w + self.input) / self.tau
    return dVdt

  def dw(self, w, t, V):
    dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
    return dwdt

  @property
  def derivative(self):
    return JointEq([self.dV, self.dw])

  def update(self, _t, _dt):
    V, w = self.integral(self.V, self.w, _t, dt=_dt)
    spike = self.V_th <= V
    self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
    self.V.value = bm.where(spike, self.V_reset, V)
    self.w.value = bm.where(spike, w + self.b, w)
    self.spike.value = spike
    self.input[:] = 0.


class GIF(NeuGroup):
  r"""Generalized Integrate-and-Fire model.

  **Model Descriptions**

  The generalized integrate-and-fire model [1]_ is given by

  .. math::

      &\frac{d I_j}{d t} = - k_j I_j

      &\frac{d V}{d t} = ( - (V - V_{rest}) + R\sum_{j}I_j + RI) / \tau

      &\frac{d V_{th}}{d t} = a(V - V_{rest}) - b(V_{th} - V_{th\infty})

  When :math:`V` meet :math:`V_{th}`, Generalized IF neuron fires:

  .. math::

      &I_j \leftarrow R_j I_j + A_j

      &V \leftarrow V_{reset}

      &V_{th} \leftarrow max(V_{th_{reset}}, V_{th})

  Note that :math:`I_j` refers to arbitrary number of internal currents.

  **Model Examples**

  - `Detailed examples to reproduce different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Niebur_2009_GIF.html>`_

  **Model Parameters**

  ============= ============== ======== ====================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- --------------------------------------------------------------------
  V_rest        -70            mV       Resting potential.
  V_reset       -70            mV       Reset potential after spike.
  V_th_inf      -50            mV       Target value of threshold potential :math:`V_{th}` updating.
  V_th_reset    -60            mV       Free parameter, should be larger than :math:`V_{reset}`.
  R             20             \        Membrane resistance.
  tau           20             ms       Membrane time constant. Compute by :math:`R * C`.
  a             0              \        Coefficient describes the dependence of
                                        :math:`V_{th}` on membrane potential.
  b             0.01           \        Coefficient describes :math:`V_{th}` update.
  k1            0.2            \        Constant pf :math:`I1`.
  k2            0.02           \        Constant of :math:`I2`.
  R1            0              \        Free parameter.
                                        Describes dependence of :math:`I_1` reset value on
                                        :math:`I_1` value before spiking.
  R2            1              \        Free parameter.
                                        Describes dependence of :math:`I_2` reset value on
                                        :math:`I_2` value before spiking.
  A1            0              \        Free parameter.
  A2            0              \        Free parameter.
  ============= ============== ======== ====================================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                  -70               Membrane potential.
  input              0                 External and synaptic input current.
  spike              False             Flag to mark whether the neuron is spiking.
  V_th               -50               Spiking threshold potential.
  I1                 0                 Internal current 1.
  I2                 0                 Internal current 2.
  t_last_spike       -1e7              Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] Mihalaş, Ştefan, and Ernst Niebur. "A generalized linear
         integrate-and-fire neural model produces diverse spiking
         behaviors." Neural computation 21.3 (2009): 704-718.
  .. [2] Teeter, Corinne, Ramakrishnan Iyer, Vilas Menon, Nathan
         Gouwens, David Feng, Jim Berg, Aaron Szafer et al. "Generalized
         leaky integrate-and-fire models classify multiple neuron types."
         Nature communications 9, no. 1 (2018): 1-15.
  """

  def __init__(self, size, V_rest=-70., V_reset=-70., V_th_inf=-50., V_th_reset=-60.,
               R=20., tau=20., a=0., b=0.01, k1=0.2, k2=0.02, R1=0., R2=1., A1=0.,
               A2=0., method='exp_auto', name=None):
    # initialization
    super(GIF, self).__init__(size=size, name=name)

    # params
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th_inf = V_th_inf
    self.V_th_reset = V_th_reset
    self.R = R
    self.tau = tau
    self.a = a
    self.b = b
    self.k1 = k1
    self.k2 = k2
    self.R1 = R1
    self.R2 = R2
    self.A1 = A1
    self.A2 = A2

    # variables
    self.I1 = bm.Variable(bm.zeros(self.num))
    self.I2 = bm.Variable(bm.zeros(self.num))
    self.V_th = bm.Variable(bm.ones(self.num) * -50.)
    self.V = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

    # integral
    self.integral = odeint(method=method, f=self.derivative)

  def dI1(self, I1, t):
    return - self.k1 * I1

  def dI2(self, I2, t):
    return - self.k2 * I2

  def dVth(self, V_th, t, V):
    return self.a * (V - self.V_rest) - self.b * (V_th - self.V_th_inf)

  def dV(self, V, t, I1, I2):
    return (- (V - self.V_rest) + self.R * self.input + self.R * I1 + self.R * I2) / self.tau

  @property
  def derivative(self):
    return JointEq([self.dI1, self.dI2, self.dVth, self.dV])

  def update(self, _t, _dt):
    I1, I2, V_th, V = self.integral(self.I1, self.I2, self.V_th, self.V, _t, dt=_dt)
    spike = self.V_th <= V
    V = bm.where(spike, self.V_reset, V)
    I1 = bm.where(spike, self.R1 * I1 + self.A1, I1)
    I2 = bm.where(spike, self.R2 * I2 + self.A2, I2)
    reset_th = bm.logical_and(V_th < self.V_th_reset, spike)
    V_th = bm.where(reset_th, self.V_th_reset, V_th)
    self.spike.value = spike
    self.I1.value = I1
    self.I2.value = I2
    self.V_th.value = V_th
    self.V.value = V
    self.input[:] = 0.
