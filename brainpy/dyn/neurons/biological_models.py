# -*- coding: utf-8 -*-

from typing import Union, Callable, Optional

import brainpy.math as bm
from brainpy.dyn.base import NeuGroup
from brainpy.initialize import OneInit, Uniform, Initializer, parameter, noise as init_noise, variable_
from brainpy.integrators.joint_eq import JointEq
from brainpy.integrators.ode import odeint
from brainpy.integrators.sde import sdeint
from brainpy.modes import Mode, BatchingMode, NormalMode, normal, check_mode
from brainpy.tools.checking import check_initializer
from brainpy.types import Shape, Array

__all__ = [
  'HH',
  'MorrisLecar',
  'PinskyRinzelModel',
  'WangBuzsakiModel',
]


class HH(NeuGroup):
  r"""Hodgkin–Huxley neuron model.

  **Model Descriptions**

  The Hodgkin-Huxley (HH; Hodgkin & Huxley, 1952) model [1]_ for the generation of
  the nerve action potential is one of the most successful mathematical models of
  a complex biological process that has ever been formulated. The basic concepts
  expressed in the model have proved a valid approach to the study of bio-electrical
  activity from the most primitive single-celled organisms such as *Paramecium*,
  right through to the neurons within our own brains.

  Mathematically, the model is given by,

  .. math::

      C \frac {dV} {dt} = -(\bar{g}_{Na} m^3 h (V &-E_{Na})
      + \bar{g}_K n^4 (V-E_K) + g_{leak} (V - E_{leak})) + I(t)

      \frac {dx} {dt} &= \alpha_x (1-x)  - \beta_x, \quad x\in {\rm{\{m, h, n\}}}

      &\alpha_m(V) = \frac {0.1(V+40)}{1-\exp(\frac{-(V + 40)} {10})}

      &\beta_m(V) = 4.0 \exp(\frac{-(V + 65)} {18})

      &\alpha_h(V) = 0.07 \exp(\frac{-(V+65)}{20})

      &\beta_h(V) = \frac 1 {1 + \exp(\frac{-(V + 35)} {10})}

      &\alpha_n(V) = \frac {0.01(V+55)}{1-\exp(-(V+55)/10)}

      &\beta_n(V) = 0.125 \exp(\frac{-(V + 65)} {80})

  The illustrated example of HH neuron model please see `this notebook <../neurons/HH_model.ipynb>`_.

  The Hodgkin–Huxley model can be thought of as a differential equation system with
  four state variables, :math:`V_{m}(t),n(t),m(t)`, and :math:`h(t)`, that change
  with respect to time :math:`t`. The system is difficult to study because it is a
  nonlinear system and cannot be solved analytically. However, there are many numeric
  methods available to analyze the system. Certain properties and general behaviors,
  such as limit cycles, can be proven to exist.

  *1. Center manifold*

  Because there are four state variables, visualizing the path in phase space can
  be difficult. Usually two variables are chosen, voltage :math:`V_{m}(t)` and the
  potassium gating variable :math:`n(t)`, allowing one to visualize the limit cycle.
  However, one must be careful because this is an ad-hoc method of visualizing the
  4-dimensional system. This does not prove the existence of the limit cycle.

  .. image:: ../../../../_static/Hodgkin_Huxley_Limit_Cycle.png
      :align: center

  A better projection can be constructed from a careful analysis of the Jacobian of
  the system, evaluated at the equilibrium point. Specifically, the eigenvalues of
  the Jacobian are indicative of the center manifold's existence. Likewise, the
  eigenvectors of the Jacobian reveal the center manifold's orientation. The
  Hodgkin–Huxley model has two negative eigenvalues and two complex eigenvalues
  with slightly positive real parts. The eigenvectors associated with the two
  negative eigenvalues will reduce to zero as time :math:`t` increases. The remaining
  two complex eigenvectors define the center manifold. In other words, the
  4-dimensional system collapses onto a 2-dimensional plane. Any solution
  starting off the center manifold will decay towards the *center manifold*.
  Furthermore, the limit cycle is contained on the center manifold.

  *2. Bifurcations*

  If the injected current :math:`I` were used as a bifurcation parameter, then the
  Hodgkin–Huxley model undergoes a Hopf bifurcation. As with most neuronal models,
  increasing the injected current will increase the firing rate of the neuron.
  One consequence of the Hopf bifurcation is that there is a minimum firing rate.
  This means that either the neuron is not firing at all (corresponding to zero
  frequency), or firing at the minimum firing rate. Because of the all-or-none
  principle, there is no smooth increase in action potential amplitude, but
  rather there is a sudden "jump" in amplitude. The resulting transition is
  known as a `canard <http://www.scholarpedia.org/article/Canards>`_.

  .. image:: ../../../../_static/Hodgkins_Huxley_bifurcation_by_I.gif
     :align: center

  The following image shows the bifurcation diagram of the Hodgkin–Huxley model
  as a function of the external drive :math:`I` [3]_. The green lines show the amplitude
  of a stable limit cycle and the blue lines indicate unstable limit-cycle behaviour,
  both born from Hopf bifurcations. The solid red line shows the stable fixed point
  and the black line shows the unstable fixed point.

  .. image:: ../../../../_static/Hodgkin_Huxley_bifurcation.png
     :align: center

  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> group = bp.dyn.HH(2)
    >>> runner = bp.dyn.DSRunner(group, monitors=['V'], inputs=('input', 10.))
    >>> runner.run(200.)
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.V, show=True)

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import matplotlib.pyplot as plt
    >>>
    >>> group = bp.dyn.HH(2)
    >>>
    >>> I1 = bp.inputs.spike_input(sp_times=[500., 550., 1000, 1030, 1060, 1100, 1200], sp_lens=5, sp_sizes=5., duration=2000, )
    >>> I2 = bp.inputs.spike_input(sp_times=[600.,       900, 950, 1500], sp_lens=5, sp_sizes=5., duration=2000, )
    >>> I1 += bp.math.random.normal(0, 3, size=I1.shape)
    >>> I2 += bp.math.random.normal(0, 3, size=I2.shape)
    >>> I = bp.math.stack((I1, I2), axis=-1)
    >>>
    >>> runner = bp.dyn.DSRunner(group, monitors=['V'], inputs=('input', I, 'iter'))
    >>> runner.run(2000.)
    >>>
    >>> fig, gs = bp.visualize.get_figure(1, 1, 3, 8)
    >>> fig.add_subplot(gs[0, 0])
    >>> plt.plot(runner.mon.ts, runner.mon.V[:, 0])
    >>> plt.plot(runner.mon.ts, runner.mon.V[:, 1] + 130)
    >>> plt.xlim(10, 2000)
    >>> plt.xticks([])
    >>> plt.yticks([])
    >>> plt.show()

  Parameters
  ----------
  size: sequence of int, int
    The size of the neuron group.
  ENa: float, JaxArray, ndarray, Initializer, callable
    The reversal potential of sodium. Default is 50 mV.
  gNa: float, JaxArray, ndarray, Initializer, callable
    The maximum conductance of sodium channel. Default is 120 msiemens.
  EK: float, JaxArray, ndarray, Initializer, callable
    The reversal potential of potassium. Default is -77 mV.
  gK: float, JaxArray, ndarray, Initializer, callable
    The maximum conductance of potassium channel. Default is 36 msiemens.
  EL: float, JaxArray, ndarray, Initializer, callable
    The reversal potential of learky channel. Default is -54.387 mV.
  gL: float, JaxArray, ndarray, Initializer, callable
    The conductance of learky channel. Default is 0.03 msiemens.
  V_th: float, JaxArray, ndarray, Initializer, callable
    The threshold of the membrane spike. Default is 20 mV.
  C: float, JaxArray, ndarray, Initializer, callable
    The membrane capacitance. Default is 1 ufarad.
  V_initializer: JaxArray, ndarray, Initializer, callable
    The initializer of membrane potential.
  m_initializer: JaxArray, ndarray, Initializer, callable
    The initializer of m channel.
  h_initializer: JaxArray, ndarray, Initializer, callable
    The initializer of h channel.
  n_initializer: JaxArray, ndarray, Initializer, callable
    The initializer of n channel.
  method: str
    The numerical integration method.
  name: str
    The group name.

  References
  ----------

  .. [1] Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description
         of membrane current and its application to conduction and excitation
         in nerve." The Journal of physiology 117.4 (1952): 500.
  .. [2] https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model
  .. [3] Ashwin, Peter, Stephen Coombes, and Rachel Nicks. "Mathematical
         frameworks for oscillatory network dynamics in neuroscience."
         The Journal of Mathematical Neuroscience 6, no. 1 (2016): 1-92.
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      ENa: Union[float, Array, Initializer, Callable] = 50.,
      gNa: Union[float, Array, Initializer, Callable] = 120.,
      EK: Union[float, Array, Initializer, Callable] = -77.,
      gK: Union[float, Array, Initializer, Callable] = 36.,
      EL: Union[float, Array, Initializer, Callable] = -54.387,
      gL: Union[float, Array, Initializer, Callable] = 0.03,
      V_th: Union[float, Array, Initializer, Callable] = 20.,
      C: Union[float, Array, Initializer, Callable] = 1.0,
      V_initializer: Union[Initializer, Callable, Array] = Uniform(-70, -60.),
      m_initializer: Optional[Union[Initializer, Callable, Array]] = None,
      h_initializer: Optional[Union[Initializer, Callable, Array]] = None,
      n_initializer: Optional[Union[Initializer, Callable, Array]] = None,
      noise: Union[float, Array, Initializer, Callable] = None,
      method: str = 'exp_auto',
      name: str = None,

      # training parameter
      mode: Mode = normal,
  ):
    # initialization
    super(HH, self).__init__(size=size,
                             keep_size=keep_size,
                             name=name,
                             mode=mode)
    check_mode(self.mode, (BatchingMode, NormalMode), self.__class__.__name__)

    # parameters
    self.ENa = parameter(ENa, self.varshape, allow_none=False)
    self.EK = parameter(EK, self.varshape, allow_none=False)
    self.EL = parameter(EL, self.varshape, allow_none=False)
    self.gNa = parameter(gNa, self.varshape, allow_none=False)
    self.gK = parameter(gK, self.varshape, allow_none=False)
    self.gL = parameter(gL, self.varshape, allow_none=False)
    self.C = parameter(C, self.varshape, allow_none=False)
    self.V_th = parameter(V_th, self.varshape, allow_none=False)
    self.noise = init_noise(noise, self.varshape, num_vars=4)

    # initializers
    check_initializer(m_initializer, 'm_initializer', allow_none=True)
    check_initializer(h_initializer, 'h_initializer', allow_none=True)
    check_initializer(n_initializer, 'n_initializer', allow_none=True)
    check_initializer(V_initializer, 'V_initializer', allow_none=False)
    self._m_initializer = m_initializer
    self._h_initializer = h_initializer
    self._n_initializer = n_initializer
    self._V_initializer = V_initializer

    # variables
    self.V = variable_(self._V_initializer, self.varshape, mode)
    self.m = (bm.Variable(self.m_inf(self.V.value))
              if m_initializer is None else
              variable_(self._m_initializer, self.varshape, mode))
    self.h = (bm.Variable(self.h_inf(self.V.value))
              if h_initializer is None else
              variable_(self._h_initializer, self.varshape, mode))
    self.n = (bm.Variable(self.n_inf(self.V.value))
              if n_initializer is None else
              variable_(self._n_initializer, self.varshape, mode))
    self.spike = variable_(lambda s: bm.zeros(s, dtype=bool), self.varshape, mode)
    self.input = variable_(bm.zeros, self.varshape, mode)

    # integral
    if self.noise is None:
      self.integral = odeint(method=method, f=self.derivative)
    else:
      self.integral = sdeint(method=method, f=self.derivative, g=self.noise)

  # m channel
  m_alpha = lambda self, V: 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
  m_beta = lambda self, V: 4.0 * bm.exp(-(V + 65) / 18)
  m_inf = lambda self, V: self.m_alpha(V) / (self.m_alpha(V) + self.m_beta(V))
  dm = lambda self, m, t, V: self.m_alpha(V) * (1 - m) - self.m_beta(V) * m

  # h channel
  h_alpha = lambda self, V: 0.07 * bm.exp(-(V + 65) / 20.)
  h_beta = lambda self, V: 1 / (1 + bm.exp(-(V + 35) / 10))
  h_inf = lambda self, V: self.h_alpha(V) / (self.h_alpha(V) + self.h_beta(V))
  dh = lambda self, h, t, V: self.h_alpha(V) * (1 - h) - self.h_beta(V) * h

  # n channel
  n_alpha = lambda self, V: 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
  n_beta = lambda self, V: 0.125 * bm.exp(-(V + 65) / 80)
  n_inf = lambda self, V: self.n_alpha(V) / (self.n_alpha(V) + self.n_beta(V))
  dn = lambda self, n, t, V: self.n_alpha(V) * (1 - n) - self.n_beta(V) * n

  def reset_state(self, batch_size=None):
    self.V.value = variable_(self._V_initializer, self.varshape, batch_size)
    if self._m_initializer is None:
      self.m.value = self.m_inf(self.V.value)
    else:
      self.m.value = variable_(self._m_initializer, self.varshape, batch_size)
    if self._h_initializer is None:
      self.h.value = self.h_inf(self.V.value)
    else:
      self.h.value = variable_(self._h_initializer, self.varshape, batch_size)
    if self._n_initializer is None:
      self.n.value = self.n_inf(self.V.value)
    else:
      self.n.value = variable_(self._n_initializer, self.varshape, batch_size)
    self.input.value = variable_(bm.zeros, self.varshape, batch_size)
    self.spike.value = variable_(lambda s: bm.zeros(s, dtype=bool), self.varshape, batch_size)

  def dV(self, V, t, m, h, n, I_ext):
    I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
    I_K = (self.gK * n ** 4.0) * (V - self.EK)
    I_leak = self.gL * (V - self.EL)
    dVdt = (- I_Na - I_K - I_leak + I_ext) / self.C
    return dVdt

  @property
  def derivative(self):
    return JointEq(self.dV, self.dm, self.dh, self.dn)

  def update(self, tdi, x=None):
    t, dt = tdi['t'], tdi['dt']
    if x is not None: self.input += x
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, t, self.input, dt)
    self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    self.V.value = V
    self.m.value = m
    self.h.value = h
    self.n.value = n

  def clear_input(self):
    self.input[:] = 0.


class MorrisLecar(NeuGroup):
  r"""The Morris-Lecar neuron model.

  **Model Descriptions**

  The Morris-Lecar model [4]_ (Also known as :math:`I_{Ca}+I_K`-model)
  is a two-dimensional "reduced" excitation model applicable to
  systems having two non-inactivating voltage-sensitive conductances.
  This model was named after Cathy Morris and Harold Lecar, who
  derived it in 1981. Because it is two-dimensional, the Morris-Lecar
  model is one of the favorite conductance-based models in computational neuroscience.

  The original form of the model employed an instantaneously
  responding voltage-sensitive Ca2+ conductance for excitation and a delayed
  voltage-dependent K+ conductance for recovery. The equations of the model are:

  .. math::

      \begin{aligned}
      C\frac{dV}{dt} =& -  g_{Ca} M_{\infty} (V - V_{Ca})- g_{K} W(V - V_{K}) -
                        g_{Leak} (V - V_{Leak}) + I_{ext} \\
      \frac{dW}{dt} =& \frac{W_{\infty}(V) - W}{ \tau_W(V)}
      \end{aligned}

  Here, :math:`V` is the membrane potential, :math:`W` is the "recovery variable",
  which is almost invariably the normalized :math:`K^+`-ion conductance, and
  :math:`I_{ext}` is the applied current stimulus.

  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>>
    >>> group = bp.dyn.MorrisLecar(1)
    >>> runner = bp.dyn.DSRunner(group, monitors=['V', 'W'], inputs=('input', 100.))
    >>> runner.run(1000)
    >>>
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
    >>> fig.add_subplot(gs[0, 0])
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.W, ylabel='W')
    >>> fig.add_subplot(gs[1, 0])
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='V', show=True)


  **Model Parameters**

  ============= ============== ======== =======================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -------------------------------------------------------
  V_Ca          130            mV       Equilibrium potentials of Ca+.(mV)
  g_Ca          4.4            \        Maximum conductance of corresponding Ca+.(mS/cm2)
  V_K           -84            mV       Equilibrium potentials of K+.(mV)
  g_K           8              \        Maximum conductance of corresponding K+.(mS/cm2)
  V_Leak        -60            mV       Equilibrium potentials of leak current.(mV)
  g_Leak        2              \        Maximum conductance of leak current.(mS/cm2)
  C             20             \        Membrane capacitance.(uF/cm2)
  V1            -1.2           \        Potential at which M_inf = 0.5.(mV)
  V2            18             \        Reciprocal of slope of voltage dependence of M_inf.(mV)
  V3            2              \        Potential at which W_inf = 0.5.(mV)
  V4            30             \        Reciprocal of slope of voltage dependence of W_inf.(mV)
  phi           0.04           \        A temperature factor. (1/s)
  V_th          10             mV       The spike threshold.
  ============= ============== ======== =======================================================

  References
  ----------

  .. [4] Lecar, Harold. "Morris-lecar model." Scholarpedia 2.10 (2007): 1333.
  .. [5] http://www.scholarpedia.org/article/Morris-Lecar_model
  .. [6] https://en.wikipedia.org/wiki/Morris%E2%80%93Lecar_model
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      V_Ca: Union[float, Array, Initializer, Callable] = 130.,
      g_Ca: Union[float, Array, Initializer, Callable] = 4.4,
      V_K: Union[float, Array, Initializer, Callable] = -84.,
      g_K: Union[float, Array, Initializer, Callable] = 8.,
      V_leak: Union[float, Array, Initializer, Callable] = -60.,
      g_leak: Union[float, Array, Initializer, Callable] = 2.,
      C: Union[float, Array, Initializer, Callable] = 20.,
      V1: Union[float, Array, Initializer, Callable] = -1.2,
      V2: Union[float, Array, Initializer, Callable] = 18.,
      V3: Union[float, Array, Initializer, Callable] = 2.,
      V4: Union[float, Array, Initializer, Callable] = 30.,
      phi: Union[float, Array, Initializer, Callable] = 0.04,
      V_th: Union[float, Array, Initializer, Callable] = 10.,
      W_initializer: Union[Callable, Initializer, Array] = OneInit(0.02),
      V_initializer: Union[Callable, Initializer, Array] = Uniform(-70., -60.),
      noise: Union[float, Array, Initializer, Callable] = None,
      method: str = 'exp_auto',
      name: str = None,

      # training parameter
      mode: Mode = normal,
  ):
    # initialization
    super(MorrisLecar, self).__init__(size=size,
                                      keep_size=keep_size,
                                      name=name,
                                      mode=mode)
    check_mode(self.mode, (BatchingMode, NormalMode), self.__class__)

    # params
    self.V_Ca = parameter(V_Ca, self.varshape, allow_none=False)
    self.g_Ca = parameter(g_Ca, self.varshape, allow_none=False)
    self.V_K = parameter(V_K, self.varshape, allow_none=False)
    self.g_K = parameter(g_K, self.varshape, allow_none=False)
    self.V_leak = parameter(V_leak, self.varshape, allow_none=False)
    self.g_leak = parameter(g_leak, self.varshape, allow_none=False)
    self.C = parameter(C, self.varshape, allow_none=False)
    self.V1 = parameter(V1, self.varshape, allow_none=False)
    self.V2 = parameter(V2, self.varshape, allow_none=False)
    self.V3 = parameter(V3, self.varshape, allow_none=False)
    self.V4 = parameter(V4, self.varshape, allow_none=False)
    self.phi = parameter(phi, self.varshape, allow_none=False)
    self.V_th = parameter(V_th, self.varshape, allow_none=False)
    self.noise = init_noise(noise, self.varshape, num_vars=2)

    # initializers
    check_initializer(V_initializer, 'V_initializer', allow_none=False)
    check_initializer(W_initializer, 'W_initializer', allow_none=False)
    self._W_initializer = W_initializer
    self._V_initializer = V_initializer

    # variables
    self.W = variable_(self._W_initializer, self.varshape, mode)
    self.V = variable_(self._V_initializer, self.varshape, mode)
    self.input = variable_(bm.zeros, self.varshape, mode)
    self.spike = variable_(lambda s: bm.zeros(s, dtype=bool), self.varshape, mode)

    # integral
    if self.noise is None:
      self.integral = odeint(method=method, f=self.derivative)
    else:
      self.integral = sdeint(method=method, f=self.derivative, g=self.noise)

  def reset_state(self, batch_size=None):
    self.W.value = variable_(self._W_initializer, self.varshape, batch_size)
    self.V.value = variable_(self._V_initializer, self.varshape, batch_size)
    self.input.value = variable_(bm.zeros, self.varshape, batch_size)
    self.spike.value = variable_(lambda s: bm.zeros(s, dtype=bool), self.varshape, batch_size)

  def dV(self, V, t, W, I_ext):
    M_inf = (1 / 2) * (1 + bm.tanh((V - self.V1) / self.V2))
    I_Ca = self.g_Ca * M_inf * (V - self.V_Ca)
    I_K = self.g_K * W * (V - self.V_K)
    I_Leak = self.g_leak * (V - self.V_leak)
    dVdt = (- I_Ca - I_K - I_Leak + I_ext) / self.C
    return dVdt

  def dW(self, W, t, V):
    tau_W = 1 / (self.phi * bm.cosh((V - self.V3) / (2 * self.V4)))
    W_inf = (1 / 2) * (1 + bm.tanh((V - self.V3) / self.V4))
    dWdt = (W_inf - W) / tau_W
    return dWdt

  @property
  def derivative(self):
    return JointEq([self.dV, self.dW])

  def update(self, tdi, x=None):
    t, dt = tdi['t'], tdi['dt']
    if x is not None: self.input += x
    V, self.W.value = self.integral(self.V, self.W, t, self.input, dt)
    spike = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    self.V.value = V
    self.spike.value = spike

  def clear_input(self):
    self.input[:] = 0.


class PinskyRinzelModel(NeuGroup):
  r"""The Pinsky and Rinsel (1994) model.

  The Pinsky and Rinsel (1994) model [7]_ is a 2-compartment (soma and dendrite),
  conductance-based (Hodgin-Huxley type) model of a hippocampal CA3 pyramidal
  neuron. It is a reduced version of an earlier, 19-compartment model by
  Traub, et. al. (1991) [8]_. This model demonstrates how similar qualitative
  and quantitative spiking behaviors can be obtained despite the reduction
  in model complexity.

  Specifically, this model demonstrates calcium bursting behavior and how
  the 'ping-pong' interplay between somatic and dendritic currents results
  in a complex shape of the burst.

  .. image:: ../../../../_static/Pinsky-Rinzel-model-illustration.png
      :align: center

  Mathematically, the model is given by:

  .. math::

     \begin{aligned}
    &\mathrm{C}_{\mathrm{m}} \mathrm{V}_{\mathrm{s}}^{\prime}=-\mathrm{I}_{\mathrm{Leak}}-\mathrm{I}_{\mathrm{Na}}-\mathrm{I}_{\mathrm{K}_{\mathrm{DR}}}-\frac{\mathrm{I}_{\mathrm{DS}}}{\mathrm{p}}+\frac{\mathrm{I}_{\mathrm{S}_{\mathrm{app}}}}{\mathrm{p}} \\
    &\mathrm{C}_{\mathrm{m}} \mathrm{V}_{\mathrm{d}}^{\prime}=-\mathrm{I}_{\mathrm{Leak}}-\mathrm{I}_{\mathrm{Ca}}-\mathrm{I}_{\mathrm{K}_{\mathrm{Ca}}}-\mathrm{I}_{\mathrm{K}_{\mathrm{AHP}}}+\frac{\mathrm{I}_{\mathrm{SD}}}{(1-\mathrm{p})}+\frac{\mathrm{I}_{\mathrm{D}_{\mathrm{app}}}}{(1-\mathrm{p})} \\
    &\frac{\mathrm{dCa}}{\mathrm{dt}}=-0.13 \mathrm{I}_{\mathrm{Ca}}-0.075 \mathrm{Ca}
    \end{aligned}

  The currents of the model are functions of potentials as follows:

  .. math::

     \begin{aligned}
      \mathrm{I}_{\mathrm{Na}} &=\mathrm{g}_{\mathrm{Na}} m_{\infty}^{2}\left(\mathrm{~V}_{\mathrm{s}}\right) h\left(\mathrm{~V}_{\mathrm{s}}-\mathrm{V}_{\mathrm{Na}}\right) \\
      \mathrm{I}_{\mathrm{K}_{\mathrm{DR}}} &=\mathrm{g}_{\mathrm{K}_{\mathrm{DR}}} n\left(\mathrm{~V}_{\mathrm{s}}-\mathrm{V}_{\mathrm{K}}\right) \\
      \mathrm{I}_{\mathrm{Ca}} &=\mathrm{g}_{\mathrm{Ca}}{ }^{2}\left(\mathrm{~V}_{\mathrm{d}}-\mathrm{V}_{\mathrm{N}}\right) \\
      \mathrm{I}_{\mathrm{K}_{\mathrm{Ca}}} &=\mathrm{g}_{\mathrm{k}_{\mathrm{Ca}}} C \chi(\mathrm{Ca})\left(\mathrm{V}_{\mathrm{d}}-\mathrm{V}_{\mathrm{Ca}}\right) \\
      \mathrm{I}_{\mathrm{K}_{\mathrm{AHP}}} &=\mathrm{g}_{\mathrm{K}_{\mathrm{AHP}}} q\left(\mathrm{~V}_{\mathrm{d}}-\mathrm{V}_{\mathrm{K}}\right) \\
      \mathrm{I}_{\mathrm{SD}} &=-\mathrm{I}_{\mathrm{DS}}=\mathrm{g}_{\mathrm{c}}\left(\mathrm{V}_{\mathrm{d}}-\mathrm{V}_{\mathrm{s}}\right) \\
      \mathrm{I}_{\mathrm{Leak}} &=\mathrm{g}_{\mathrm{L}}\left(\mathrm{V}-\mathrm{V}_{\mathrm{L}}\right)
      \end{aligned}

  The activation and inactivation variables should satisfy these equations

  .. math::

     \begin{aligned}
    \omega^{\prime}(\mathrm{V}) &=\frac{\omega_{\infty}(\mathrm{V})-\omega}{\tau_{\omega}(\mathrm{V})} \\
    \omega_{\infty}(\mathrm{V}) &=\frac{\alpha_{\omega}(\mathrm{V})}{\alpha_{\omega}(\mathrm{V})+\beta_{\omega}(\mathrm{V})} \\
    \tau_{\omega}(\mathrm{V}) &=\frac{1}{\alpha_{\omega}(\mathrm{V})+\beta_{\omega}(\mathrm{V})}
    \end{aligned}

  where, independently, we consider :math:`\omega = h, n, s, m, c, q`.

  The rate functions are defined as follows

  .. math::

     \begin{aligned}
    \alpha_{m}\left(\mathrm{~V}_{\mathrm{s}}\right) &=\frac{0.32\left(-46.9-\mathrm{V}_{\mathrm{s}}\right)}{\exp \left(\frac{-46.9-\mathrm{V}_{\mathrm{s}}}{4}\right)-1} \\
    \beta_{m}\left(\mathrm{~V}_{\mathrm{s}}\right) &=\frac{0.28\left(\mathrm{~V}_{\mathrm{s}}+19.9\right)}{\exp \left(\frac{\mathrm{V}_{\mathrm{s}}+19.9}{5}\right)-1}, \\
    \alpha_{n}\left(\mathrm{~V}_{\mathrm{s}}\right) &=\frac{0.016\left(-24.9-\mathrm{V}_{\mathrm{s}}\right)}{\exp \left(\frac{-24.9-\mathrm{V}_{\mathrm{s}}}{5}\right)-1} \\
    \beta_{n}\left(\mathrm{~V}_{\mathrm{s}}\right) &=0.25 \exp \left(-1-0.025 \mathrm{~V}_{\mathrm{s}}\right) \\
    \alpha_{h}\left(\mathrm{~V}_{\mathrm{s}}\right) &=0.128 \exp \left(\frac{-43-\mathrm{V}_{\mathrm{s}}}{18}\right) \\
    \beta_{h}\left(\mathrm{~V}_{\mathrm{s}}\right) &=\frac{4}{1+\exp \left(\frac{\left(-20-\mathrm{V}_{\mathrm{s}}\right.}{5}\right)}, \\
    \alpha_{s}\left(\mathrm{~V}_{\mathrm{d}}\right) &=\frac{1.6}{1+\exp \left(-0.072\left(\mathrm{~V}_{\mathrm{d}}-5\right)\right)} \\
    \beta_{s}\left(\mathrm{~V}_{\mathrm{d}}\right) &=\frac{0.02\left(\mathrm{~V}_{\mathrm{d}}+8.9\right)}{\exp \left(\frac{\left(\mathrm{V}_{\mathrm{d}}+8.9\right)}{5}\right)-1}, \\
    \alpha_{C}\left(\mathrm{~V}_{\mathrm{d}}\right) &=\frac{\left(1-H\left(\mathrm{~V}_{\mathrm{d}}+10\right)\right) \exp \left(\frac{\left(\mathrm{V}_{\mathrm{d}}+50\right)}{11}-\frac{\left(\mathrm{V}_{\mathrm{d}}+53.5\right)}{27}\right)}{18.975}+H\left(\mathrm{~V}_{\mathrm{d}}+10\right)\left(2 \exp \left(\frac{\left(-53.5-\mathrm{V}_{\mathrm{d}}\right.}{27}\right)\right) \\
    \beta_{C}\left(\mathrm{~V}_{\mathrm{d}}\right) &=\left(1-H\left(\mathrm{~V}_{\mathrm{d}}+10\right)\right)\left(2 \exp \left(\frac{\left(-53.5-\mathrm{V}_{\mathrm{d}}\right)}{27}\right)-\alpha_{c}\left(\mathrm{~V}_{\mathrm{d}}\right)\right) \\
    \alpha_{q}(\mathrm{Ca}) &=\min (0.00002 \mathrm{Ca}, 0.01) \\
    \beta_{q}(\mathrm{Ca}) &=0.001 \\
    \chi(\mathrm{Ca}) &=\min \left(\frac{\mathrm{Ca}}{250}, 1\right)
    \end{aligned}

  The standard values of the parameters are given below. The maximal conductances
  (in :math:`\mathrm{mS} / \mathrm{cm}^{2}`) are
  :math:`\bar{g}_{L}=0.1`, :math:`\bar{g}_{\mathrm{Na}}=30`,
  :math:`\bar{g}_{\mathrm{K}-\mathrm{DR}}=15`,
  :math:`\bar{g}_{\mathrm{Ca}}=10`,
  :math:`\bar{g}_{\mathrm{K}-\mathrm{AHP}}=0.8`,
  :math:`\bar{g}_{\mathrm{K}-\mathrm{C}}=15`,
  :math:`\bar{g}_{\mathrm{NMDA}}=0.0` and
  :math:`\bar{g}_{\mathrm{AMPA}}=0.0`.
  The reversal potentials (in :math:`\mathrm{mV}` ) are
  :math:`V_{\mathrm{Na}}=120, V_{\mathrm{C}}=140, V_{\mathrm{K}}=-15 \mathrm{mV})`
  are :math:`V_{\mathrm{Na}}=120, V_{\mathrm{Ca}}=140, V_{\mathrm{K}}=-15, $V_{L}=0`
  and :math:`V_{\text {Syn }}=60`. The applied currents
  (in :math:`\mu \mathrm{A} / \mathrm{cm}^{2}` ) are :math:`I_{s}=-0.5` and :math:`I_{d}=0.0`.
  The coupling parameters are :math:`g_{c}=2.1 \mathrm{mS} / \mathrm{cm}^{2}` and
  :math:`p=0.5`. The capacitance, :math:`C_{M}`, is
  :math:`3 \mu \mathrm{F} / \mathrm{cm}^{2}` and :math:`\chi(C a)=\min (C a / 250,1)`.
  Values for these parameters, and these function definitions, are taken from Traub et al, 1991.


  Parameters
  ----------
  size: sequence of int, int
    The size of the neuron group.
  gNa: float, JaxArray, ndarray, Initializer, callable
    The maximum conductance of sodium channel.
  gK: float, JaxArray, ndarray, Initializer, callable
    The maximum conductance of potassium delayed-rectifier channel.
  gCa: float, JaxArray, ndarray, Initializer, callable
    The maximum conductance of calcium channel.
  gAHP: float, JaxArray, ndarray, Initializer, callable
    The maximum conductance of potassium after-hyper-polarization channel.
  gC: float, JaxArray, ndarray, Initializer, callable
    The maximum conductance of calcium activated potassium channel.
  gL: float, JaxArray, ndarray, Initializer, callable
    The conductance of leaky channel.
  ENa: float, JaxArray, ndarray, Initializer, callable
    The reversal potential of sodium channel.
  EK: float, JaxArray, ndarray, Initializer, callable
    The reversal potential of potassium delayed-rectifier channel.
  ECa: float, JaxArray, ndarray, Initializer, callable
    The reversal potential of calcium channel.
  EL: float, JaxArray, ndarray, Initializer, callable
    The reversal potential of leaky channel.
  gc: float, JaxArray, ndarray, Initializer, callable
    The coupling strength between the soma and dendrite.
  V_th: float, JaxArray, ndarray, Initializer, callable
    The threshold of the membrane spike.
  Cm: float, JaxArray, ndarray, Initializer, callable
    The threshold of the membrane spike.
  A: float, JaxArray, ndarray, Initializer, callable
    The total cell membrane area, which is normalized to 1.
  p: float, JaxArray, ndarray, Initializer, callable
    The proportion of cell area taken up by the soma.
  Vs_initializer: JaxArray, ndarray, Initializer, callable
    The initializer of somatic membrane potential.
  Vd_initializer: JaxArray, ndarray, Initializer, callable
    The initializer of dendritic membrane potential.
  Ca_initializer: JaxArray, ndarray, Initializer, callable
    The initializer of Calcium concentration.
  method: str
    The numerical integration method.
  name: str
    The group name.

  References
  ----------
  .. [7] Pinsky, Paul F., and John Rinzel. "Intrinsic and network
         rhythmogenesis in a reduced Traub model for CA3 neurons."
         Journal of computational neuroscience 1.1 (1994): 39-60.
  .. [8] Traub, R. D., Wong, R. K., Miles, R., & Michelson, H. (1991).
         A model of a CA3 hippocampal pyramidal neuron incorporating
         voltage-clamp data on intrinsic conductances. Journal of
         neurophysiology, 66(2), 635-650.
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      # maximum conductance
      gNa: Union[float, Array, Initializer, Callable] = 30.,
      gK: Union[float, Array, Initializer, Callable] = 15.,
      gCa: Union[float, Array, Initializer, Callable] = 10.,
      gAHP: Union[float, Array, Initializer, Callable] = 0.8,
      gC: Union[float, Array, Initializer, Callable] = 15.,
      gL: Union[float, Array, Initializer, Callable] = 0.1,
      # reversal potential
      ENa: Union[float, Array, Initializer, Callable] = 60.,
      EK: Union[float, Array, Initializer, Callable] = -75.,
      ECa: Union[float, Array, Initializer, Callable] = 80.,
      EL: Union[float, Array, Initializer, Callable] = -60.,
      # other parameters
      gc: Union[float, Array, Initializer, Callable] = 2.1,
      V_th: Union[float, Array, Initializer, Callable] = 20.,
      Cm: Union[float, Array, Initializer, Callable] = 3.0,
      p: Union[float, Array, Initializer, Callable] = 0.5,
      A: Union[float, Array, Initializer, Callable] = 1.,
      # initializers
      Vs_initializer: Union[Initializer, Callable, Array] = OneInit(-64.6),
      Vd_initializer: Union[Initializer, Callable, Array] = OneInit(-64.5),
      Ca_initializer: Union[Initializer, Callable, Array] = OneInit(0.2),
      # others
      noise: Union[float, Array, Initializer, Callable] = None,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    # initialization
    super(PinskyRinzelModel, self).__init__(size=size,
                                            keep_size=keep_size,
                                            name=name,
                                            mode=mode)
    check_mode(self.mode, (NormalMode, BatchingMode), self.__class__)

    # conductance parameters
    self.gAHP = parameter(gAHP, self.varshape, allow_none=False)
    self.gCa = parameter(gCa, self.varshape, allow_none=False)
    self.gNa = parameter(gNa, self.varshape, allow_none=False)
    self.gK = parameter(gK, self.varshape, allow_none=False)
    self.gL = parameter(gL, self.varshape, allow_none=False)
    self.gC = parameter(gC, self.varshape, allow_none=False)

    # reversal potential parameters
    self.ENa = parameter(ENa, self.varshape, allow_none=False)
    self.ECa = parameter(ECa, self.varshape, allow_none=False)
    self.EK = parameter(EK, self.varshape, allow_none=False)
    self.EL = parameter(EL, self.varshape, allow_none=False)

    # other neuronal parameters
    self.V_th = parameter(V_th, self.varshape, allow_none=False)
    self.Cm = parameter(Cm, self.varshape, allow_none=False)
    self.gc = parameter(gc, self.varshape, allow_none=False)
    self.p = parameter(p, self.varshape, allow_none=False)
    self.A = parameter(A, self.varshape, allow_none=False)
    self.noise = init_noise(noise, self.varshape, num_vars=8)

    # initializers
    check_initializer(Vs_initializer, 'Vs_initializer', allow_none=False)
    check_initializer(Vd_initializer, 'Vd_initializer', allow_none=False)
    check_initializer(Ca_initializer, 'Ca_initializer', allow_none=False)
    self._Vs_initializer = Vs_initializer
    self._Vd_initializer = Vd_initializer
    self._Ca_initializer = Ca_initializer

    # variables
    self.Vs = variable_(self._Vs_initializer, self.varshape, mode)
    self.Vd = variable_(self._Vd_initializer, self.varshape, mode)
    self.Ca = variable_(self._Ca_initializer, self.varshape, mode)
    self.h = bm.Variable(self.inf_h(self.Vs), batch_axis=0 if isinstance(mode, BatchingMode) else None)
    self.n = bm.Variable(self.inf_n(self.Vs), batch_axis=0 if isinstance(mode, BatchingMode) else None)
    self.s = bm.Variable(self.inf_s(self.Vd), batch_axis=0 if isinstance(mode, BatchingMode) else None)
    self.c = bm.Variable(self.inf_c(self.Vd), batch_axis=0 if isinstance(mode, BatchingMode) else None)
    self.q = bm.Variable(self.inf_q(self.Ca), batch_axis=0 if isinstance(mode, BatchingMode) else None)
    self.Id = variable_(bm.zeros, self.varshape, mode)  # input to soma
    self.Is = variable_(bm.zeros, self.varshape, mode)  # input to dendrite
    # self.spike = bm.Variable(bm.zeros(self.varshape, dtype=bool))

    # integral
    if self.noise is None:
      self.integral = odeint(method=method, f=self.derivative)
    else:
      self.integral = sdeint(method=method, f=self.derivative, g=self.noise)

  def reset_state(self, batch_size=None):
    self.Vd.value = variable_(self._Vd_initializer, self.varshape, batch_size)
    self.Vs.value = variable_(self._Vs_initializer, self.varshape, batch_size)
    self.Ca.value = variable_(self._Ca_initializer, self.varshape, batch_size)
    batch_axis = 0 if isinstance(self.mode, BatchingMode) else None
    self.h.value = bm.Variable(self.inf_h(self.Vs), batch_axis=batch_axis)
    self.n.value = bm.Variable(self.inf_n(self.Vs), batch_axis=batch_axis)
    self.s.value = bm.Variable(self.inf_s(self.Vd), batch_axis=batch_axis)
    self.c.value = bm.Variable(self.inf_c(self.Vd), batch_axis=batch_axis)
    self.q.value = bm.Variable(self.inf_q(self.Ca), batch_axis=batch_axis)
    self.Id.value = variable_(bm.zeros, self.varshape, batch_size)
    self.Is.value = variable_(bm.zeros, self.varshape, batch_size)
    # self.spike[:] = False

  def dCa(self, Ca, t, s, Vd):
    ICa = self.gCa * s * s * (Vd - self.ECa)
    return -0.13 * ICa - 0.075 * Ca

  def dh(self, h, t, Vs):
    return self.alpha_h(Vs) * (1 - h) - self.beta_h(Vs) * h

  def dn(self, n, t, Vs):
    return self.alpha_n(Vs) * (1 - n) - self.beta_n(Vs) * n

  def ds(self, s, t, Vd):
    return self.alpha_s(Vd) * (1 - s) - self.beta_s(Vd) * s

  def dc(self, c, t, Vd):
    return self.alpha_c(Vd) * (1 - c) - self.beta_c(Vd) * c

  def dq(self, q, t, Ca):
    return self.alpha_q(Ca) * (1 - q) - self.beta_q(Ca) * q

  def dVs(self, Vs, t, h, n, Vd):
    I_Na = (self.gNa * self.inf_m(Vs) ** 2 * h) * (Vs - self.ENa)
    I_KDR = (self.gK * n) * (Vs - self.EK)
    I_leak = self.gL * (Vs - self.EL)
    I_gj = self.gc / self.p * (Vd - Vs)
    dVdt = (- I_Na - I_KDR - I_leak + I_gj + self.Is / self.p) / self.Cm
    return dVdt

  def dVd(self, Vd, t, s, q, c, Ca, Vs):
    I_leak = self.gL * (Vd - self.EL)
    I_Ca = self.gCa * s * s * (Vd - self.ECa)
    I_AHP = self.gAHP * q * (Vd - self.EK)
    I_C = self.gC * bm.minimum(Ca / 250., 1.) * (Vd - self.EK)
    p = 1 - self.p
    I_gj = self.gc / p * (Vs - Vd)
    dVdt = (- I_leak - I_Ca - I_AHP - I_C + I_gj + self.Id / p) / self.Cm
    return dVdt

  @property
  def derivative(self):
    return JointEq([self.dVs, self.dVd, self.dCa, self.dh, self.dn, self.ds, self.dc, self.dq])

  def update(self, tdi, x=None):
    assert x is None
    Vs, Vd, Ca, h, n, s, c, q = self.integral(Vs=self.Vs.value,
                                              Vd=self.Vd.value,
                                              Ca=self.Ca.value,
                                              h=self.h.value,
                                              n=self.n.value,
                                              s=self.s.value,
                                              c=self.c.value,
                                              q=self.q.value,
                                              t=tdi['t'],
                                              dt=tdi['dt'])
    self.Vs.value = Vs
    self.Vd.value = Vd
    self.Ca.value = Ca
    self.h.value = h
    self.n.value = n
    self.s.value = s
    self.c.value = c
    self.q.value = q

  def clear_input(self):
    self.Id[:] = 0.
    self.Is[:] = 0.

  def alpha_m(self, Vs):
    return 0.32 * (13.1 - (Vs + 60.)) / (bm.exp((13.1 - (Vs + 60.)) / 4.) - 1.)

  def beta_m(self, Vs):
    return 0.28 * ((Vs + 60.) - 40.1) / (bm.exp(((Vs + 60.) - 40.1) / 5.) - 1.)

  def inf_m(self, Vs):
    alpha = self.alpha_m(Vs)
    beta = self.beta_m(Vs)
    return alpha / (alpha + beta)

  def alpha_n(self, Vs):
    return 0.016 * (35.1 - (Vs + 60.)) / (bm.exp((35.1 - (Vs + 60.)) / 5) - 1)

  def beta_n(self, Vs):
    return 0.25 * bm.exp(0.5 - 0.025 * (Vs + 60.))

  def inf_n(self, Vs):
    alpha = self.alpha_n(Vs)
    beta = self.beta_n(Vs)
    return alpha / (alpha + beta)

  def alpha_h(self, Vs):
    return 0.128 * bm.exp((17. - (Vs + 60.)) / 18.)

  def beta_h(self, Vs):
    return 4. / (1 + bm.exp((40. - (Vs + 60.)) / 5))

  def inf_h(self, Vs):
    alpha = self.alpha_h(Vs)
    beta = self.beta_h(Vs)
    return alpha / (alpha + beta)

  def alpha_s(self, Vd):
    return 1.6 / (1 + bm.exp(-0.072 * ((Vd + 60.) - 65.)))

  def beta_s(self, Vd):
    return 0.02 * ((Vd + 60.) - 51.1) / (bm.exp(((Vd + 60.) - 51.1) / 5.) - 1.)

  def inf_s(self, Vd):
    alpha = self.alpha_s(Vd)
    beta = self.beta_s(Vd)
    return alpha / (alpha + beta)

  def alpha_c(self, Vd):
    return bm.where((Vd + 60.) <= 50.,
                    (bm.exp(((Vd + 60.) - 10.) / 11.) - bm.exp(((Vd + 60.) - 6.5) / 27.)) / 18.975,
                    2. * bm.exp((6.5 - (Vd + 60.)) / 27.))

  def beta_c(self, Vd):
    alpha_c = (bm.exp(((Vd + 60.) - 10.) / 11.) - bm.exp(((Vd + 60.) - 6.5) / 27.)) / 18.975
    return bm.where((Vd + 60.) <= 50., 2. * bm.exp((6.5 - (Vd + 60.)) / 27.) - alpha_c, 0.)

  def inf_c(self, Vd):
    alpha_c = self.alpha_c(Vd)
    beta_c = self.beta_c(Vd)
    return alpha_c / (alpha_c + beta_c)

  def alpha_q(self, Ca):
    return bm.minimum(2e-5 * Ca, 1e-2)

  def beta_q(self, Ca):
    return 1e-3

  def inf_q(self, Ca):
    alpha = self.alpha_q(Ca)
    beta = self.beta_q(Ca)
    return alpha / (alpha + beta)


class WangBuzsakiModel(NeuGroup):
  r"""Wang-Buzsaki model [9]_, an implementation of a modified Hodgkin-Huxley model.

  Each model is described by a single compartment and obeys the current balance equation:

  .. math::

      C_{m} \frac{d V}{d t}=-I_{\mathrm{Na}}-I_{\mathrm{K}}-I_{\mathrm{L}}-I_{\mathrm{syn}}+I_{\mathrm{app}}

  where :math:`C_{m}=1 \mu \mathrm{F} / \mathrm{cm}^{2}` and :math:`I_{\mathrm{app}}` is the
  injected current (in :math:`\mu \mathrm{A} / \mathrm{cm}^{2}` ). The leak current
  :math:`I_{\mathrm{L}}=g_{\mathrm{L}}\left(V-E_{\mathrm{L}}\right)` has a conductance
  :math:`g_{\mathrm{L}}=0.1 \mathrm{mS} / \mathrm{cm}^{2}`, so that the passive time constant
  :math:`\tau_{0}=C_{m} / g_{\mathrm{L}}=10 \mathrm{msec} ; E_{\mathrm{L}}=-65 \mathrm{mV}`.

  The spike-generating :math:`\mathrm{Na}^{+}` and :math:`\mathrm{K}^{+}` voltage-dependent ion
  currents :math:`\left(I_{\mathrm{Na}}\right.` and :math:`I_{\mathrm{K}}` ) are of the
  Hodgkin-Huxley type (Hodgkin and Huxley, 1952). The transient sodium current
  :math:`I_{\mathrm{Na}}=g_{\mathrm{Na}} m_{\infty}^{3} h\left(V-E_{\mathrm{Na}}\right)`,
  where the activation variable :math:`m` is assumed fast and substituted by its steady-state
  function :math:`m_{\infty}=\alpha_{m} /\left(\alpha_{m}+\beta_{m}\right)` ;
  :math:`\alpha_{m}(V)=-0.1(V+35) /(\exp (-0.1(V+35))-1), \beta_{m}(V)=4 \exp (-(V+60) / 18)`.
  The inactivation variable :math:`h` obeys a first-order kinetics:

  .. math::

      \frac{d h}{d t}=\phi\left(\alpha_{h}(1-h)-\beta_{h} h\right)

  where :math:`\alpha_{h}(V)=0.07 \exp (-(V+58) / 20)` and
  :math:`\beta_{h}(V)=1 /(\exp (-0.1(V+28)) +1) \cdot g_{\mathrm{Na}}=35 \mathrm{mS} / \mathrm{cm}^{2}` ;
  :math:`E_{\mathrm{Na}}=55 \mathrm{mV}, \phi=5 .`

  The delayed rectifier :math:`I_{\mathrm{K}}=g_{\mathrm{K}} n^{4}\left(V-E_{\mathrm{K}}\right)`,
  where the activation variable :math:`n` obeys the following equation:

  .. math::

     \frac{d n}{d t}=\phi\left(\alpha_{n}(1-n)-\beta_{n} n\right)

  with :math:`\alpha_{n}(V)=-0.01(V+34) /(\exp (-0.1(V+34))-1)` and
  :math:`\beta_{n}(V)=0.125\exp (-(V+44) / 80)` ; :math:`g_{\mathrm{K}}=9 \mathrm{mS} / \mathrm{cm}^{2}`, and
  :math:`E_{\mathrm{K}}=-90 \mathrm{mV}`.


  Parameters
  ----------
  size: sequence of int, int
    The size of the neuron group.
  ENa: float, JaxArray, ndarray, Initializer, callable
    The reversal potential of sodium. Default is 50 mV.
  gNa: float, JaxArray, ndarray, Initializer, callable
    The maximum conductance of sodium channel. Default is 120 msiemens.
  EK: float, JaxArray, ndarray, Initializer, callable
    The reversal potential of potassium. Default is -77 mV.
  gK: float, JaxArray, ndarray, Initializer, callable
    The maximum conductance of potassium channel. Default is 36 msiemens.
  EL: float, JaxArray, ndarray, Initializer, callable
    The reversal potential of learky channel. Default is -54.387 mV.
  gL: float, JaxArray, ndarray, Initializer, callable
    The conductance of learky channel. Default is 0.03 msiemens.
  V_th: float, JaxArray, ndarray, Initializer, callable
    The threshold of the membrane spike. Default is 20 mV.
  C: float, JaxArray, ndarray, Initializer, callable
    The membrane capacitance. Default is 1 ufarad.
  phi: float, JaxArray, ndarray, Initializer, callable
    The temperature regulator constant.
  V_initializer: JaxArray, ndarray, Initializer, callable
    The initializer of membrane potential.
  h_initializer: JaxArray, ndarray, Initializer, callable
    The initializer of h channel.
  n_initializer: JaxArray, ndarray, Initializer, callable
    The initializer of n channel.
  method: str
    The numerical integration method.
  name: str
    The group name.

  References
  ----------
  .. [9] Wang, X.J. and Buzsaki, G., (1996) Gamma oscillation by synaptic
         inhibition in a hippocampal interneuronal network model. Journal of
         neuroscience, 16(20), pp.6402-6413.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      ENa: Union[float, Array, Initializer, Callable] = 55.,
      gNa: Union[float, Array, Initializer, Callable] = 35.,
      EK: Union[float, Array, Initializer, Callable] = -90.,
      gK: Union[float, Array, Initializer, Callable] = 9.,
      EL: Union[float, Array, Initializer, Callable] = -65,
      gL: Union[float, Array, Initializer, Callable] = 0.1,
      V_th: Union[float, Array, Initializer, Callable] = 20.,
      phi: Union[float, Array, Initializer, Callable] = 5.0,
      C: Union[float, Array, Initializer, Callable] = 1.0,
      V_initializer: Union[Initializer, Callable, Array] = OneInit(-65.),
      h_initializer: Union[Initializer, Callable, Array] = OneInit(0.6),
      n_initializer: Union[Initializer, Callable, Array] = OneInit(0.32),
      noise: Union[float, Array, Initializer, Callable] = None,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    # initialization
    super(WangBuzsakiModel, self).__init__(size=size, keep_size=keep_size, name=name, mode=mode)
    check_mode(self.mode, (BatchingMode, NormalMode), self.__class__)

    # parameters
    self.ENa = parameter(ENa, self.varshape, allow_none=False)
    self.EK = parameter(EK, self.varshape, allow_none=False)
    self.EL = parameter(EL, self.varshape, allow_none=False)
    self.gNa = parameter(gNa, self.varshape, allow_none=False)
    self.gK = parameter(gK, self.varshape, allow_none=False)
    self.gL = parameter(gL, self.varshape, allow_none=False)
    self.C = parameter(C, self.varshape, allow_none=False)
    self.phi = parameter(phi, self.varshape, allow_none=False)
    self.V_th = parameter(V_th, self.varshape, allow_none=False)
    self.noise = init_noise(noise, self.varshape, num_vars=3)

    # initializers
    check_initializer(h_initializer, 'h_initializer', allow_none=False)
    check_initializer(n_initializer, 'n_initializer', allow_none=False)
    check_initializer(V_initializer, 'V_initializer', allow_none=False)
    self._h_initializer = h_initializer
    self._n_initializer = n_initializer
    self._V_initializer = V_initializer

    # variables
    self.h = variable_(self._h_initializer, self.varshape, mode)
    self.n = variable_(self._n_initializer, self.varshape, mode)
    self.V = variable_(self._V_initializer, self.varshape, mode)
    self.input = variable_(bm.zeros, self.varshape, mode)
    self.spike = variable_(lambda s: bm.zeros(s, dtype=bool), self.varshape, mode)

    # integral
    if self.noise is None:
      self.integral = odeint(method=method, f=self.derivative)
    else:
      self.integral = sdeint(method=method, f=self.derivative, g=self.noise)

  def reset_state(self, batch_size=None):
    self.h.value = variable_(self._h_initializer, self.varshape, batch_size)
    self.n.value = variable_(self._n_initializer, self.varshape, batch_size)
    self.V.value = variable_(self._V_initializer, self.varshape, batch_size)
    self.input.value = variable_(bm.zeros, self.varshape, batch_size)
    self.spike.value = variable_(lambda s: bm.zeros(s, dtype=bool), self.varshape, batch_size)

  def m_inf(self, V):
    alpha = -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)
    beta = 4. * bm.exp(-(V + 60.) / 18.)
    return alpha / (alpha + beta)

  def dh(self, h, t, V):
    alpha = 0.07 * bm.exp(-(V + 58) / 20)
    beta = 1 / (bm.exp(-0.1 * (V + 28)) + 1)
    dhdt = alpha * (1 - h) - beta * h
    return self.phi * dhdt

  def dn(self, n, t, V):
    alpha = -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1)
    beta = 0.125 * bm.exp(-(V + 44) / 80)
    dndt = alpha * (1 - n) - beta * n
    return self.phi * dndt

  def dV(self, V, t, h, n, I_ext):
    INa = self.gNa * self.m_inf(V) ** 3 * h * (V - self.ENa)
    IK = self.gK * n ** 4 * (V - self.EK)
    IL = self.gL * (V - self.EL)
    dVdt = (- INa - IK - IL + I_ext) / self.C
    return dVdt

  @property
  def derivative(self):
    return JointEq([self.dV, self.dh, self.dn])

  def update(self, tdi, x=None):
    t, dt = tdi['t'], tdi['dt']
    if x is not None: self.input += x
    V, h, n = self.integral(self.V, self.h, self.n, t, self.input, dt)
    self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    self.V.value = V
    self.h.value = h
    self.n.value = n

  def clear_input(self):
    self.input[:] = 0.
