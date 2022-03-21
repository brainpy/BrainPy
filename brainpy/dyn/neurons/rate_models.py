# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.dyn.base import NeuGroup
from brainpy.integrators.dde import ddeint
from brainpy.integrators.joint_eq import JointEq
from brainpy.integrators.ode import odeint
from brainpy.types import Parameter, Shape

__all__ = [
  'FHN',
  'FeedbackFHN',
]


class FHN(NeuGroup):
  r"""FitzHugh-Nagumo neuron model.

  **Model Descriptions**

  The FitzHugh–Nagumo model (FHN), named after Richard FitzHugh (1922–2007)
  who suggested the system in 1961 [1]_ and J. Nagumo et al. who created the
  equivalent circuit the following year, describes a prototype of an excitable
  system (e.g., a neuron).

  The motivation for the FitzHugh-Nagumo model was to isolate conceptually
  the essentially mathematical properties of excitation and propagation from
  the electrochemical properties of sodium and potassium ion flow. The model
  consists of

  - a *voltage-like variable* having cubic nonlinearity that allows regenerative
    self-excitation via a positive feedback, and
  - a *recovery variable* having a linear dynamics that provides a slower negative feedback.

  .. math::

     \begin{aligned}
     {\dot {v}} &=v-{\frac {v^{3}}{3}}-w+RI_{\rm {ext}},  \\
     \tau {\dot  {w}}&=v+a-bw.
     \end{aligned}

  The FHN Model is an example of a relaxation oscillator
  because, if the external stimulus :math:`I_{\text{ext}}`
  exceeds a certain threshold value, the system will exhibit
  a characteristic excursion in phase space, before the
  variables :math:`v` and :math:`w` relax back to their rest values.
  This behaviour is typical for spike generations (a short,
  nonlinear elevation of membrane voltage :math:`v`,
  diminished over time by a slower, linear recovery variable
  :math:`w`) in a neuron after stimulation by an external
  input current.

  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> fhn = bp.dyn.FHN(1)
    >>> runner = bp.dyn.DSRunner(fhn, inputs=('input', 1.), monitors=['V', 'w'])
    >>> runner.run(100.)
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.w, legend='w')
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', show=True)

  **Model Parameters**

  ============= ============== ======== ========================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------
  a             1              \        Positive constant
  b             1              \        Positive constant
  tau           10             ms       Membrane time constant.
  V_th          1.8            mV       Threshold potential of spike.
  ============= ============== ======== ========================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                   0                 Membrane potential.
  w                   0                 A recovery variable which represents
                                        the combined effects of sodium channel
                                        de-inactivation and potassium channel
                                        deactivation.
  input               0                 External and synaptic input current.
  spike               False             Flag to mark whether the neuron is spiking.
  t_last_spike       -1e7               Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] FitzHugh, Richard. "Impulses and physiological states in theoretical models of nerve membrane." Biophysical journal 1.6 (1961): 445-466.
  .. [2] https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model
  .. [3] http://www.scholarpedia.org/article/FitzHugh-Nagumo_model

  """

  def __init__(self,
               size: Shape,
               a: Parameter = 0.7,
               b: Parameter = 0.8,
               tau: Parameter = 12.5,
               Vth: Parameter = 1.8,
               method: str = 'exp_auto',
               name: str = None):
    # initialization
    super(FHN, self).__init__(size=size, name=name)

    # parameters
    self.a = a
    self.b = b
    self.tau = tau
    self.Vth = Vth

    # variables
    self.w = bm.Variable(bm.zeros(self.num))
    self.V = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

    # integral
    self.integral = odeint(method=method, f=self.derivative)

  def dV(self, V, t, w, I_ext):
    return V - V * V * V / 3 - w + I_ext

  def dw(self, w, t, V):
    return (V + self.a - self.b * w) / self.tau

  @property
  def derivative(self):
    return JointEq([self.dV, self.dw])

  def update(self, _t, _dt):
    V, w = self.integral(self.V, self.w, _t, self.input, dt=_dt)
    self.spike.value = bm.logical_and(V >= self.Vth, self.V < self.Vth)
    self.t_last_spike.value = bm.where(self.spike, _t, self.t_last_spike)
    self.V.value = V
    self.w.value = w
    self.input[:] = 0.


class FeedbackFHN(NeuGroup):
  r"""FitzHugh-Nagumo model with recurrent neural feedback.

  The equation of the feedback FitzHugh-Nagumo model [4]_ is given by

  .. math::

     \begin{aligned}
     \frac{dv}{dt} &= v(t) - \frac{v^3(t)}{3} - w(t) + \mu[v(t-\mathrm{delay}) - v_0] \\
     \frac{dw}{dt} &= [v(t) + a - b w(t)] / \tau
     \end{aligned}


  **Model Examples**

  >>> import brainpy as bp
  >>> fhn = bp.dyn.FeedbackFHN(1, delay=10.)
  >>> runner = bp.dyn.DSRunner(fhn, inputs=('input', 1.), monitors=['V', 'w'])
  >>> runner.run(100.)
  >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.w, legend='w')
  >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', show=True)


  **Model Parameters**

  ============= ============== ======== ========================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------
  a             1              \        Positive constant
  b             1              \        Positive constant
  tau           12.5           ms       Membrane time constant.
  delay         10             ms       Synaptic delay time constant.
  V_th          1.8            mV       Threshold potential of spike.
  v0            -1             mV       Resting potential.
  mu            1.8            \        The feedback strength. When positive, it is a excitatory feedback;
                                        when negative, it is a inhibitory feedback.
  ============= ============== ======== ========================

  References
  ----------
  .. [4] Plant, Richard E. (1981). *A FitzHugh Differential-Difference
         Equation Modeling Recurrent Neural Feedback. SIAM Journal on
         Applied Mathematics, 40(1), 150–162.* doi:10.1137/0140012

  """

  def __init__(self,
               size: Shape,
               a: Parameter = 0.7,
               b: Parameter = 0.8,
               delay: Parameter = 10.,
               tau: Parameter = 12.5,
               mu: Parameter = 1.6886,
               v0: Parameter = -1,
               method: str = 'rk4',
               name: str = None):
    super(FeedbackFHN, self).__init__(size=size, name=name)

    # parameters
    self.a = a
    self.b = b
    self.delay = delay
    self.tau = tau
    self.mu = mu  # feedback strength
    self.v0 = v0  # resting potential

    # variables
    self.w = bm.Variable(bm.zeros(self.num))
    self.V = bm.Variable(bm.zeros(self.num))
    self.Vdelay = bm.TimeDelay(self.V, self.delay, interp_method='round')
    self.input = bm.Variable(bm.zeros(self.num))

    # integral
    self.integral = ddeint(method=method,
                           f=self.derivative,
                           state_delays={'V': self.Vdelay})

  def dV(self, V, t, w):
    return (V - V * V * V / 3 - w + self.input +
            self.mu * (self.Vdelay(t - self.delay) - self.v0))

  def dw(self, w, t, V):
    return (V + self.a - self.b * w) / self.tau

  @property
  def derivative(self):
    return JointEq([self.dV, self.dw])

  def update(self, _t, _dt):
    V, w = self.integral(self.V, self.w, _t, dt=_dt)
    self.V.value = V
    self.w.value = w
    self.input[:] = 0.
