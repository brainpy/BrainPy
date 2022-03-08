# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.integrators.joint_eq import JointEq
from brainpy.integrators.ode import odeint
from brainpy.dyn.base import NeuGroup

__all__ = [
  'FHN',
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

  def __init__(self, size, a=0.7, b=0.8, tau=12.5, Vth=1.8, method='exp_auto', name=None):
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
