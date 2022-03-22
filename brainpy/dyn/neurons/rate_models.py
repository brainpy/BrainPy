# -*- coding: utf-8 -*-
import numpy as np
from jax.experimental.host_callback import id_tap

import brainpy.math as bm
from brainpy import check
from brainpy.dyn.base import NeuGroup
from brainpy.integrators.dde import ddeint
from brainpy.integrators.joint_eq import JointEq
from brainpy.integrators.ode import odeint
from brainpy.tools.checking import check_float
from brainpy.types import Parameter, Shape
from .noise_models import OUProcess

__all__ = [
  'RateGroup',
  'RateFHN',
  'FeedbackFHN',
  'RateQIF',
  'StuartLandauOscillator',
  'WilsonCowanModel',
]


class RateGroup(NeuGroup):
  def update(self, _t, _dt):
    raise NotImplementedError


class RateFHN(NeuGroup):
  r"""FitzHugh-Nagumo system used in [1]_.

  .. math::

     \frac{dx}{dt} = -\alpha V^3 + \beta V^2 + \gamma V - w + I_{ext}\\
     \tau \frac{dy}{dt} = (V - \delta  - \epsilon w)

  Parameters
  ----------
  size: Shape
    The model size.
  x_ou_mean
    The noise mean of the :math:`x` variable, [mV/ms]
  y_ou_mean
    The noise mean of the :math:`y` variable, [mV/ms].
  x_ou_sigma
    The noise intensity of the :math:`x` variable, [mV/ms/sqrt(ms)].
  y_ou_sigma
    The noise intensity of the :math:`y` variable, [mV/ms/sqrt(ms)].
  x_ou_tau
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`x` variable, [ms].
  y_ou_tau
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`y` variable, [ms].


  References
  ----------
  .. [1] Kostova, T., Ravindran, R., & Schonbek, M. (2004). FitzHugh–Nagumo
         revisited: Types of bifurcations, periodical forcing and stability
         regions by a Lyapunov functional. International journal of
         bifurcation and chaos, 14(03), 913-925.

  """

  def __init__(
      self,
      size: Shape,

      # fhn parameters
      alpha: Parameter = 3.0,
      beta: Parameter = 4.0,
      gamma: Parameter = -1.5,
      delta: Parameter = 0.0,
      epsilon: Parameter = 0.5,
      tau: Parameter = 20.0,

      # noise parameters
      x_ou_mean: Parameter = 0.0,
      x_ou_sigma: Parameter = 0.0,
      x_ou_tau: Parameter = 5.0,
      y_ou_mean: Parameter = 0.0,
      y_ou_sigma: Parameter = 0.0,
      y_ou_tau: Parameter = 5.0,

      # other parameters
      method: str = None,
      sde_method: str = None,
      name: str = None,
  ):
    super(RateFHN, self).__init__(size=size, name=name)

    # model parameters
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.delta = delta
    self.epsilon = epsilon
    self.tau = tau

    # noise parameters
    self.x_ou_mean = x_ou_mean  # mV/ms, OU process
    self.y_ou_mean = y_ou_mean  # mV/ms, OU process
    self.x_ou_sigma = x_ou_sigma  # mV/ms/sqrt(ms), noise intensity
    self.y_ou_sigma = y_ou_sigma  # mV/ms/sqrt(ms), noise intensity
    self.x_ou_tau = x_ou_tau  # ms, timescale of the Ornstein-Uhlenbeck noise process
    self.y_ou_tau = y_ou_tau  # ms, timescale of the Ornstein-Uhlenbeck noise process

    # variables
    self.x = bm.Variable(bm.random.random(self.num) * 0.05)
    self.y = bm.Variable(bm.random.random(self.num) * 0.05)
    self.input = bm.Variable(bm.zeros(self.num))

    # noise variables
    self.x_ou = self.y_ou = None
    if bm.any(self.x_ou_mean > 0.) or bm.any(self.x_ou_sigma > 0.):
      self.x_ou = OUProcess(self.num,
                            self.x_ou_mean, self.x_ou_sigma, self.x_ou_tau,
                            method=sde_method)
    if bm.any(self.y_ou_mean > 0.) or bm.any(self.y_ou_sigma > 0.):
      self.y_ou = OUProcess(self.num,
                            self.y_ou_mean, self.y_ou_sigma, self.y_ou_tau,
                            method=sde_method)

    # integral functions
    self.integral = odeint(f=JointEq([self.dx, self.dy]), method=method)

  def dx(self, x, t, y, x_ext):
    return - self.alpha * x ** 3 + self.beta * x ** 2 + self.gamma * x - y + x_ext

  def dy(self, y, t, x, y_ext=0.):
    return (x - self.delta - self.epsilon * y) / self.tau + y_ext

  def update(self, _t, _dt):
    if self.x_ou is not None:
      self.input += self.x_ou.x
      self.x_ou.update(_t, _dt)
    y_ext = 0.
    if self.y_ou is not None:
      y_ext = self.y_ou.x
      self.y_ou.update(_t, _dt)
    x, y = self.integral(self.x, self.y, _t, x_ext=self.input, y_ext=y_ext, dt=_dt)
    self.x.value = x
    self.y.value = y
    self.input[:] = 0.


class FeedbackFHN(NeuGroup):
  r"""FitzHugh-Nagumo model with recurrent neural feedback.

  The equation of the feedback FitzHugh-Nagumo model [4]_ is given by

  .. math::

     \begin{aligned}
     \frac{dx}{dt} &= x(t) - \frac{x^3(t)}{3} - y(t) + \mu[x(t-\mathrm{delay}) - x_0] \\
     \frac{dy}{dt} &= [x(t) + a - b y(t)] / \tau
     \end{aligned}


  **Model Examples**

  >>> import brainpy as bp
  >>> fhn = bp.dyn.FeedbackFHN(1, delay=10.)
  >>> runner = bp.dyn.DSRunner(fhn, inputs=('input', 1.), monitors=['x', 'y'])
  >>> runner.run(100.)
  >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.y, legend='y')
  >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.x, legend='x', show=True)


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

  Parameters
  ----------
    x_ou_mean
    The noise mean of the :math:`x` variable, [mV/ms]
  y_ou_mean
    The noise mean of the :math:`y` variable, [mV/ms].
  x_ou_sigma
    The noise intensity of the :math:`x` variable, [mV/ms/sqrt(ms)].
  y_ou_sigma
    The noise intensity of the :math:`y` variable, [mV/ms/sqrt(ms)].
  x_ou_tau
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`x` variable, [ms].
  y_ou_tau
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`y` variable, [ms].



  References
  ----------
  .. [4] Plant, Richard E. (1981). *A FitzHugh Differential-Difference
         Equation Modeling Recurrent Neural Feedback. SIAM Journal on
         Applied Mathematics, 40(1), 150–162.* doi:10.1137/0140012

  """

  def __init__(
      self,
      size: Shape,

      # model parameters
      a: Parameter = 0.7,
      b: Parameter = 0.8,
      delay: Parameter = 10.,
      tau: Parameter = 12.5,
      mu: Parameter = 1.6886,
      x0: Parameter = -1,

      # noise parameters
      x_ou_mean: Parameter = 0.0,
      x_ou_sigma: Parameter = 0.0,
      x_ou_tau: Parameter = 5.0,
      y_ou_mean: Parameter = 0.0,
      y_ou_sigma: Parameter = 0.0,
      y_ou_tau: Parameter = 5.0,

      # other parameters
      method: str = 'rk4',
      sde_method: str = None,
      name: str = None,
      dt: float = None
  ):
    super(FeedbackFHN, self).__init__(size=size, name=name)

    # dt
    self.dt = bm.get_dt() if dt is None else dt
    check_float(self.dt, 'dt', allow_none=False, min_bound=0., allow_int=False)

    # parameters
    self.a = a
    self.b = b
    self.delay = delay
    self.tau = tau
    self.mu = mu  # feedback strength
    self.v0 = x0  # resting potential

    # noise parameters
    self.x_ou_mean = x_ou_mean
    self.y_ou_mean = y_ou_mean
    self.x_ou_sigma = x_ou_sigma
    self.y_ou_sigma = y_ou_sigma
    self.x_ou_tau = x_ou_tau
    self.y_ou_tau = y_ou_tau

    # variables
    self.x = bm.Variable(bm.zeros(self.num))
    self.y = bm.Variable(bm.zeros(self.num))
    self.x_delay = bm.TimeDelay(self.x, self.delay, dt=self.dt, interp_method='round')
    self.input = bm.Variable(bm.zeros(self.num))

    # noise variables
    self.x_ou = self.y_ou = None
    if bm.any(self.x_ou_mean > 0.) or bm.any(self.x_ou_sigma > 0.):
      self.x_ou = OUProcess(self.num,
                            self.x_ou_mean, self.x_ou_sigma, self.x_ou_tau,
                            method=sde_method)
    if bm.any(self.y_ou_mean > 0.) or bm.any(self.y_ou_sigma > 0.):
      self.y_ou = OUProcess(self.num,
                            self.y_ou_mean, self.y_ou_sigma, self.y_ou_tau,
                            method=sde_method)

    # integral
    self.integral = ddeint(method=method,
                           f=JointEq([self.dx, self.dy]),
                           state_delays={'V': self.x_delay})

  def dx(self, x, t, y, x_ext):
    return x - x * x * x / 3 - y + x_ext + self.mu * (self.x_delay(t - self.delay) - self.v0)

  def dy(self, y, t, x, y_ext):
    return (x + self.a - self.b * y + y_ext) / self.tau

  def _check_dt(self, dt, *args):
    if np.absolute(dt - self.dt) > 1e-6:
      raise ValueError(f'The "dt" {dt} used in model running is '
                       f'not consistent with the "dt" {self.dt} '
                       f'used in model definition.')

  def update(self, _t, _dt):
    if check.is_checking():
      id_tap(self._check_dt, _dt)
    if self.x_ou is not None:
      self.input += self.x_ou.x
      self.x_ou.update(_t, _dt)
    y_ext = 0.
    if self.y_ou is not None:
      y_ext = self.y_ou.x
      self.y_ou.update(_t, _dt)
    x, y = self.integral(self.x, self.y, _t, x_ext=self.input, y_ext=y_ext, dt=_dt)
    self.x.value = x
    self.y.value = y
    self.input[:] = 0.


class RateQIF(NeuGroup):
  r"""A mean-field model of a quadratic integrate-and-fire neuron population.

  **Model Descriptions**

  The QIF population mean-field model, which has been derived from a
  population of all-to-all coupled QIF neurons in [5]_.
  The model equations are given by:

  .. math::

     \begin{aligned}
     \tau \dot{r} &=\frac{\Delta}{\pi \tau}+2 r v \\
     \tau \dot{v} &=v^{2}+\bar{\eta}+I(t)+J r \tau-(\pi r \tau)^{2}
     \end{aligned}

  where :math:`r` is the average firing rate and :math:`v` is the
  average membrane potential of the QIF population [5]_.

  This mean-field model is an exact representation of the macroscopic
  firing rate and membrane potential dynamics of a spiking neural network
  consisting of QIF neurons with Lorentzian distributed background
  excitabilities. While the mean-field derivation is mathematically
  only valid for all-to-all coupled populations of infinite size, it
  has been shown that there is a close correspondence between the
  mean-field model and neural populations with sparse coupling and
  population sizes of a few thousand neurons [6]_.

  **Model Parameters**

  ============= ============== ======== ========================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------
  tau           1              ms       the population time constant
  eta           -5.            \        the mean of a Lorenzian distribution over the neural excitability in the population
  delta         1.0            \        the half-width at half maximum of the Lorenzian distribution over the neural excitability
  J             15             \        the strength of the recurrent coupling inside the population
  ============= ============== ======== ========================

  Parameters
  ----------
    x_ou_mean
    The noise mean of the :math:`x` variable, [mV/ms]
  y_ou_mean
    The noise mean of the :math:`y` variable, [mV/ms].
  x_ou_sigma
    The noise intensity of the :math:`x` variable, [mV/ms/sqrt(ms)].
  y_ou_sigma
    The noise intensity of the :math:`y` variable, [mV/ms/sqrt(ms)].
  x_ou_tau
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`x` variable, [ms].
  y_ou_tau
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`y` variable, [ms].


  References
  ----------
  .. [5] E. Montbrió, D. Pazó, A. Roxin (2015) Macroscopic description for
         networks of spiking neurons. Physical Review X, 5:021028,
         https://doi.org/10.1103/PhysRevX.5.021028.
  .. [6] R. Gast, H. Schmidt, T.R. Knösche (2020) A Mean-Field Description
         of Bursting Dynamics in Spiking Neural Networks with Short-Term
         Adaptation. Neural Computation 32.9 (2020): 1615-1634.

  """

  def __init__(
      self,
      size: Shape,

      # model parameters
      tau: Parameter = 1.,
      eta: Parameter = -5.0,
      delta: Parameter = 1.0,
      J: Parameter = 15.,

      # noise parameters
      x_ou_mean: Parameter = 0.0,
      x_ou_sigma: Parameter = 0.0,
      x_ou_tau: Parameter = 5.0,
      y_ou_mean: Parameter = 0.0,
      y_ou_sigma: Parameter = 0.0,
      y_ou_tau: Parameter = 5.0,

      # other parameters
      method: str = 'exp_auto',
      name: str = None,
      sde_method: str = None,
  ):
    super(RateQIF, self).__init__(size=size, name=name)

    # parameters
    self.tau = tau  #
    self.eta = eta  # the mean of a Lorenzian distribution over the neural excitability in the population
    self.delta = delta  # the half-width at half maximum of the Lorenzian distribution over the neural excitability
    self.J = J  # the strength of the recurrent coupling inside the population

    # noise parameters
    self.x_ou_mean = x_ou_mean
    self.y_ou_mean = y_ou_mean
    self.x_ou_sigma = x_ou_sigma
    self.y_ou_sigma = y_ou_sigma
    self.x_ou_tau = x_ou_tau
    self.y_ou_tau = y_ou_tau

    # variables
    self.y = bm.Variable(bm.ones(self.num))
    self.x = bm.Variable(bm.ones(self.num))
    self.input = bm.Variable(bm.zeros(self.num))

    # noise variables
    self.x_ou = self.y_ou = None
    if bm.any(self.x_ou_mean > 0.) or bm.any(self.x_ou_sigma > 0.):
      self.x_ou = OUProcess(self.num,
                            self.x_ou_mean, self.x_ou_sigma, self.x_ou_tau,
                            method=sde_method)
    if bm.any(self.y_ou_mean > 0.) or bm.any(self.y_ou_sigma > 0.):
      self.y_ou = OUProcess(self.num,
                            self.y_ou_mean, self.y_ou_sigma, self.y_ou_tau,
                            method=sde_method)

    # functions
    self.integral = odeint(JointEq([self.dx, self.dy]), method=method)

  def dy(self, y, t, x, y_ext):
    return (self.delta / (bm.pi * self.tau) + 2. * x * y + y_ext) / self.tau

  def dx(self, x, t, y, x_ext):
    return (x ** 2 + self.eta + x_ext + self.J * y * self.tau -
            (bm.pi * y * self.tau) ** 2) / self.tau

  def update(self, _t, _dt):
    if self.x_ou is not None:
      self.input += self.x_ou.x
      self.x_ou.update(_t, _dt)
    y_ext = 0.
    if self.y_ou is not None:
      y_ext = self.y_ou.x
      self.y_ou.update(_t, _dt)
    x, y = self.integral(self.x, self.y, t=_t, x_ext=self.input, y_ext=y_ext, dt=_dt)
    self.x.value = x
    self.y.value = y
    self.input[:] = 0.


class StuartLandauOscillator(RateGroup):
  r"""
  Stuart-Landau model with Hopf bifurcation.

  .. math::

     \frac{dx}{dt} = (a - x^2 - y^2) * x - w*y + I^x_{ext} \\
     \frac{dy}{dt} = (a - x^2 - y^2) * y + w*x + I^y_{ext}

  Parameters
  ----------
    x_ou_mean
    The noise mean of the :math:`x` variable, [mV/ms]
  y_ou_mean
    The noise mean of the :math:`y` variable, [mV/ms].
  x_ou_sigma
    The noise intensity of the :math:`x` variable, [mV/ms/sqrt(ms)].
  y_ou_sigma
    The noise intensity of the :math:`y` variable, [mV/ms/sqrt(ms)].
  x_ou_tau
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`x` variable, [ms].
  y_ou_tau
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`y` variable, [ms].

  """

  def __init__(
      self,
      size: Shape,

      # model parameters
      a=0.25,
      w=0.2,

      # noise parameters
      x_ou_mean: Parameter = 0.0,
      x_ou_sigma: Parameter = 0.0,
      x_ou_tau: Parameter = 5.0,
      y_ou_mean: Parameter = 0.0,
      y_ou_sigma: Parameter = 0.0,
      y_ou_tau: Parameter = 5.0,

      # other parameters
      method: str = None,
      sde_method: str = None,
      name: str = None,
  ):
    super(StuartLandauOscillator, self).__init__(size=size,
                                                 name=name)

    # model parameters
    self.a = a
    self.w = w

    # noise parameters
    self.x_ou_mean = x_ou_mean
    self.y_ou_mean = y_ou_mean
    self.x_ou_sigma = x_ou_sigma
    self.y_ou_sigma = y_ou_sigma
    self.x_ou_tau = x_ou_tau
    self.y_ou_tau = y_ou_tau

    # variables
    self.x = bm.Variable(bm.random.random(self.num) * 0.5)
    self.y = bm.Variable(bm.random.random(self.num) * 0.5)
    self.input = bm.Variable(bm.zeros(self.num))

    # noise variables
    self.x_ou = self.y_ou = None
    if bm.any(self.x_ou_mean > 0.) or bm.any(self.x_ou_sigma > 0.):
      self.x_ou = OUProcess(self.num,
                            self.x_ou_mean, self.x_ou_sigma, self.x_ou_tau,
                            method=sde_method)
    if bm.any(self.y_ou_mean > 0.) or bm.any(self.y_ou_sigma > 0.):
      self.y_ou = OUProcess(self.num,
                            self.y_ou_mean, self.y_ou_sigma, self.y_ou_tau,
                            method=sde_method)

    # integral functions
    self.integral = odeint(f=JointEq([self.dx, self.dy]), method=method)

  def dx(self, x, t, y, x_ext, a, w):
    return (a - x * x - y * y) * x - w * y + x_ext

  def dy(self, y, t, x, y_ext, a, w):
    return (a - x * x - y * y) * y - w * y + y_ext

  def update(self, _t, _dt):
    if self.x_ou is not None:
      self.input += self.x_ou.x
      self.x_ou.update(_t, _dt)
    y_ext = 0.
    if self.y_ou is not None:
      y_ext = self.y_ou.x
      self.y_ou.update(_t, _dt)
    x, y = self.integral(self.x, self.y, _t, x_ext=self.input,
                         y_ext=y_ext, a=self.a, w=self.w, dt=_dt)
    self.x.value = x
    self.y.value = y
    self.input[:] = 0.


class WilsonCowanModel(RateGroup):
  """Wilson-Cowan population model.


  Parameters
  ----------
    x_ou_mean
    The noise mean of the :math:`x` variable, [mV/ms]
  y_ou_mean
    The noise mean of the :math:`y` variable, [mV/ms].
  x_ou_sigma
    The noise intensity of the :math:`x` variable, [mV/ms/sqrt(ms)].
  y_ou_sigma
    The noise intensity of the :math:`y` variable, [mV/ms/sqrt(ms)].
  x_ou_tau
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`x` variable, [ms].
  y_ou_tau
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`y` variable, [ms].


  """

  def __init__(
      self,
      size: Shape,

      # Excitatory parameters
      E_tau=2.5,  # excitatory time constant
      E_a=1.5,  # excitatory gain
      E_theta=3.0,  # excitatory firing threshold

      # Inhibitory parameters
      I_tau=3.75,  # inhibitory time constant
      I_a=1.5,  # inhibitory gain
      I_theta=3.0,  # inhibitory firing threshold

      # connection parameters
      wEE=16.,  # local E-E coupling
      wIE=15.,  # local E-I coupling
      wEI=12.,  # local I-E coupling
      wII=3.,  # local I-I coupling

      # Refractory parameter
      r=1,

      # noise parameters
      x_ou_mean: Parameter = 0.0,
      x_ou_sigma: Parameter = 0.0,
      x_ou_tau: Parameter = 5.0,
      y_ou_mean: Parameter = 0.0,
      y_ou_sigma: Parameter = 0.0,
      y_ou_tau: Parameter = 5.0,

      # other parameters
      sde_method: str = None,
      method: str = None,
      name: str = None,
  ):
    super(WilsonCowanModel, self).__init__(size=size, name=name)

    # model parameters
    self.E_tau = E_tau
    self.E_a = E_a
    self.E_theta = E_theta
    self.I_tau = I_tau
    self.I_a = I_a
    self.I_theta = I_theta
    self.wEE = wEE
    self.wIE = wIE
    self.wEI = wEI
    self.wII = wII
    self.r = r

    # noise parameters
    self.x_ou_mean = x_ou_mean
    self.y_ou_mean = y_ou_mean
    self.x_ou_sigma = x_ou_sigma
    self.y_ou_sigma = y_ou_sigma
    self.x_ou_tau = x_ou_tau
    self.y_ou_tau = y_ou_tau

    # variables
    self.x = bm.Variable(bm.random.random(self.num) * 0.05)
    self.y = bm.Variable(bm.random.random(self.num) * 0.05)
    self.input = bm.Variable(bm.zeros(self.num))

    # noise variables
    self.x_ou = self.y_ou = None
    if bm.any(self.x_ou_mean > 0.) or bm.any(self.x_ou_sigma > 0.):
      self.x_ou = OUProcess(self.num,
                            self.x_ou_mean, self.x_ou_sigma, self.x_ou_tau,
                            method=sde_method)
    if bm.any(self.y_ou_mean > 0.) or bm.any(self.y_ou_sigma > 0.):
      self.y_ou = OUProcess(self.num,
                            self.y_ou_mean, self.y_ou_sigma, self.y_ou_tau,
                            method=sde_method)

    # functions
    self.integral = odeint(f=JointEq([self.dx, self.dy]), method=method)

  # functions
  def F(self, x, a, theta):
    return 1 / (1 + bm.exp(-a * (x - theta)))

  def dx(self, x, t, y, x_ext):
    x = self.wEE * x - self.wIE * y + x_ext
    return (-x + (1 - self.r * x) * self.F(x, self.E_a, self.E_theta)) / self.E_tau

  def dy(self, y, t, x, y_ext):
    x = self.wEI * x - self.wII * y + y_ext
    return (-y + (1 - self.r * y) * self.F(x, self.I_a, self.I_theta)) / self.I_tau

  def update(self, _t, _dt):
    if self.x_ou is not None:
      self.input += self.x_ou.x
      self.x_ou.update(_t, _dt)
    y_ext = 0.
    if self.y_ou is not None:
      y_ext = self.y_ou.x
      self.y_ou.update(_t, _dt)
    x, y = self.integral(self.x, self.y, _t, x_ext=self.input, y_ext=y_ext, dt=_dt)
    self.x.value = x
    self.y.value = y
    self.input[:] = 0.


class JansenRitModel(RateGroup):
  pass


class KuramotoOscillator(RateGroup):
  pass


class ThetaNeuron(RateGroup):
  pass


class RateQIFWithSFA(RateGroup):
  pass


class VanDerPolOscillator(RateGroup):
  pass
