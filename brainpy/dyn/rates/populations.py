# -*- coding: utf-8 -*-

from typing import Union, Callable

import brainpy.math as bm
from brainpy import check
from brainpy.dyn.base import NeuGroup
from brainpy.dyn.others.noises import OUProcess
from brainpy.initialize import Initializer, Uniform, init_param, ZeroInit
from brainpy.integrators.dde import ddeint
from brainpy.integrators.joint_eq import JointEq
from brainpy.integrators.ode import odeint
from brainpy.tools.checking import check_float, check_initializer
from brainpy.tools.errors import check_error_in_jit
from brainpy.types import Shape, Tensor

__all__ = [
  'Population',
  'FHN',
  'FeedbackFHN',
  'QIF',
  'StuartLandauOscillator',
  'WilsonCowanModel',
  'ThresholdLinearModel',
]


class Population(NeuGroup):
  def update(self, t, dt):
    raise NotImplementedError


class FHN(NeuGroup):
  r"""FitzHugh-Nagumo system used in [1]_.

  .. math::

     \frac{dx}{dt} = -\alpha V^3 + \beta V^2 + \gamma V - w + I_{ext}\\
     \tau \frac{dy}{dt} = (V - \delta  - \epsilon w)

  Parameters
  ----------
  size: Shape
    The model size.
  x_ou_mean: Parameter
    The noise mean of the :math:`x` variable, [mV/ms]
  y_ou_mean: Parameter
    The noise mean of the :math:`y` variable, [mV/ms].
  x_ou_sigma: Parameter
    The noise intensity of the :math:`x` variable, [mV/ms/sqrt(ms)].
  y_ou_sigma: Parameter
    The noise intensity of the :math:`y` variable, [mV/ms/sqrt(ms)].
  x_ou_tau: Parameter
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`x` variable, [ms].
  y_ou_tau: Parameter
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
      alpha: Union[float, Tensor, Initializer, Callable] = 3.0,
      beta: Union[float, Tensor, Initializer, Callable] = 4.0,
      gamma: Union[float, Tensor, Initializer, Callable] = -1.5,
      delta: Union[float, Tensor, Initializer, Callable] = 0.0,
      epsilon: Union[float, Tensor, Initializer, Callable] = 0.5,
      tau: Union[float, Tensor, Initializer, Callable] = 20.0,

      # noise parameters
      x_ou_mean: Union[float, Tensor, Initializer, Callable] = 0.0,
      x_ou_sigma: Union[float, Tensor, Initializer, Callable] = 0.0,
      x_ou_tau: Union[float, Tensor, Initializer, Callable] = 5.0,
      y_ou_mean: Union[float, Tensor, Initializer, Callable] = 0.0,
      y_ou_sigma: Union[float, Tensor, Initializer, Callable] = 0.0,
      y_ou_tau: Union[float, Tensor, Initializer, Callable] = 5.0,

      # other parameters
      x_initializer: Union[Initializer, Callable, Tensor] = Uniform(0, 0.05),
      y_initializer: Union[Initializer, Callable, Tensor] = Uniform(0, 0.05),
      method: str = 'exp_auto',
      sde_method: str = None,
      keep_size: bool = False,
      name: str = None,
  ):
    super(FHN, self).__init__(size=size, name=name)

    # model parameters
    self.alpha = init_param(alpha, self.num, allow_none=False)
    self.beta = init_param(beta, self.num, allow_none=False)
    self.gamma = init_param(gamma, self.num, allow_none=False)
    self.delta = init_param(delta, self.num, allow_none=False)
    self.epsilon = init_param(epsilon, self.num, allow_none=False)
    self.tau = init_param(tau, self.num, allow_none=False)

    # noise parameters
    self.x_ou_mean = init_param(x_ou_mean, self.num, allow_none=False)  # mV/ms, OU process
    self.y_ou_mean = init_param(y_ou_mean, self.num, allow_none=False)  # mV/ms, OU process
    self.x_ou_sigma = init_param(x_ou_sigma, self.num, allow_none=False)  # mV/ms/sqrt(ms), noise intensity
    self.y_ou_sigma = init_param(y_ou_sigma, self.num, allow_none=False)  # mV/ms/sqrt(ms), noise intensity
    self.x_ou_tau = init_param(x_ou_tau, self.num,
                               allow_none=False)  # ms, timescale of the Ornstein-Uhlenbeck noise process
    self.y_ou_tau = init_param(y_ou_tau, self.num,
                               allow_none=False)  # ms, timescale of the Ornstein-Uhlenbeck noise process

    # initializers
    check_initializer(x_initializer, 'x_initializer')
    check_initializer(y_initializer, 'y_initializer')
    self._x_initializer = x_initializer
    self._y_initializer = y_initializer

    # variables
    self.x = bm.Variable(init_param(x_initializer, (self.num,)))
    self.y = bm.Variable(init_param(y_initializer, (self.num,)))
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

  def reset(self):
    self.x.value = init_param(self._x_initializer, (self.num,))
    self.y.value = init_param(self._y_initializer, (self.num,))
    self.input[:] = 0
    if self.x_ou is not None:
      self.x_ou.reset()
    if self.y_ou is not None:
      self.y_ou.reset()

  def dx(self, x, t, y, x_ext):
    return - self.alpha * x ** 3 + self.beta * x ** 2 + self.gamma * x - y + x_ext

  def dy(self, y, t, x, y_ext=0.):
    return (x - self.delta - self.epsilon * y) / self.tau + y_ext

  def update(self, t, dt):
    if self.x_ou is not None:
      self.input += self.x_ou.x
      self.x_ou.update(t, dt)
    y_ext = 0.
    if self.y_ou is not None:
      y_ext = self.y_ou.x
      self.y_ou.update(t, dt)
    x, y = self.integral(self.x, self.y, t, x_ext=self.input, y_ext=y_ext, dt=dt)
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
  x_ou_mean: Parameter
    The noise mean of the :math:`x` variable, [mV/ms]
  y_ou_mean: Parameter
    The noise mean of the :math:`y` variable, [mV/ms].
  x_ou_sigma: Parameter
    The noise intensity of the :math:`x` variable, [mV/ms/sqrt(ms)].
  y_ou_sigma: Parameter
    The noise intensity of the :math:`y` variable, [mV/ms/sqrt(ms)].
  x_ou_tau: Parameter
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`x` variable, [ms].
  y_ou_tau: Parameter
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
      a: Union[float, Tensor, Initializer, Callable] = 0.7,
      b: Union[float, Tensor, Initializer, Callable] = 0.8,
      delay: Union[float, Tensor, Initializer, Callable] = 10.,
      tau: Union[float, Tensor, Initializer, Callable] = 12.5,
      mu: Union[float, Tensor, Initializer, Callable] = 1.6886,
      v0: Union[float, Tensor, Initializer, Callable] = -1,

      # noise parameters
      x_ou_mean: Union[float, Tensor, Initializer, Callable] = 0.0,
      x_ou_sigma: Union[float, Tensor, Initializer, Callable] = 0.0,
      x_ou_tau: Union[float, Tensor, Initializer, Callable] = 5.0,
      y_ou_mean: Union[float, Tensor, Initializer, Callable] = 0.0,
      y_ou_sigma: Union[float, Tensor, Initializer, Callable] = 0.0,
      y_ou_tau: Union[float, Tensor, Initializer, Callable] = 5.0,

      # other parameters
      x_initializer: Union[Initializer, Callable, Tensor] = Uniform(0, 0.05),
      y_initializer: Union[Initializer, Callable, Tensor] = Uniform(0, 0.05),
      method: str = 'rk4',
      sde_method: str = None,
      name: str = None,
      keep_size: bool = False,
      dt: float = None
  ):
    super(FeedbackFHN, self).__init__(size=size, name=name)

    # dt
    self.dt = bm.get_dt() if dt is None else dt
    check_float(self.dt, 'dt', allow_none=False, min_bound=0., allow_int=False)

    # parameters
    self.a = init_param(a, self.num, allow_none=False)
    self.b = init_param(b, self.num, allow_none=False)
    self.delay = init_param(delay, self.num, allow_none=False)
    self.tau = init_param(tau, self.num, allow_none=False)
    self.mu = init_param(mu, self.num, allow_none=False)  # feedback strength
    self.v0 = init_param(v0, self.num, allow_none=False)  # resting potential

    # noise parameters
    self.x_ou_mean = init_param(x_ou_mean, self.num, allow_none=False)
    self.y_ou_mean = init_param(y_ou_mean, self.num, allow_none=False)
    self.x_ou_sigma = init_param(x_ou_sigma, self.num, allow_none=False)
    self.y_ou_sigma = init_param(y_ou_sigma, self.num, allow_none=False)
    self.x_ou_tau = init_param(x_ou_tau, self.num, allow_none=False)
    self.y_ou_tau = init_param(y_ou_tau, self.num, allow_none=False)

    # initializers
    check_initializer(x_initializer, 'x_initializer')
    check_initializer(y_initializer, 'y_initializer')
    self._x_initializer = x_initializer
    self._y_initializer = y_initializer

    # variables
    self.x = bm.Variable(init_param(x_initializer, (self.num,)))
    self.y = bm.Variable(init_param(y_initializer, (self.num,)))
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

  def reset(self):
    self.x.value = init_param(self._x_initializer, (self.num,))
    self.y.value = init_param(self._y_initializer, (self.num,))
    self.x_delay.reset(self.x, self.delay)
    self.input[:] = 0
    if self.x_ou is not None:
      self.x_ou.reset()
    if self.y_ou is not None:
      self.y_ou.reset()

  def dx(self, x, t, y, x_ext):
    return x - x * x * x / 3 - y + x_ext + self.mu * (self.x_delay(t - self.delay) - self.v0)

  def dy(self, y, t, x, y_ext):
    return (x + self.a - self.b * y + y_ext) / self.tau

  def _check_dt(self, dt):
    raise ValueError(f'The "dt" {dt} used in model running is '
                     f'not consistent with the "dt" {self.dt} '
                     f'used in model definition.')

  def update(self, t, dt):
    if check.is_checking():
      check_error_in_jit(not bm.isclose(dt, self.dt), self._check_dt, dt)
    if self.x_ou is not None:
      self.input += self.x_ou.x
      self.x_ou.update(t, dt)
    y_ext = 0.
    if self.y_ou is not None:
      y_ext = self.y_ou.x
      self.y_ou.update(t, dt)
    x, y = self.integral(self.x, self.y, t, x_ext=self.input, y_ext=y_ext, dt=dt)
    self.x.value = x
    self.y.value = y
    self.input[:] = 0.


class QIF(NeuGroup):
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
  x_ou_mean: Parameter
    The noise mean of the :math:`x` variable, [mV/ms]
  y_ou_mean: Parameter
    The noise mean of the :math:`y` variable, [mV/ms].
  x_ou_sigma: Parameter
    The noise intensity of the :math:`x` variable, [mV/ms/sqrt(ms)].
  y_ou_sigma: Parameter
    The noise intensity of the :math:`y` variable, [mV/ms/sqrt(ms)].
  x_ou_tau: Parameter
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`x` variable, [ms].
  y_ou_tau: Parameter
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
      tau: Union[float, Tensor, Initializer, Callable] = 1.,
      eta: Union[float, Tensor, Initializer, Callable] = -5.0,
      delta: Union[float, Tensor, Initializer, Callable] = 1.0,
      J: Union[float, Tensor, Initializer, Callable] = 15.,

      # noise parameters
      x_ou_mean: Union[float, Tensor, Initializer, Callable] = 0.0,
      x_ou_sigma: Union[float, Tensor, Initializer, Callable] = 0.0,
      x_ou_tau: Union[float, Tensor, Initializer, Callable] = 5.0,
      y_ou_mean: Union[float, Tensor, Initializer, Callable] = 0.0,
      y_ou_sigma: Union[float, Tensor, Initializer, Callable] = 0.0,
      y_ou_tau: Union[float, Tensor, Initializer, Callable] = 5.0,

      # other parameters
      x_initializer: Union[Initializer, Callable, Tensor] = Uniform(0, 0.05),
      y_initializer: Union[Initializer, Callable, Tensor] = Uniform(0, 0.05),
      method: str = 'exp_auto',
      name: str = None,
      keep_size: bool = False,
      sde_method: str = None,
  ):
    super(QIF, self).__init__(size=size, name=name)

    # parameters
    self.tau = init_param(tau, self.num, allow_none=False)
    # the mean of a Lorenzian distribution over the neural excitability in the population
    self.eta = init_param(eta, self.num, allow_none=False)
    # the half-width at half maximum of the Lorenzian distribution over the neural excitability
    self.delta = init_param(delta, self.num, allow_none=False)
    # the strength of the recurrent coupling inside the population
    self.J = init_param(J, self.num, allow_none=False)

    # noise parameters
    self.x_ou_mean = init_param(x_ou_mean, self.num, allow_none=False)
    self.y_ou_mean = init_param(y_ou_mean, self.num, allow_none=False)
    self.x_ou_sigma = init_param(x_ou_sigma, self.num, allow_none=False)
    self.y_ou_sigma = init_param(y_ou_sigma, self.num, allow_none=False)
    self.x_ou_tau = init_param(x_ou_tau, self.num, allow_none=False)
    self.y_ou_tau = init_param(y_ou_tau, self.num, allow_none=False)

    # initializers
    check_initializer(x_initializer, 'x_initializer')
    check_initializer(y_initializer, 'y_initializer')
    self._x_initializer = x_initializer
    self._y_initializer = y_initializer

    # variables
    self.x = bm.Variable(init_param(x_initializer, (self.num,)))
    self.y = bm.Variable(init_param(y_initializer, (self.num,)))
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

  def reset(self):
    self.x.value = init_param(self._x_initializer, (self.num,))
    self.y.value = init_param(self._y_initializer, (self.num,))
    self.input[:] = 0
    if self.x_ou is not None:
      self.x_ou.reset()
    if self.y_ou is not None:
      self.y_ou.reset()

  def dy(self, y, t, x, y_ext):
    return (self.delta / (bm.pi * self.tau) + 2. * x * y + y_ext) / self.tau

  def dx(self, x, t, y, x_ext):
    return (x ** 2 + self.eta + x_ext + self.J * y * self.tau -
            (bm.pi * y * self.tau) ** 2) / self.tau

  def update(self, t, dt):
    if self.x_ou is not None:
      self.input += self.x_ou.x
      self.x_ou.update(t, dt)
    y_ext = 0.
    if self.y_ou is not None:
      y_ext = self.y_ou.x
      self.y_ou.update(t, dt)
    x, y = self.integral(self.x, self.y, t=t, x_ext=self.input, y_ext=y_ext, dt=dt)
    self.x.value = x
    self.y.value = y
    self.input[:] = 0.


class StuartLandauOscillator(Population):
  r"""
  Stuart-Landau model with Hopf bifurcation.

  .. math::

     \frac{dx}{dt} = (a - x^2 - y^2) * x - w*y + I^x_{ext} \\
     \frac{dy}{dt} = (a - x^2 - y^2) * y + w*x + I^y_{ext}

  Parameters
  ----------
  x_ou_mean: Parameter
    The noise mean of the :math:`x` variable, [mV/ms]
  y_ou_mean: Parameter
    The noise mean of the :math:`y` variable, [mV/ms].
  x_ou_sigma: Parameter
    The noise intensity of the :math:`x` variable, [mV/ms/sqrt(ms)].
  y_ou_sigma: Parameter
    The noise intensity of the :math:`y` variable, [mV/ms/sqrt(ms)].
  x_ou_tau: Parameter
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`x` variable, [ms].
  y_ou_tau: Parameter
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`y` variable, [ms].

  """

  def __init__(
      self,
      size: Shape,

      # model parameters
      a: Union[float, Tensor, Initializer, Callable] = 0.25,
      w: Union[float, Tensor, Initializer, Callable] = 0.2,

      # noise parameters
      x_ou_mean: Union[float, Tensor, Initializer, Callable] = 0.0,
      x_ou_sigma: Union[float, Tensor, Initializer, Callable] = 0.0,
      x_ou_tau: Union[float, Tensor, Initializer, Callable] = 5.0,
      y_ou_mean: Union[float, Tensor, Initializer, Callable] = 0.0,
      y_ou_sigma: Union[float, Tensor, Initializer, Callable] = 0.0,
      y_ou_tau: Union[float, Tensor, Initializer, Callable] = 5.0,

      # other parameters
      x_initializer: Union[Initializer, Callable, Tensor] = Uniform(0, 0.5),
      y_initializer: Union[Initializer, Callable, Tensor] = Uniform(0, 0.5),
      method: str = 'exp_auto',
      keep_size: bool = False,
      sde_method: str = None,
      name: str = None,
  ):
    super(StuartLandauOscillator, self).__init__(size=size,
                                                 name=name)

    # model parameters
    self.a = init_param(a, self.num, allow_none=False)
    self.w = init_param(w, self.num, allow_none=False)

    # noise parameters
    self.x_ou_mean = init_param(x_ou_mean, self.num, allow_none=False)
    self.y_ou_mean = init_param(y_ou_mean, self.num, allow_none=False)
    self.x_ou_sigma = init_param(x_ou_sigma, self.num, allow_none=False)
    self.y_ou_sigma = init_param(y_ou_sigma, self.num, allow_none=False)
    self.x_ou_tau = init_param(x_ou_tau, self.num, allow_none=False)
    self.y_ou_tau = init_param(y_ou_tau, self.num, allow_none=False)

    # initializers
    check_initializer(x_initializer, 'x_initializer')
    check_initializer(y_initializer, 'y_initializer')
    self._x_initializer = x_initializer
    self._y_initializer = y_initializer

    # variables
    self.x = bm.Variable(init_param(x_initializer, (self.num,)))
    self.y = bm.Variable(init_param(y_initializer, (self.num,)))
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

  def reset(self):
    self.x.value = init_param(self._x_initializer, (self.num,))
    self.y.value = init_param(self._y_initializer, (self.num,))
    self.input[:] = 0
    if self.x_ou is not None:
      self.x_ou.reset()
    if self.y_ou is not None:
      self.y_ou.reset()

  def dx(self, x, t, y, x_ext, a, w):
    return (a - x * x - y * y) * x - w * y + x_ext

  def dy(self, y, t, x, y_ext, a, w):
    return (a - x * x - y * y) * y - w * y + y_ext

  def update(self, t, dt):
    if self.x_ou is not None:
      self.input += self.x_ou.x
      self.x_ou.update(t, dt)
    y_ext = 0.
    if self.y_ou is not None:
      y_ext = self.y_ou.x
      self.y_ou.update(t, dt)
    x, y = self.integral(self.x, self.y, t, x_ext=self.input,
                         y_ext=y_ext, a=self.a, w=self.w, dt=dt)
    self.x.value = x
    self.y.value = y
    self.input[:] = 0.


class WilsonCowanModel(Population):
  """Wilson-Cowan population model.


  Parameters
  ----------
  x_ou_mean: Parameter
    The noise mean of the :math:`x` variable, [mV/ms]
  y_ou_mean: Parameter
    The noise mean of the :math:`y` variable, [mV/ms].
  x_ou_sigma: Parameter
    The noise intensity of the :math:`x` variable, [mV/ms/sqrt(ms)].
  y_ou_sigma: Parameter
    The noise intensity of the :math:`y` variable, [mV/ms/sqrt(ms)].
  x_ou_tau: Parameter
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`x` variable, [ms].
  y_ou_tau: Parameter
    The timescale of the Ornstein-Uhlenbeck noise process of :math:`y` variable, [ms].


  """

  def __init__(
      self,
      size: Shape,

      # Excitatory parameters
      E_tau: Union[float, Tensor, Initializer, Callable] = 1.,  # excitatory time constant
      E_a: Union[float, Tensor, Initializer, Callable] = 1.2,  # excitatory gain
      E_theta: Union[float, Tensor, Initializer, Callable] = 2.8,  # excitatory firing threshold

      # Inhibitory parameters
      I_tau: Union[float, Tensor, Initializer, Callable] = 1.,  # inhibitory time constant
      I_a: Union[float, Tensor, Initializer, Callable] = 1.,  # inhibitory gain
      I_theta: Union[float, Tensor, Initializer, Callable] = 4.0,  # inhibitory firing threshold

      # connection parameters
      wEE: Union[float, Tensor, Initializer, Callable] = 12.,  # local E-E coupling
      wIE: Union[float, Tensor, Initializer, Callable] = 4.,  # local E-I coupling
      wEI: Union[float, Tensor, Initializer, Callable] = 13.,  # local I-E coupling
      wII: Union[float, Tensor, Initializer, Callable] = 11.,  # local I-I coupling

      # Refractory parameter
      r: Union[float, Tensor, Initializer, Callable] = 1.,

      # noise parameters
      x_ou_mean: Union[float, Tensor, Initializer, Callable] = 0.0,
      x_ou_sigma: Union[float, Tensor, Initializer, Callable] = 0.0,
      x_ou_tau: Union[float, Tensor, Initializer, Callable] = 5.0,
      y_ou_mean: Union[float, Tensor, Initializer, Callable] = 0.0,
      y_ou_sigma: Union[float, Tensor, Initializer, Callable] = 0.0,
      y_ou_tau: Union[float, Tensor, Initializer, Callable] = 5.0,

      # state initializer
      x_initializer: Union[Initializer, Callable, Tensor] = Uniform(max_val=0.05),
      y_initializer: Union[Initializer, Callable, Tensor] = Uniform(max_val=0.05),

      # other parameters
      sde_method: str = None,
      keep_size: bool = False,
      method: str = 'exp_euler_auto',
      name: str = None,
  ):
    super(WilsonCowanModel, self).__init__(size=size, name=name)

    # model parameters
    self.E_a = init_param(E_a, self.num, allow_none=False)
    self.I_a = init_param(I_a, self.num, allow_none=False)
    self.E_tau = init_param(E_tau, self.num, allow_none=False)
    self.I_tau = init_param(I_tau, self.num, allow_none=False)
    self.E_theta = init_param(E_theta, self.num, allow_none=False)
    self.I_theta = init_param(I_theta, self.num, allow_none=False)
    self.wEE = init_param(wEE, self.num, allow_none=False)
    self.wIE = init_param(wIE, self.num, allow_none=False)
    self.wEI = init_param(wEI, self.num, allow_none=False)
    self.wII = init_param(wII, self.num, allow_none=False)
    self.r = init_param(r, self.num, allow_none=False)

    # noise parameters
    self.x_ou_mean = init_param(x_ou_mean, self.num, allow_none=False)
    self.y_ou_mean = init_param(y_ou_mean, self.num, allow_none=False)
    self.x_ou_sigma = init_param(x_ou_sigma, self.num, allow_none=False)
    self.y_ou_sigma = init_param(y_ou_sigma, self.num, allow_none=False)
    self.x_ou_tau = init_param(x_ou_tau, self.num, allow_none=False)
    self.y_ou_tau = init_param(y_ou_tau, self.num, allow_none=False)

    # initializers
    check_initializer(x_initializer, 'x_initializer')
    check_initializer(y_initializer, 'y_initializer')
    self._x_initializer = x_initializer
    self._y_initializer = y_initializer

    # variables
    self.x = bm.Variable(init_param(x_initializer, (self.num,)))
    self.y = bm.Variable(init_param(y_initializer, (self.num,)))
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

  def reset(self):
    self.x.value = init_param(self._x_initializer, (self.num,))
    self.y.value = init_param(self._y_initializer, (self.num,))
    self.input[:] = 0
    if self.x_ou is not None:
      self.x_ou.reset()
    if self.y_ou is not None:
      self.y_ou.reset()

  def F(self, x, a, theta):
    return 1 / (1 + bm.exp(-a * (x - theta))) - 1 / (1 + bm.exp(a * theta))

  def dx(self, x, t, y, x_ext):
    x = self.wEE * x - self.wIE * y + x_ext
    return (-x + (1 - self.r * x) * self.F(x, self.E_a, self.E_theta)) / self.E_tau

  def dy(self, y, t, x, y_ext):
    x = self.wEI * x - self.wII * y + y_ext
    return (-y + (1 - self.r * y) * self.F(x, self.I_a, self.I_theta)) / self.I_tau

  def update(self, t, dt):
    if self.x_ou is not None:
      self.input += self.x_ou.x
      self.x_ou.update(t, dt)
    y_ext = 0.
    if self.y_ou is not None:
      y_ext = self.y_ou.x
      self.y_ou.update(t, dt)
    x, y = self.integral(self.x, self.y, t, x_ext=self.input, y_ext=y_ext, dt=dt)
    self.x.value = x
    self.y.value = y
    self.input[:] = 0.


class JansenRitModel(Population):
  pass


class KuramotoOscillator(Population):
  pass


class ThetaNeuron(Population):
  pass


class RateQIFWithSFA(Population):
  pass


class VanDerPolOscillator(Population):
  pass


class ThresholdLinearModel(Population):
  r"""A threshold linear rate model.

  The threshold linear rate model is given by [1]_

  .. math::

     \begin{aligned}
      &\tau_{E} \frac{d \nu_{E}}{d t}=-\nu_{E}+\beta_{E}\left[I_{E}\right]_{+} \\
      &\tau_{I} \frac{d \nu_{I}}{d t}=-\nu_{I}+\beta_{I}\left[I_{I}\right]_{+}
      \end{aligned}

  where :math:`\left[I_{E}\right]_{+}=\max \left(I_{E}, 0\right)`.
  :math:`v_E` and :math:`v_I` denote the firing rates of the excitatory and inhibitory
  populations respectively, :math:`\tau_E` and :math:`\tau_I` are the corresponding
  intrinsic time constants.


  Reference
  ---------
  .. [1] Chaudhuri, Rishidev, et al. "A large-scale circuit mechanism
         for hierarchical dynamical processing in the primate cortex."
         Neuron 88.2 (2015): 419-431.

  """

  def __init__(
      self,
      size: Shape,
      tau_e: Union[float, Callable, Initializer, Tensor] = 2e-2,
      tau_i: Union[float, Callable, Initializer, Tensor] = 1e-2,
      beta_e: Union[float, Callable, Initializer, Tensor] = .066,
      beta_i: Union[float, Callable, Initializer, Tensor] = .351,
      noise_e: Union[float, Callable, Initializer, Tensor] = 0.,
      noise_i: Union[float, Callable, Initializer, Tensor] = 0.,
      e_initializer: Union[Tensor, Callable, Initializer] = ZeroInit(),
      i_initializer: Union[Tensor, Callable, Initializer] = ZeroInit(),
      seed: int = None,
      keep_size: bool = False,
      name: str = None
  ):
    super(ThresholdLinearModel, self).__init__(size, name=name)

    # parameters
    self.seed = seed
    self.tau_e = init_param(tau_e, self.num, False)
    self.tau_i = init_param(tau_i, self.num, False)
    self.beta_e = init_param(beta_e, self.num, False)
    self.beta_i = init_param(beta_i, self.num, False)
    self.noise_e = init_param(noise_e, self.num, False)
    self.noise_i = init_param(noise_i, self.num, False)
    self._e_initializer = e_initializer
    self._i_initializer = i_initializer

    # variables
    self.e = bm.Variable(init_param(e_initializer, self.num))  # Firing rate of excitatory population
    self.i = bm.Variable(init_param(i_initializer, self.num))  # Firing rate of inhibitory population
    self.Ie = bm.Variable(bm.zeros(self.num))  # Input of excitaory population
    self.Ii = bm.Variable(bm.zeros(self.num))  # Input of inhibitory population
    if bm.any(self.noise_e != 0) or bm.any(self.noise_i != 0):
      self.rng = bm.random.RandomState(self.seed)

  def reset(self):
    self.rng.seed(self.seed)
    self.e.value = init_param(self._e_initializer, self.num)
    self.i.value = init_param(self._i_initializer, self.num)
    self.Ie[:] = 0.
    self.Ii[:] = 0.

  def update(self, t, dt):
    de = -self.e + self.beta_e * bm.maximum(self.Ie, 0.)
    if bm.any(self.noise_e != 0.):
      de += self.rng.randn(self.num) * self.noise_e
    de = de / self.tau_e
    self.e.value = bm.maximum(self.e + de * dt, 0.)
    di = -self.i + self.beta_i * bm.maximum(self.Ii, 0.)
    if bm.any(self.noise_i != 0.):
      di += self.rng.randn(self.num) * self.noise_i
    di = di / self.tau_i
    self.i.value = bm.maximum(self.i + di * dt, 0.)
    self.Ie[:] = 0.
    self.Ii[:] = 0.
