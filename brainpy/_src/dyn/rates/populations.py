# -*- coding: utf-8 -*-

from typing import Union, Callable

import jax

from brainpy import math as bm
from brainpy._src.context import share
from brainpy._src.dyn.others.noise import OUProcess
from brainpy._src.dyn.base import NeuDyn
from brainpy._src.initialize import (Initializer,
                                     Uniform,
                                     parameter,
                                     variable,
                                     variable_,
                                     ZeroInit)
from brainpy._src.integrators.joint_eq import JointEq
from brainpy._src.integrators.ode.generic import odeint
from brainpy.check import is_initializer
from brainpy.types import Shape, ArrayType

__all__ = [
  'FHN',
  'FeedbackFHN',
  'QIF',
  'StuartLandauOscillator',
  'WilsonCowanModel',
  'ThresholdLinearModel',
]


class RateModel(NeuDyn):
  pass


class FHN(RateModel):
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
      keep_size: bool = False,

      # fhn parameters
      alpha: Union[float, ArrayType, Initializer, Callable] = 3.0,
      beta: Union[float, ArrayType, Initializer, Callable] = 4.0,
      gamma: Union[float, ArrayType, Initializer, Callable] = -1.5,
      delta: Union[float, ArrayType, Initializer, Callable] = 0.0,
      epsilon: Union[float, ArrayType, Initializer, Callable] = 0.5,
      tau: Union[float, ArrayType, Initializer, Callable] = 20.0,

      # noise parameters
      x_ou_mean: Union[float, ArrayType, Initializer, Callable] = 0.0,
      x_ou_sigma: Union[float, ArrayType, Initializer, Callable] = 0.0,
      x_ou_tau: Union[float, ArrayType, Initializer, Callable] = 5.0,
      y_ou_mean: Union[float, ArrayType, Initializer, Callable] = 0.0,
      y_ou_sigma: Union[float, ArrayType, Initializer, Callable] = 0.0,
      y_ou_tau: Union[float, ArrayType, Initializer, Callable] = 5.0,

      # other parameters
      x_initializer: Union[Initializer, Callable, ArrayType] = Uniform(0, 0.05),
      y_initializer: Union[Initializer, Callable, ArrayType] = Uniform(0, 0.05),
      method: str = 'exp_auto',
      name: str = None,

      # parameter for training
      mode: bm.Mode = None,
      input_var: bool = True,
  ):
    super().__init__(size=size,
                              name=name,
                              keep_size=keep_size,
                              mode=mode)

    # model parameters
    self.alpha = parameter(alpha, self.varshape, allow_none=False)
    self.beta = parameter(beta, self.varshape, allow_none=False)
    self.gamma = parameter(gamma, self.varshape, allow_none=False)
    self.delta = parameter(delta, self.varshape, allow_none=False)
    self.epsilon = parameter(epsilon, self.varshape, allow_none=False)
    self.tau = parameter(tau, self.varshape, allow_none=False)

    # noise parameters
    self.x_ou_mean = parameter(x_ou_mean, self.varshape, allow_none=False)  # mV/ms, OU process
    self.y_ou_mean = parameter(y_ou_mean, self.varshape, allow_none=False)  # mV/ms, OU process
    self.x_ou_sigma = parameter(x_ou_sigma, self.varshape, allow_none=False)  # mV/ms/sqrt(ms), noise intensity
    self.y_ou_sigma = parameter(y_ou_sigma, self.varshape, allow_none=False)  # mV/ms/sqrt(ms), noise intensity
    self.x_ou_tau = parameter(x_ou_tau, self.varshape,
                              allow_none=False)  # ms, timescale of the Ornstein-Uhlenbeck noise process
    self.y_ou_tau = parameter(y_ou_tau, self.varshape,
                              allow_none=False)  # ms, timescale of the Ornstein-Uhlenbeck noise process
    self.input_var = input_var

    # initializers
    is_initializer(x_initializer, 'x_initializer')
    is_initializer(y_initializer, 'y_initializer')
    self._x_initializer = x_initializer
    self._y_initializer = y_initializer

    # variables
    self.x = variable_(self._x_initializer, self.varshape, self.mode)
    self.y = variable_(self._y_initializer, self.varshape, self.mode)
    if self.input_var:
      self.input = variable_(bm.zeros, self.varshape, self.mode)
      self.input_y = variable_(bm.zeros, self.varshape, self.mode)

    # noise variables
    self.x_ou = self.y_ou = None
    if bm.any(self.x_ou_mean > 0.) or bm.any(self.x_ou_sigma > 0.):
      self.x_ou = OUProcess(self.varshape,
                            self.x_ou_mean,
                            self.x_ou_sigma,
                            self.x_ou_tau,
                            method=method)
    if bm.any(self.y_ou_mean > 0.) or bm.any(self.y_ou_sigma > 0.):
      self.y_ou = OUProcess(self.varshape,
                            self.y_ou_mean,
                            self.y_ou_sigma,
                            self.y_ou_tau,
                            method=method)

    # integral functions
    self.integral = odeint(f=JointEq(self.dx, self.dy), method=method)

  def reset_state(self, batch_size=None):
    self.x.value = variable(self._x_initializer, batch_size, self.varshape)
    self.y.value = variable(self._y_initializer, batch_size, self.varshape)
    if self.input_var:
      self.input.value = variable(bm.zeros, batch_size, self.varshape)
      self.input_y.value = variable(bm.zeros, batch_size, self.varshape)
    if self.x_ou is not None:
      self.x_ou.reset_state(batch_size)
    if self.y_ou is not None:
      self.y_ou.reset_state(batch_size)

  def dx(self, x, t, y, x_ext):
    return - self.alpha * x ** 3 + self.beta * x ** 2 + self.gamma * x - y + x_ext

  def dy(self, y, t, x, y_ext=0.):
    return (x - self.delta - self.epsilon * y) / self.tau + y_ext

  def update(self, inp_x=None, inp_y=None):
    t = share.load('t')
    dt = share.load('dt')

    # input
    if self.input_var:
      if inp_x is not None:
        self.input += inp_x
      if self.x_ou is not None:
        self.input += self.x_ou()
      if inp_y is not None:
        self.input_y += inp_y
      if self.y_ou is not None:
        self.input_y += self.y_ou()
      input_x = self.input.value
      input_y = self.input_y.value
    else:
      input_x = inp_x if (inp_x is not None) else 0.
      if self.x_ou is not None: input_x += self.x_ou()
      input_y = inp_y if (inp_y is not None) else 0.
      if self.y_ou is not None: input_y += self.y_ou()

    # integral
    x, y = self.integral(self.x.value, self.y.value, t, x_ext=input_x, y_ext=input_y, dt=dt)
    self.x.value = x
    self.y.value = y
    return x

  def clear_input(self):
    if self.input_var:
      self.input.value = bm.zeros_like(self.input)
      self.input_y.value = bm.zeros_like(self.input_y)


class FeedbackFHN(RateModel):
  r"""FitzHugh-Nagumo model with recurrent neural feedback.

  The equation of the feedback FitzHugh-Nagumo model [4]_ is given by

  .. math::

     \begin{aligned}
     \frac{dx}{dt} &= x(t) - \frac{x^3(t)}{3} - y(t) + \mu[x(t-\mathrm{delay}) - x_0] \\
     \frac{dy}{dt} &= [x(t) + a - b y(t)] / \tau
     \end{aligned}


  **Model Examples**

  >>> import brainpy as bp
  >>> fhn = bp.rates.FeedbackFHN(1, delay=10.)
  >>> runner = bp.DSRunner(fhn, inputs=('input', 1.), monitors=['x', 'y'])
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
      keep_size: bool = False,

      # model parameters
      a: Union[float, ArrayType, Initializer, Callable] = 0.7,
      b: Union[float, ArrayType, Initializer, Callable] = 0.8,
      delay: Union[float, ArrayType, Initializer, Callable] = 10.,
      tau: Union[float, ArrayType, Initializer, Callable] = 12.5,
      mu: Union[float, ArrayType, Initializer, Callable] = 1.6886,
      v0: Union[float, ArrayType, Initializer, Callable] = -1,

      # noise parameters
      x_ou_mean: Union[float, ArrayType, Initializer, Callable] = 0.0,
      x_ou_sigma: Union[float, ArrayType, Initializer, Callable] = 0.0,
      x_ou_tau: Union[float, ArrayType, Initializer, Callable] = 5.0,
      y_ou_mean: Union[float, ArrayType, Initializer, Callable] = 0.0,
      y_ou_sigma: Union[float, ArrayType, Initializer, Callable] = 0.0,
      y_ou_tau: Union[float, ArrayType, Initializer, Callable] = 5.0,

      # other parameters
      x_initializer: Union[Initializer, Callable, ArrayType] = Uniform(0, 0.05),
      y_initializer: Union[Initializer, Callable, ArrayType] = Uniform(0, 0.05),
      method: str = 'exp_auto',
      name: str = None,

      # parameter for training
      mode: bm.Mode = None,
      input_var: bool = True,
  ):
    super(FeedbackFHN, self).__init__(size=size,
                                      name=name,
                                      keep_size=keep_size,
                                      mode=mode)

    # parameters
    self.a = parameter(a, self.varshape, allow_none=False)
    self.b = parameter(b, self.varshape, allow_none=False)
    self.delay = parameter(delay, self.varshape, allow_none=False)
    self.tau = parameter(tau, self.varshape, allow_none=False)
    self.mu = parameter(mu, self.varshape, allow_none=False)  # feedback strength
    self.v0 = parameter(v0, self.varshape, allow_none=False)  # resting potential

    # noise parameters
    self.x_ou_mean = parameter(x_ou_mean, self.varshape, allow_none=False)
    self.y_ou_mean = parameter(y_ou_mean, self.varshape, allow_none=False)
    self.x_ou_sigma = parameter(x_ou_sigma, self.varshape, allow_none=False)
    self.y_ou_sigma = parameter(y_ou_sigma, self.varshape, allow_none=False)
    self.x_ou_tau = parameter(x_ou_tau, self.varshape, allow_none=False)
    self.y_ou_tau = parameter(y_ou_tau, self.varshape, allow_none=False)
    self.input_var = input_var

    # initializers
    is_initializer(x_initializer, 'x_initializer')
    is_initializer(y_initializer, 'y_initializer')
    self._x_initializer = x_initializer
    self._y_initializer = y_initializer

    # variables
    self.x = variable(x_initializer, self.mode, self.varshape)
    self.y = variable(y_initializer, self.mode, self.varshape)
    self.x_delay = bm.TimeDelay(self.x, self.delay, dt=bm.dt, interp_method='round')
    if self.input_var:
      self.input = variable(bm.zeros, self.mode, self.varshape)
      self.input_y = variable(bm.zeros, self.mode, self.varshape)

    # noise variables
    self.x_ou = self.y_ou = None
    if bm.any(self.x_ou_mean > 0.) or bm.any(self.x_ou_sigma > 0.):
      self.x_ou = OUProcess(self.varshape,
                            self.x_ou_mean,
                            self.x_ou_sigma,
                            self.x_ou_tau,
                            method=method)
    if bm.any(self.y_ou_mean > 0.) or bm.any(self.y_ou_sigma > 0.):
      self.y_ou = OUProcess(self.varshape,
                            self.y_ou_mean,
                            self.y_ou_sigma,
                            self.y_ou_tau,
                            method=method)

    # integral
    self.integral = odeint(method=method,
                           f=JointEq([self.dx, self.dy]),
                           state_delays={'x': self.x_delay})

  def reset_state(self, batch_size=None):
    self.x.value = variable(self._x_initializer, batch_size, self.varshape)
    self.y.value = variable(self._y_initializer, batch_size, self.varshape)
    self.x_delay.reset(self.x, self.delay)
    if self.input_var:
      self.input = variable(bm.zeros, batch_size, self.varshape)
      self.input_y = variable(bm.zeros, batch_size, self.varshape)
    if self.x_ou is not None:
      self.x_ou.reset_state(batch_size)
    if self.y_ou is not None:
      self.y_ou.reset_state(batch_size)

  def dx(self, x, t, y, x_ext):
    return x - x * x * x / 3 - y + x_ext + self.mu * (self.x_delay(t - self.delay) - self.v0)

  def dy(self, y, t, x, y_ext):
    return (x + self.a - self.b * y + y_ext) / self.tau

  def update(self, inp_x=None, inp_y=None):
    t = share.load('t')
    dt = share.load('dt')

    # input
    if self.input_var:
      if inp_x is not None:
        self.input += inp_x
      if self.x_ou is not None:
        self.input += self.x_ou()
      if inp_y is not None:
        self.input_y += inp_y
      if self.y_ou is not None:
        self.input_y += self.y_ou()
      input_x = self.input.value
      input_y = self.input_y.value
    else:
      input_x = inp_x if (inp_x is not None) else 0.
      if self.x_ou is not None: input_x += self.x_ou()
      input_y = inp_y if (inp_y is not None) else 0.
      if self.y_ou is not None: input_y += self.y_ou()

    x, y = self.integral(self.x.value, self.y.value, t, x_ext=input_x, y_ext=input_y, dt=dt)
    self.x.value = x
    self.y.value = y
    return x

  def clear_input(self):
    if self.input_var:
      self.input.value = bm.zeros_like(self.input)
      self.input_y.value = bm.zeros_like(self.input_y)


class QIF(RateModel):
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
  excitability. While the mean-field derivation is mathematically
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
      keep_size: bool = False,

      # model parameters
      tau: Union[float, ArrayType, Initializer, Callable] = 1.,
      eta: Union[float, ArrayType, Initializer, Callable] = -5.0,
      delta: Union[float, ArrayType, Initializer, Callable] = 1.0,
      J: Union[float, ArrayType, Initializer, Callable] = 15.,

      # noise parameters
      x_ou_mean: Union[float, ArrayType, Initializer, Callable] = 0.0,
      x_ou_sigma: Union[float, ArrayType, Initializer, Callable] = 0.0,
      x_ou_tau: Union[float, ArrayType, Initializer, Callable] = 5.0,
      y_ou_mean: Union[float, ArrayType, Initializer, Callable] = 0.0,
      y_ou_sigma: Union[float, ArrayType, Initializer, Callable] = 0.0,
      y_ou_tau: Union[float, ArrayType, Initializer, Callable] = 5.0,

      # other parameters
      x_initializer: Union[Initializer, Callable, ArrayType] = Uniform(0, 0.05),
      y_initializer: Union[Initializer, Callable, ArrayType] = Uniform(0, 0.05),
      method: str = 'exp_auto',
      name: str = None,
      input_var: bool = True,

      # parameter for training
      mode: bm.Mode = None,
  ):
    super(QIF, self).__init__(size=size,
                              name=name,
                              keep_size=keep_size,
                              mode=mode)

    # parameters
    self.tau = parameter(tau, self.varshape, allow_none=False)
    # the mean of a Lorenzian distribution over the neural excitability in the population
    self.eta = parameter(eta, self.varshape, allow_none=False)
    # the half-width at half maximum of the Lorenzian distribution over the neural excitability
    self.delta = parameter(delta, self.varshape, allow_none=False)
    # the strength of the recurrent coupling inside the population
    self.J = parameter(J, self.varshape, allow_none=False)

    # noise parameters
    self.x_ou_mean = parameter(x_ou_mean, self.varshape, allow_none=False)
    self.y_ou_mean = parameter(y_ou_mean, self.varshape, allow_none=False)
    self.x_ou_sigma = parameter(x_ou_sigma, self.varshape, allow_none=False)
    self.y_ou_sigma = parameter(y_ou_sigma, self.varshape, allow_none=False)
    self.x_ou_tau = parameter(x_ou_tau, self.varshape, allow_none=False)
    self.y_ou_tau = parameter(y_ou_tau, self.varshape, allow_none=False)
    self.input_var = input_var

    # initializers
    is_initializer(x_initializer, 'x_initializer')
    is_initializer(y_initializer, 'y_initializer')
    self._x_initializer = x_initializer
    self._y_initializer = y_initializer

    # variables
    self.x = variable(x_initializer, self.mode, self.varshape)
    self.y = variable(y_initializer, self.mode, self.varshape)
    if self.input_var:
      self.input = variable(bm.zeros, self.mode, self.varshape)
      self.input_y = variable(bm.zeros, self.mode, self.varshape)

    # noise variables
    self.x_ou = self.y_ou = None
    if bm.any(self.x_ou_mean > 0.) or bm.any(self.x_ou_sigma > 0.):
      self.x_ou = OUProcess(self.varshape,
                            self.x_ou_mean,
                            self.x_ou_sigma,
                            self.x_ou_tau,
                            method=method)
    if bm.any(self.y_ou_mean > 0.) or bm.any(self.y_ou_sigma > 0.):
      self.y_ou = OUProcess(self.varshape,
                            self.y_ou_mean,
                            self.y_ou_sigma,
                            self.y_ou_tau,
                            method=method)

    # functions
    self.integral = odeint(JointEq([self.dx, self.dy]), method=method)

  def reset_state(self, batch_size=None):
    self.x.value = variable(self._x_initializer, batch_size, self.varshape)
    self.y.value = variable(self._y_initializer, batch_size, self.varshape)
    if self.input_var:
      self.input.value = variable(bm.zeros, batch_size, self.varshape)
      self.input_y.value = variable(bm.zeros, batch_size, self.varshape)
    if self.x_ou is not None:
      self.x_ou.reset_state(batch_size)
    if self.y_ou is not None:
      self.y_ou.reset_state(batch_size)

  def dy(self, y, t, x, y_ext):
    return (self.delta / (bm.pi * self.tau) + 2. * x * y + y_ext) / self.tau

  def dx(self, x, t, y, x_ext):
    return (x ** 2 + self.eta + x_ext + self.J * y * self.tau -
            (bm.pi * y * self.tau) ** 2) / self.tau

  def update(self, inp_x=None, inp_y=None):
    t = share.load('t')
    dt = share.load('dt')

    # input
    if self.input_var:
      if inp_x is not None:
        self.input += inp_x
      if self.x_ou is not None:
        self.input += self.x_ou()
      if inp_y is not None:
        self.input_y += inp_y
      if self.y_ou is not None:
        self.input_y += self.y_ou()
      input_x = self.input.value
      input_y = self.input_y.value
    else:
      input_x = inp_x if (inp_x is not None) else 0.
      if self.x_ou is not None: input_x += self.x_ou()
      input_y = inp_y if (inp_y is not None) else 0.
      if self.y_ou is not None: input_y += self.y_ou()

    x, y = self.integral(self.x, self.y, t=t, x_ext=input_x, y_ext=input_y, dt=dt)
    self.x.value = x
    self.y.value = y
    return x

  def clear_input(self):
    if self.input_var:
      self.input.value = bm.zeros_like(self.input)
      self.input_y.value = bm.zeros_like(self.input_y)


class StuartLandauOscillator(RateModel):
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
      keep_size: bool = False,

      # model parameters
      a: Union[float, ArrayType, Initializer, Callable] = 0.25,
      w: Union[float, ArrayType, Initializer, Callable] = 0.2,

      # noise parameters
      x_ou_mean: Union[float, ArrayType, Initializer, Callable] = 0.0,
      x_ou_sigma: Union[float, ArrayType, Initializer, Callable] = 0.0,
      x_ou_tau: Union[float, ArrayType, Initializer, Callable] = 5.0,
      y_ou_mean: Union[float, ArrayType, Initializer, Callable] = 0.0,
      y_ou_sigma: Union[float, ArrayType, Initializer, Callable] = 0.0,
      y_ou_tau: Union[float, ArrayType, Initializer, Callable] = 5.0,

      # other parameters
      x_initializer: Union[Initializer, Callable, ArrayType] = Uniform(0, 0.5),
      y_initializer: Union[Initializer, Callable, ArrayType] = Uniform(0, 0.5),
      method: str = 'exp_auto',
      name: str = None,

      # parameter for training
      mode: bm.Mode = None,
      input_var: bool = True,
  ):
    super(StuartLandauOscillator, self).__init__(size=size,
                                                 name=name,
                                                 keep_size=keep_size,
                                                 mode=mode)

    # model parameters
    self.a = parameter(a, self.varshape, allow_none=False)
    self.w = parameter(w, self.varshape, allow_none=False)

    # noise parameters
    self.x_ou_mean = parameter(x_ou_mean, self.varshape, allow_none=False)
    self.y_ou_mean = parameter(y_ou_mean, self.varshape, allow_none=False)
    self.x_ou_sigma = parameter(x_ou_sigma, self.varshape, allow_none=False)
    self.y_ou_sigma = parameter(y_ou_sigma, self.varshape, allow_none=False)
    self.x_ou_tau = parameter(x_ou_tau, self.varshape, allow_none=False)
    self.y_ou_tau = parameter(y_ou_tau, self.varshape, allow_none=False)
    self.input_var = input_var

    # initializers
    is_initializer(x_initializer, 'x_initializer')
    is_initializer(y_initializer, 'y_initializer')
    self._x_initializer = x_initializer
    self._y_initializer = y_initializer

    # variables
    self.x = variable(x_initializer, self.mode, self.varshape)
    self.y = variable(y_initializer, self.mode, self.varshape)
    if input_var:
      self.input = variable(bm.zeros, self.mode, self.varshape)
      self.input_y = variable(bm.zeros, self.mode, self.varshape)

    # noise variables
    self.x_ou = self.y_ou = None
    if bm.any(self.x_ou_mean > 0.) or bm.any(self.x_ou_sigma > 0.):
      self.x_ou = OUProcess(self.varshape,
                            self.x_ou_mean,
                            self.x_ou_sigma,
                            self.x_ou_tau,
                            method=method)
    if bm.any(self.y_ou_mean > 0.) or bm.any(self.y_ou_sigma > 0.):
      self.y_ou = OUProcess(self.varshape,
                            self.y_ou_mean,
                            self.y_ou_sigma,
                            self.y_ou_tau,
                            method=method)

    # integral functions
    self.integral = odeint(f=JointEq([self.dx, self.dy]), method=method)

  def reset_state(self, batch_size=None):
    self.x.value = variable(self._x_initializer, batch_size, self.varshape)
    self.y.value = variable(self._y_initializer, batch_size, self.varshape)
    if self.input_var:
      self.input.value = variable(bm.zeros, batch_size, self.varshape)
      self.input_y.value = variable(bm.zeros, batch_size, self.varshape)
    if self.x_ou is not None:
      self.x_ou.reset_state(batch_size)
    if self.y_ou is not None:
      self.y_ou.reset_state(batch_size)

  def dx(self, x, t, y, x_ext, a, w):
    return (a - x * x - y * y) * x - w * y + x_ext

  def dy(self, y, t, x, y_ext, a, w):
    return (a - x * x - y * y) * y - w * y + y_ext

  def update(self, inp_x=None, inp_y=None):
    t = share.load('t')
    dt = share.load('dt')

    # input
    if self.input_var:
      if inp_x is not None:
        self.input += inp_x
      if self.x_ou is not None:
        self.input += self.x_ou()
      if inp_y is not None:
        self.input_y += inp_y
      if self.y_ou is not None:
        self.input_y += self.y_ou()
      input_x = self.input.value
      input_y = self.input_y.value
    else:
      input_x = inp_x if (inp_x is not None) else 0.
      if self.x_ou is not None: input_x += self.x_ou()
      input_y = inp_y if (inp_y is not None) else 0.
      if self.y_ou is not None: input_y += self.y_ou()

    x, y = self.integral(self.x,
                         self.y,
                         t=t,
                         x_ext=input_x,
                         y_ext=input_y,
                         a=self.a,
                         w=self.w,
                         dt=dt)
    self.x.value = x
    self.y.value = y
    return x

  def clear_input(self):
    if self.input_var:
      self.input.value = bm.zeros_like(self.input)
      self.input_y.value = bm.zeros_like(self.input_y)


class WilsonCowanModel(RateModel):
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
      keep_size: bool = False,

      # Excitatory parameters
      E_tau: Union[float, ArrayType, Initializer, Callable] = 1.,  # excitatory time constant
      E_a: Union[float, ArrayType, Initializer, Callable] = 1.2,  # excitatory gain
      E_theta: Union[float, ArrayType, Initializer, Callable] = 2.8,  # excitatory firing threshold

      # Inhibitory parameters
      I_tau: Union[float, ArrayType, Initializer, Callable] = 1.,  # inhibitory time constant
      I_a: Union[float, ArrayType, Initializer, Callable] = 1.,  # inhibitory gain
      I_theta: Union[float, ArrayType, Initializer, Callable] = 4.0,  # inhibitory firing threshold

      # connection parameters
      wEE: Union[float, ArrayType, Initializer, Callable] = 12.,  # local E-E coupling
      wIE: Union[float, ArrayType, Initializer, Callable] = 4.,  # local E-I coupling
      wEI: Union[float, ArrayType, Initializer, Callable] = 13.,  # local I-E coupling
      wII: Union[float, ArrayType, Initializer, Callable] = 11.,  # local I-I coupling

      # Refractory parameter
      r: Union[float, ArrayType, Initializer, Callable] = 1.,

      # noise parameters
      x_ou_mean: Union[float, ArrayType, Initializer, Callable] = 0.0,
      x_ou_sigma: Union[float, ArrayType, Initializer, Callable] = 0.0,
      x_ou_tau: Union[float, ArrayType, Initializer, Callable] = 5.0,
      y_ou_mean: Union[float, ArrayType, Initializer, Callable] = 0.0,
      y_ou_sigma: Union[float, ArrayType, Initializer, Callable] = 0.0,
      y_ou_tau: Union[float, ArrayType, Initializer, Callable] = 5.0,

      # state initializer
      x_initializer: Union[Initializer, Callable, ArrayType] = Uniform(max_val=0.05),
      y_initializer: Union[Initializer, Callable, ArrayType] = Uniform(max_val=0.05),

      # other parameters
      method: str = 'exp_euler_auto',
      name: str = None,

      # parameter for training
      mode: bm.Mode = None,
      input_var: bool = True,
  ):
    super(WilsonCowanModel, self).__init__(size=size, name=name, keep_size=keep_size, mode=mode)

    # model parameters
    self.E_a = parameter(E_a, self.varshape, allow_none=False)
    self.I_a = parameter(I_a, self.varshape, allow_none=False)
    self.E_tau = parameter(E_tau, self.varshape, allow_none=False)
    self.I_tau = parameter(I_tau, self.varshape, allow_none=False)
    self.E_theta = parameter(E_theta, self.varshape, allow_none=False)
    self.I_theta = parameter(I_theta, self.varshape, allow_none=False)
    self.wEE = parameter(wEE, self.varshape, allow_none=False)
    self.wIE = parameter(wIE, self.varshape, allow_none=False)
    self.wEI = parameter(wEI, self.varshape, allow_none=False)
    self.wII = parameter(wII, self.varshape, allow_none=False)
    self.r = parameter(r, self.varshape, allow_none=False)
    self.input_var = input_var

    # noise parameters
    self.x_ou_mean = parameter(x_ou_mean, self.varshape, allow_none=False)
    self.y_ou_mean = parameter(y_ou_mean, self.varshape, allow_none=False)
    self.x_ou_sigma = parameter(x_ou_sigma, self.varshape, allow_none=False)
    self.y_ou_sigma = parameter(y_ou_sigma, self.varshape, allow_none=False)
    self.x_ou_tau = parameter(x_ou_tau, self.varshape, allow_none=False)
    self.y_ou_tau = parameter(y_ou_tau, self.varshape, allow_none=False)

    # initializers
    is_initializer(x_initializer, 'x_initializer')
    is_initializer(y_initializer, 'y_initializer')
    self._x_initializer = x_initializer
    self._y_initializer = y_initializer

    # variables
    self.x = variable(x_initializer, self.mode, self.varshape)
    self.y = variable(y_initializer, self.mode, self.varshape)
    if self.input_var:
      self.input = variable(bm.zeros, self.mode, self.varshape)
      self.input_y = variable(bm.zeros, self.mode, self.varshape)

    # noise variables
    self.x_ou = self.y_ou = None
    if bm.any(self.x_ou_mean > 0.) or bm.any(self.x_ou_sigma > 0.):
      self.x_ou = OUProcess(self.varshape,
                            self.x_ou_mean,
                            self.x_ou_sigma,
                            self.x_ou_tau,
                            method=method)
    if bm.any(self.y_ou_mean > 0.) or bm.any(self.y_ou_sigma > 0.):
      self.y_ou = OUProcess(self.varshape,
                            self.y_ou_mean,
                            self.y_ou_sigma,
                            self.y_ou_tau,
                            method=method)

    # functions
    self.integral = odeint(f=JointEq([self.dx, self.dy]), method=method)

  def reset_state(self, batch_size=None):
    self.x.value = variable(self._x_initializer, batch_size, self.varshape)
    self.y.value = variable(self._y_initializer, batch_size, self.varshape)
    if self.input_var:
      self.input.value = variable(bm.zeros, batch_size, self.varshape)
      self.input_y.value = variable(bm.zeros, batch_size, self.varshape)
    if self.x_ou is not None:
      self.x_ou.reset_state(batch_size)
    if self.y_ou is not None:
      self.y_ou.reset_state(batch_size)

  def F(self, x, a, theta):
    return 1 / (1 + bm.exp(-a * (x - theta))) - 1 / (1 + bm.exp(a * theta))

  def dx(self, x, t, y, x_ext):
    xx = self.wEE * x - self.wIE * y + x_ext
    return (-x + (1 - self.r * x) * self.F(xx, self.E_a, self.E_theta)) / self.E_tau

  def dy(self, y, t, x, y_ext):
    xx = self.wEI * x - self.wII * y + y_ext
    return (-y + (1 - self.r * y) * self.F(xx, self.I_a, self.I_theta)) / self.I_tau

  def update(self, inp_x=None, inp_y=None):
    t = share.load('t')
    dt = share.load('dt')

    # input
    if self.input_var:
      if inp_x is not None:
        self.input += inp_x
      if self.x_ou is not None:
        self.input += self.x_ou()
      if inp_y is not None:
        self.input_y += inp_y
      if self.y_ou is not None:
        self.input_y += self.y_ou()
      input_x = self.input.value
      input_y = self.input_y.value
    else:
      input_x = inp_x if (inp_x is not None) else 0.
      if self.x_ou is not None: input_x += self.x_ou()
      input_y = inp_y if (inp_y is not None) else 0.
      if self.y_ou is not None: input_y += self.y_ou()

    x, y = self.integral(self.x, self.y, t, x_ext=input_x, y_ext=input_y, dt=dt)
    self.x.value = x
    self.y.value = y
    return x

  def clear_input(self):
    if self.input_var:
      self.input.value = bm.zeros_like(self.input)
      self.input_y.value = bm.zeros_like(self.input_y)


class JansenRitModel(RateModel):
  pass


class KuramotoOscillator(RateModel):
  pass


class ThetaNeuron(RateModel):
  pass


class RateQIFWithSFA(RateModel):
  pass


class VanDerPolOscillator(RateModel):
  pass


class ThresholdLinearModel(RateModel):
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
      tau_e: Union[float, Callable, Initializer, ArrayType] = 2e-2,
      tau_i: Union[float, Callable, Initializer, ArrayType] = 1e-2,
      beta_e: Union[float, Callable, Initializer, ArrayType] = .066,
      beta_i: Union[float, Callable, Initializer, ArrayType] = .351,
      noise_e: Union[float, Callable, Initializer, ArrayType] = 0.,
      noise_i: Union[float, Callable, Initializer, ArrayType] = 0.,
      e_initializer: Union[ArrayType, Callable, Initializer] = ZeroInit(),
      i_initializer: Union[ArrayType, Callable, Initializer] = ZeroInit(),
      seed: int = None,
      keep_size: bool = False,
      name: str = None,

      # parameter for training
      mode: bm.Mode = None,
      input_var: bool = True,
  ):
    super(ThresholdLinearModel, self).__init__(size,
                                               name=name,
                                               keep_size=keep_size,
                                               mode=mode)

    # parameters
    self.seed = seed
    self.tau_e = parameter(tau_e, self.varshape, False)
    self.tau_i = parameter(tau_i, self.varshape, False)
    self.beta_e = parameter(beta_e, self.varshape, False)
    self.beta_i = parameter(beta_i, self.varshape, False)
    self.noise_e = parameter(noise_e, self.varshape, False)
    self.noise_i = parameter(noise_i, self.varshape, False)
    self._e_initializer = e_initializer
    self._i_initializer = i_initializer
    self.input_var = input_var

    # variables
    self.e = variable(e_initializer, self.mode, self.varshape)  # Firing rate of excitatory population
    self.i = variable(i_initializer, self.mode, self.varshape)  # Firing rate of inhibitory population
    if self.input_var:
       self.Ie = variable(bm.zeros, self.mode, self.varshape)  # Input of excitaory population
       self.Ii = variable(bm.zeros, self.mode, self.varshape)  # Input of inhibitory population

  def reset(self, batch_size=None):
    self.reset_state(batch_size)

  def reset_state(self, batch_size=None):
    self.e.value = variable(self._e_initializer, batch_size, self.varshape)
    self.i.value = variable(self._i_initializer, batch_size, self.varshape)
    if self.input_var:
      self.Ie.value = variable(bm.zeros, batch_size, self.varshape)
      self.Ii.value = variable(bm.zeros, batch_size, self.varshape)

  def update(self, inp_e=None, inp_i=None):
    dt = share.load('dt')

    # input
    if self.input_var:
      if inp_e is not None:
        self.Ie += inp_e
      if inp_i is not None:
        self.Ii += inp_i
      input_e = self.Ie.value
      input_i = self.Ii.value
    else:
      input_e = inp_e if (inp_e is not None) else 0.
      input_i = inp_i if (inp_i is not None) else 0.

    de = -self.e + self.beta_e * bm.maximum(input_e, 0.)
    with jax.ensure_compile_time_eval():
      has_noise = bm.any(self.noise_e != 0.)

    if has_noise:
      de += bm.random.randn(self.varshape) * self.noise_e
    de = de / self.tau_e
    self.e.value = bm.maximum(self.e + de * dt, 0.)

    di = -self.i + self.beta_i * bm.maximum(input_i, 0.)
    with jax.ensure_compile_time_eval():
      has_noise = bm.any(self.noise_i != 0.)

    if has_noise:
      di += bm.random.randn(self.varshape) * self.noise_i
    di = di / self.tau_i
    self.i.value = bm.maximum(self.i + di * dt, 0.)
    return self.e.value

  def clear_input(self):
    if self.input_var:
      self.Ie.value = bm.zeros_like(self.Ie)
      self.Ii.value = bm.zeros_like(self.Ii)
