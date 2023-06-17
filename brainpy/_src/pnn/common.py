from typing import Union, Callable, Optional, Sequence

import brainpy.math as bm
from brainpy._src import initialize as init
from brainpy._src import tools
from brainpy._src.context import share
from brainpy._src.dynsys import DynamicalSystemNS
from brainpy._src.integrators import odeint
from brainpy._src.pnn.utils.axis_names import NEU_AXIS
from brainpy.check import is_initializer
from brainpy.types import Shape, ArrayType

__all__ = [
  'Leaky',
  'Integrator',
]


class Leaky(DynamicalSystemNS):
  r"""Leaky Integrator Model.

  **Model Descriptions**

  This class implements a leaky model, in which its dynamics is
  given by:

  .. math::

     x(t + \Delta t) = \exp{-\Delta t/\tau} x(t) + I

  Args:
    size: sequence of int, int. The size of the neuron group.
    tau: float, ArrayType, Initializer, callable. Membrane time constant.
    method: str. The numerical integration method. Default "exp_auto".
    mode: Mode. The computing mode. Default None.
    name: str. The group name.
  """

  supported_modes = (bm.TrainingMode, bm.NonBatchingMode)

  def __init__(
      self,
      size: Shape,
      axis_names: Optional[Sequence[str]] = (NEU_AXIS,),
      tau: Union[float, ArrayType, Callable] = 10.,
      method: str = 'exp_auto',
      mode: bm.Mode = None,
      name: str = None,
      init_var: bool = True
  ):
    super().__init__(mode=mode, name=name)

    # parameters
    self.size = tools.to_size(size)
    self.axis_names = axis_names
    self.tau = init.parameter(tau, self.size, axis_names=axis_names)

    # integral
    self.integral = odeint(method=method, f=self.derivative)

    # variables
    if init_var:
      self.reset_state(self.mode)

  def derivative(self, x, t):
    return -x / self.tau

  def reset_state(self, batch_size=None):
    self.x = init.variable_(bm.zeros, self.size, batch_size, axis_names=self.axis_names)

  def update(self, inp=None):
    t = share.load('t')
    dt = share.load('dt')
    self.x.value = self.integral(self.x.value, t, dt)
    if inp is not None:
      self.x += inp
    return self.x.value


class Integrator(DynamicalSystemNS):
  r"""Integrator Model.

  This class implements an integrator model, in which its dynamics is
  given by:

  .. math::

     \tau \frac{dx}{dt} = - x(t) + I(t)

  where :math:`x` is the integrator value, and :math:`\tau` is the time constant.

  Args:
    size: sequence of int, int. The size of the neuron group.
    tau: float, ArrayType, Initializer, callable. Membrane time constant.
    method: str. The numerical integration method. Default "exp_auto".
    name: str. The group name.
    mode: Mode. The computing mode. Default None.
    x_initializer: ArrayType, Initializer, callable. The initializer of :math:`x`.
  """

  supported_modes = (bm.TrainingMode, bm.NonBatchingMode)

  def __init__(
      self,
      size: Shape,
      axis_names: Optional[Sequence[str]] = (NEU_AXIS,),
      tau: Union[float, ArrayType, Callable] = 10.,
      x_initializer: Union[Callable, ArrayType] = init.ZeroInit(),
      name: str = None,
      mode: bm.Mode = None,
      method: str = 'exp_auto',
      init_var: bool = True,
  ):
    super().__init__(mode=mode, name=name)

    # parameters
    self.size = tools.to_size(size)
    self.axis_names = axis_names
    self.tau = init.parameter(tau, self.size, axis_names=self.axis_names)

    # initializers
    self._x_initializer = is_initializer(x_initializer)

    # integral
    self.integral = odeint(method=method, f=self.derivative)

    # variables
    if init_var:
      self.reset_state(self.mode)

  def derivative(self, V, t, I_ext):
    return (-V + I_ext) / self.tau

  def reset_state(self, batch_size=None):
    self.x = init.variable_(self._x_initializer, self.size, batch_size, axis_names=self.axis_names)

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    x = 0. if x is None else x
    self.x.value = self.integral(self.x.value, t, I_ext=x, dt=dt)
    return self.x.value

