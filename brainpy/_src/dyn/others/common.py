from typing import Union, Callable, Optional, Sequence

import brainpy.math as bm
from brainpy._src import initialize as init
from brainpy._src import tools
from brainpy._src.context import share
from brainpy._src.dyn._docs import pneu_doc
from brainpy._src.dyn.base import NeuDyn
from brainpy._src.integrators import odeint
from brainpy.check import is_initializer
from brainpy.types import ArrayType

__all__ = [
  'Leaky',
  'Integrator',
]


class Leaky(NeuDyn):
  r"""Leaky Integrator Model.

  **Model Descriptions**

  This class implements a leaky model, in which its dynamics is
  given by:

  .. math::

     x(t + \Delta t) = \exp{-\Delta t/\tau} x(t) + I

  Args:
    tau: float, ArrayType, Initializer, callable. Membrane time constant.
    method: str. The numerical integration method. Default "exp_auto".
    init_var: Initialize the variable or not.
    %s
  """

  supported_modes = (bm.TrainingMode, bm.NonBatchingMode)

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      sharding: Optional[Sequence[str]] = None,
      keep_size: bool = False,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,

      tau: Union[float, ArrayType, Callable] = 10.,
      method: str = 'exp_auto',
      init_var: bool = True
  ):
    super().__init__(size,
                     mode=mode,
                     name=name,
                     sharding=sharding,
                     keep_size=keep_size)

    # parameters
    self.sharding = sharding
    self.tau = self.init_param(tau)

    # integral
    self.integral = odeint(method=method, f=self.derivative)

    # variables
    if init_var:
      self.reset_state(self.mode)

  def derivative(self, x, t):
    return -x / self.tau

  def reset_state(self, batch_size=None):
    self.x = self.init_variable(bm.zeros, batch_size)

  def update(self, inp=None):
    t = share.load('t')
    dt = share.load('dt')
    self.x.value = self.integral(self.x.value, t, dt)
    if inp is not None:
      self.x += inp
    return self.x.value

  def return_info(self):
    return self.x


Leaky.__doc__ = Leaky.__doc__ % pneu_doc


class Integrator(NeuDyn):
  r"""Integrator Model.

  This class implements an integrator model, in which its dynamics is
  given by:

  .. math::

     \tau \frac{dx}{dt} = - x(t) + I(t)

  where :math:`x` is the integrator value, and :math:`\tau` is the time constant.

  Args:
    tau: float, ArrayType, Initializer, callable. Membrane time constant.
    method: str. The numerical integration method. Default "exp_auto".
    x_initializer: ArrayType, Initializer, callable. The initializer of :math:`x`.
    %s
  """

  supported_modes = (bm.TrainingMode, bm.NonBatchingMode)

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      sharding: Optional[Sequence[str]] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,

      tau: Union[float, ArrayType, Callable] = 10.,
      x_initializer: Union[Callable, ArrayType] = init.ZeroInit(),
      method: str = 'exp_auto',
      init_var: bool = True,
  ):
    super().__init__(size,
                     mode=mode,
                     name=name,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.size = tools.to_size(size)
    self.sharding = sharding
    self.tau = init.parameter(tau, self.size, sharding=self.sharding)

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
    self.x = self.init_variable(self._x_initializer, batch_size)

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    x = 0. if x is None else x
    self.x.value = self.integral(self.x.value, t, I_ext=x, dt=dt)
    return self.x.value

  def return_info(self):
    return self.x


Integrator.__doc__ = Integrator.__doc__ % pneu_doc
