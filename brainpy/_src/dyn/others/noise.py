from typing import Union, Callable

import jax.numpy as jnp

import brainpy.math as bm
from brainpy._src.context import share
from brainpy._src.dyn.base import NeuDyn
from brainpy._src.initialize import variable_, parameter
from brainpy._src.integrators.sde.generic import sdeint
from brainpy.types import Shape, ArrayType

__all__ = [
  'OUProcess',
]


class OUProcess(NeuDyn):
  r"""The Ornstein–Uhlenbeck process.

  The Ornstein–Uhlenbeck process :math:`x_{t}` is defined by the following
  stochastic differential equation:

  .. math::

     \tau dx_{t}=-\theta \,x_{t}\,dt+\sigma \,dW_{t}

  where :math:`\theta >0` and :math:`\sigma >0` are parameters and :math:`W_{t}`
  denotes the Wiener process.

  Parameters
  ----------
  size: int, sequence of int
    The model size.
  mean: Parameter
    The noise mean value.
  sigma: Parameter
    The noise amplitude.
  tau: Parameter
    The decay time constant.
  method: str
    The numerical integration method for stochastic differential equation.
  name: str
    The model name.
  """

  def __init__(
      self,
      size: Shape,
      mean: Union[float, ArrayType, Callable] = 0.,
      sigma: Union[float, ArrayType, Callable] = 1.,
      tau: Union[float, ArrayType, Callable] = 10.,
      method: str = 'exp_euler',
      keep_size: bool = False,
      mode: bm.Mode = None,
      name: str = None,
  ):
    super(OUProcess, self).__init__(size=size, name=name, keep_size=keep_size, mode=mode)

    # parameters
    self.mean = parameter(mean, self.varshape, allow_none=False)
    self.sigma = parameter(sigma, self.varshape, allow_none=False)
    self.tau = parameter(tau, self.varshape, allow_none=False)

    # variables
    self.reset_state(self.mode)

    # integral functions
    self.integral = sdeint(f=self.df, g=self.dg, method=method)

  def reset_state(self, batch_size=None):
    self.x = variable_(lambda s: jnp.ones(s) * self.mean, self.varshape, batch_size)

  def df(self, x, t):
    return (self.mean - x) / self.tau

  def dg(self, x, t):
    return self.sigma

  def update(self):
    t = share.load('t')
    dt = share.load('dt')
    self.x.value = self.integral(self.x, t, dt)
    return self.x.value
