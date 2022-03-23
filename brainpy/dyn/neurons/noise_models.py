# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.dyn.base import NeuGroup
from brainpy.integrators.sde import sdeint
from brainpy.types import Parameter, Shape

__all__ = [
  'OUProcess',
]


class OUProcess(NeuGroup):
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
      mean: Parameter,
      sigma: Parameter,
      tau: Parameter,
      method: str = 'euler',
      name: str = None
  ):
    super(OUProcess, self).__init__(size=size, name=name)

    # parameters
    self.mean = mean
    self.sigma = sigma
    self.tau = tau

    # variables
    self.x = bm.Variable(bm.ones(self.num) * mean)

    # integral functions
    self.integral = sdeint(f=self.df, g=self.dg, method=method)

  def df(self, x, t):
    f_x_ou = (self.mean - x) / self.tau
    return f_x_ou

  def dg(self, x, t):
    return self.sigma

  def update(self, _t, _dt):
    self.x.value = self.integral(self.x, _t, _dt)
