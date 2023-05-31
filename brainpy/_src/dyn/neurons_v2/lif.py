
from functools import partial
from typing import Union, Callable, Optional
from jax.sharding import Sharding


import brainpy.math as bm
from brainpy._src.dynsys import NeuGroupNS
from brainpy._src.context import share
from brainpy._src.initialize import (ZeroInit,
                                     OneInit,
                                     Initializer,
                                     parameter,
                                     variable_,
                                     noise as init_noise)
from brainpy._src.integrators import sdeint, odeint, JointEq
from brainpy.check import is_initializer, is_callable, is_subclass
from brainpy.types import Shape, ArrayType

__all__ = [
  'Leaky',
]

class Leaky(NeuGroupNS):
  r"""Leaky Integrator Model.

  **Model Descriptions**

  This class implements a leaky model, in which its dynamics is
  given by:

  .. math::

     x(t + \Delta t) = \exp{-1/\tau \Delta t} x(t) + I

  Parameters
  ----------
  size: sequence of int, int
    The size of the neuron group.
  tau: float, ArrayType, Initializer, callable
    Membrane time constant.
  method: str
    The numerical integration method.
  name: str
    The group name.
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      tau: Union[float, ArrayType, Initializer, Callable] = 10.,
      name: str = None,
      mode: bm.Mode = None,
      method: str = 'exp_auto',
  ):
    super().__init__(size=size,
                     mode=mode,
                     keep_size=keep_size,
                     name=name)
    assert self.mode.is_parent_of(bm.TrainingMode, bm.NonBatchingMode)

    # parameters
    self.tau = parameter(tau, self.varshape, allow_none=False)

    # integral
    self.integral = odeint(method=method, f=self.derivative)

    # variables
    self.reset_state(self.mode)

  def derivative(self, x, t):
    return -x / self.tau

  def reset_state(self, batch_size=None):
    self.x = variable_(bm.zeros, self.varshape, batch_size)

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    r = self.integral(self.x.value, t, dt)
    if x is not None:
      r += x
    self.x.value = r
    return r
