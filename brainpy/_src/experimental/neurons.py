from typing import Union, Callable, Optional

from jax.lax import stop_gradient

import brainpy.math as bm
from brainpy._src.dyn.base import NeuGroup, not_pass_shargs
from brainpy._src.initialize import (ZeroInit, OneInit, Initializer, parameter, variable_)
from brainpy._src.integrators import odeint
from brainpy.check import is_initializer, is_callable, is_subclass
from brainpy.types import Shape, ArrayType


class LIF(NeuGroup):
  r"""Leaky integrate-and-fire neuron model.

  **Model Descriptions**

  The formal equations of a LIF model [1]_ is given by:

  .. math::

      \tau \frac{dV}{dt} = - (V(t) - V_{rest}) + RI(t) \\
      \text{after} \quad V(t) \gt V_{th}, V(t) = V_{reset} \quad
      \text{last} \quad \tau_{ref} \quad  \text{ms}

  where :math:`V` is the membrane potential, :math:`V_{rest}` is the resting
  membrane potential, :math:`V_{reset}` is the reset membrane potential,
  :math:`V_{th}` is the spike threshold, :math:`\tau` is the time constant,
  :math:`\tau_{ref}` is the refractory time period,
  and :math:`I` is the time-variant synaptic inputs.

  **Model Examples**

  - `(Brette, Romain. 2004) LIF phase locking <https://brainpy-examples.readthedocs.io/en/latest/neurons/Romain_2004_LIF_phase_locking.html>`_


  Parameters
  ----------
  size: sequence of int, int
    The size of the neuron group.
  V_rest: float, ArrayType, Initializer, callable
    Resting membrane potential.
  V_reset: float, ArrayType, Initializer, callable
    Reset potential after spike.
  V_th: float, ArrayType, Initializer, callable
    Threshold potential of spike.
  R: float, ArrayType, Initializer, callable
    Membrane resistance.
  tau: float, ArrayType, Initializer, callable
    Membrane time constant.
  tau_ref: float, ArrayType, Initializer, callable
    Refractory period length.(ms)
  V_initializer: ArrayType, Initializer, callable
    The initializer of membrane potential.
  noise: ArrayType, Initializer, callable
    The noise added onto the membrane potential
  method: str
    The numerical integration method.
  name: str
    The group name.

  References
  ----------

  .. [1] Abbott, Larry F. "Lapicque’s introduction of the integrate-and-fire model
         neuron (1907)." Brain research bulletin 50, no. 5-6 (1999): 303-304.
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,

      # neuron parameter
      V_rest: Union[float, ArrayType, Initializer, Callable] = 0.,
      V_reset: Union[float, ArrayType, Initializer, Callable] = -5.,
      V_th: Union[float, ArrayType, Initializer, Callable] = 20.,
      R: Union[float, ArrayType, Initializer, Callable] = 1.,
      tau: Union[float, ArrayType, Initializer, Callable] = 10.,
      tau_ref: Optional[Union[float, ArrayType, Initializer, Callable]] = None,
      V_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),

      # training parameter
      mode: Optional[bm.Mode] = None,
      spike_fun: Callable = bm.surrogate.inv_square_grad,

      # other parameters
      method: str = 'exp_auto',
      name: Optional[str] = None,
  ):
    # initialization
    super(LIF, self).__init__(size=size,
                              name=name,
                              keep_size=keep_size,
                              mode=mode)
    is_subclass(self.mode, (bm.TrainingMode, bm.NonBatchingMode), self.name)

    # parameters
    self.V_rest = parameter(V_rest, self.varshape, allow_none=False)
    self.V_reset = parameter(V_reset, self.varshape, allow_none=False)
    self.V_th = parameter(V_th, self.varshape, allow_none=False)
    self.tau = parameter(tau, self.varshape, allow_none=False)
    self.R = parameter(R, self.varshape, allow_none=False)
    self.tau_ref = parameter(tau_ref, self.varshape, allow_none=True)
    self.spike_fun = is_callable(spike_fun, 'spike_fun')

    # initializers
    is_initializer(V_initializer, 'V_initializer')
    self._V_initializer = V_initializer

    # integral
    self.integral = odeint(method=method, f=self.derivative)

    # variables
    self.reset_state(self.mode)

  def derivative(self, V, t, I_ext):
    return (-V + self.V_rest + self.R * I_ext) / self.tau

  def reset_state(self, batch_size=None):
    self.V = variable_(self._V_initializer, self.varshape, batch_size)
    self.spike = variable_(bm.zeros, self.varshape, batch_size)
    if self.tau_ref is not None:
      self.t_last_spike = variable_(OneInit(-1e7), self.varshape, batch_size)

  @not_pass_shargs
  def update(self, current):
    t = bm.share.get('t')

    # integrate membrane potential
    V = self.integral(self.V.value, t, current, bm.dt)

    if self.tau_ref is not None:
      refractory = stop_gradient((t - self.t_last_spike) <= self.tau_ref)
      V = bm.where(refractory, self.V.value, V)

      # spike, refractory, spiking time, and membrane potential reset
      spike = self.spike_fun(V - self.V_th)
      spike_no_grad = stop_gradient(spike)
      V += (self.V_reset - V) * spike_no_grad
      t_last_spike = bm.where(spike_no_grad, t, self.t_last_spike)

      # updates
      self.V.value = V
      self.spike.value = spike
      self.t_last_spike.value = stop_gradient(t_last_spike)

    else:
      # spike, spiking time, and membrane potential reset
      spike = self.spike_fun(V - self.V_th)
      V += (self.V_reset - V) * stop_gradient(spike)

      # updates
      self.V.value = V
      self.spike.value = spike

    return spike
