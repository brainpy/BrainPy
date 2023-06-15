from functools import partial
from typing import Union, Callable, Optional, Sequence, Any

from jax.lax import stop_gradient

import brainpy.math as bm
from brainpy._src.context import share
from brainpy._src.initialize import (ZeroInit, Initializer)
from brainpy._src.integrators import odeint
from brainpy._src.pnn.utils.axis_names import NEU_AXIS
from brainpy.check import is_initializer
from brainpy.types import Shape, ArrayType
from ._docs import ref_doc, lif_doc, pneu_doc, dpneu_doc, ltc_doc
from .base import DPNeuGroup

__all__ = [
  'LIF',
  'LIFLtc',
  'LIFRef',
  'LIFRefLtc',
]


class IF(DPNeuGroup):
  pass


class LIFLtc(DPNeuGroup):
  r"""Leaky integrate-and-fire neuron model %s.

  The formal equations of a LIF model [1]_ is given by:

  .. math::

      \tau \frac{dV}{dt} = - (V(t) - V_{rest}) + RI(t) \\
      \text{after} \quad V(t) \gt V_{th}, V(t) = V_{reset}

  where :math:`V` is the membrane potential, :math:`V_{rest}` is the resting
  membrane potential, :math:`V_{reset}` is the reset membrane potential,
  :math:`V_{th}` is the spike threshold, :math:`\tau` is the time constant,
  and :math:`I` is the time-variant synaptic inputs.

  .. [1] Abbott, Larry F. "Lapicque’s introduction of the integrate-and-fire model
         neuron (1907)." Brain research bulletin 50, no. 5-6 (1999): 303-304.

  Args:
    %s
    %s
    %s

  """

  def __init__(
      self,
      size: Shape,
      axis_names: Optional[Sequence[str]] = (NEU_AXIS,),
      keep_size: bool = False,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
      spk_fun: Callable = bm.surrogate.InvSquareGrad(),
      spk_type: Any = None,
      detach_spk: bool = False,
      method: str = 'exp_auto',
      init_var: bool = True,

      # neuron parameters
      V_rest: Union[float, ArrayType, Initializer, Callable] = 0.,
      V_reset: Union[float, ArrayType, Initializer, Callable] = -5.,
      V_th: Union[float, ArrayType, Initializer, Callable] = 20.,
      R: Union[float, ArrayType, Initializer, Callable] = 1.,
      tau: Union[float, ArrayType, Initializer, Callable] = 10.,
      V_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),
  ):
    # initialization
    super().__init__(size=size,
                     name=name,
                     keep_size=keep_size,
                     mode=mode,
                     axis_names=axis_names,
                     spk_fun=spk_fun,
                     detach_spk=detach_spk,
                     method=method,
                     spk_type=spk_type)

    # parameters
    self.V_rest = self.sharding_param(V_rest)
    self.V_reset = self.sharding_param(V_reset)
    self.V_th = self.sharding_param(V_th)
    self.tau = self.sharding_param(tau)
    self.R = self.sharding_param(R)

    # initializers
    self._V_initializer = is_initializer(V_initializer)

    # integral
    self.integral = odeint(method=method, f=self.derivative)

    # variables
    if init_var:
      self.reset_state(self.mode)

  def derivative(self, V, t, I):
    for out in self.cur_outputs.values():
      I += out(V)
    return (-V + self.V_rest + self.R * I) / self.tau

  def reset_state(self, batch_size=None):
    self.V = self.sharding_variable(self._V_initializer, batch_size)
    self.spike = self.sharding_variable(partial(bm.zeros, dtype=self.spk_type), batch_size)

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    x = 0. if x is None else x

    # integrate membrane potential
    V = self.integral(self.V.value, t, x, dt)

    # spike, spiking time, and membrane potential reset
    if isinstance(self.mode, bm.TrainingMode):
      spike = self.spk_fun(V - self.V_th)
      spike = stop_gradient(spike) if self.detach_spk else spike
      V += (self.V_reset - V) * spike

    else:
      spike = V >= self.V_th
      V = bm.where(spike, self.V_reset, V)

    self.V.value = V
    self.spike.value = spike
    return spike


class LIF(LIFLtc):
  def derivative(self, V, t, I):
    return (-V + self.V_rest + self.R * I) / self.tau

  def update(self, x=None):
    x = 0. if x is None else x
    for out in self.cur_outputs.values():
      x += out(self.V.value)
    super().update(x)


LIF.__doc__ = LIFLtc.__doc__ % ('', lif_doc, pneu_doc, dpneu_doc)
LIFLtc.__doc__ = LIFLtc.__doc__ % (ltc_doc, lif_doc, pneu_doc, dpneu_doc)


class LIFRefLtc(LIFLtc):
  r"""Leaky integrate-and-fire neuron model %s which has refractory periods.

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

  .. [1] Abbott, Larry F. "Lapicque’s introduction of the integrate-and-fire model
         neuron (1907)." Brain research bulletin 50, no. 5-6 (1999): 303-304.

  Args:
    %s
    %s
    %s
    %s

  """

  def __init__(
      self,
      size: Shape,
      axis_names: Optional[Sequence[str]] = (NEU_AXIS,),
      keep_size: bool = False,
      mode: Optional[bm.Mode] = None,
      spk_fun: Callable = bm.surrogate.InvSquareGrad(),
      spk_type: Any = None,
      detach_spk: bool = False,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      init_var: bool = True,

      # old neuron parameter
      V_rest: Union[float, ArrayType, Initializer, Callable] = 0.,
      V_reset: Union[float, ArrayType, Initializer, Callable] = -5.,
      V_th: Union[float, ArrayType, Initializer, Callable] = 20.,
      R: Union[float, ArrayType, Initializer, Callable] = 1.,
      tau: Union[float, ArrayType, Initializer, Callable] = 10.,
      V_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),

      # new neuron parameter
      tau_ref: Optional[Union[float, ArrayType, Initializer, Callable]] = None,
      has_ref_var: bool = False,
  ):
    # initialization
    super().__init__(
      size=size,
      name=name,
      keep_size=keep_size,
      mode=mode,
      method=method,
      axis_names=axis_names,
      spk_fun=spk_fun,
      detach_spk=detach_spk,
      spk_type=spk_type,

      init_var=False,

      V_rest=V_rest,
      V_reset=V_reset,
      V_th=V_th,
      R=R,
      tau=tau,
      V_initializer=V_initializer,
    )

    # parameters
    self.has_ref_var = has_ref_var
    self.tau_ref = self.sharding_param(tau_ref)

    # initializers
    self._V_initializer = is_initializer(V_initializer)

    # integral
    self.integral = odeint(method=method, f=self.derivative)

    # variables
    if init_var:
      self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    super().reset_state(batch_size)
    self.t_last_spike = self.sharding_variable(bm.ones, batch_size)
    self.t_last_spike.fill_(-1e7)
    if self.has_ref_var:
      self.refractory = self.sharding_variable(partial(bm.zeros, dtype=bool), batch_size)

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    x = 0. if x is None else x

    # integrate membrane potential
    V = self.integral(self.V.value, t, x, dt)

    # refractory
    refractory = (t - self.t_last_spike) <= self.tau_ref
    if isinstance(self.mode, bm.TrainingMode):
      refractory = stop_gradient(refractory)
    V = bm.where(refractory, self.V_reset, V)

    # spike, refractory, spiking time, and membrane potential reset
    if isinstance(self.mode, bm.TrainingMode):
      spike = self.spk_fun(V - self.V_th)
      spike_no_grad = stop_gradient(spike) if self.detach_spk else spike
      V += (self.V_reset - V) * spike_no_grad
      spike_ = spike_no_grad > 0.
      # will be used in other place, like Delta Synapse, so stop its gradient
      if self.has_ref_var:
        self.refractory.value = stop_gradient(bm.logical_or(refractory, spike_).value)
      t_last_spike = stop_gradient(bm.where(spike_, t, self.t_last_spike.value))

    else:
      spike = V >= self.V_th
      V = bm.where(spike, self.V_reset, V)
      if self.has_ref_var:
        self.refractory.value = bm.logical_or(refractory, spike)
      t_last_spike = bm.where(spike, t, self.t_last_spike.value)
    self.V.value = V
    self.spike.value = spike
    self.t_last_spike.value = t_last_spike
    return spike


class LIFRef(LIFRefLtc):
  def derivative(self, V, t, I):
    return (-V + self.V_rest + self.R * I) / self.tau

  def update(self, x=None):
    x = 0. if x is None else x
    for out in self.cur_outputs.values():
      x += out(self.V.value)
    super().update(x)


LIFRef.__doc__ = LIFRefLtc.__doc__ % ('', lif_doc, pneu_doc, dpneu_doc, ref_doc)
LIFRefLtc.__doc__ = LIFRefLtc.__doc__ % (ltc_doc, lif_doc, pneu_doc, dpneu_doc, ref_doc)


class ExpIF(DPNeuGroup):
  pass


class AdExIF(DPNeuGroup):
  pass


class QuaIF(DPNeuGroup):
  pass


class AdQuaIF(DPNeuGroup):
  pass


class GIF(DPNeuGroup):
  pass

