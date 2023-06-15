# -*- coding: utf-8 -*-

from functools import partial
from typing import Union, Callable, Optional

from jax.lax import stop_gradient

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
  'LeakyIntegrator',
  'LIF',
  'ExpIF',
  'AdExIF',
  'QuaIF',
  'AdQuaIF',
  'GIF',
  'ALIFBellec2020',
  'Izhikevich',
  'HindmarshRose',
  'FHN',
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


class Integrator(NeuGroupNS):
  r"""Integrator Model.

  This class implements an integrator model, in which its dynamics is
  given by:

  .. math::

     \tau \frac{dx}{dt} = - x(t) + I(t)

  where :math:`x` is the integrator value, and :math:`\tau` is the time constant.

  Parameters
  ----------
  size: sequence of int, int
    The size of the neuron group.
  tau: float, ArrayType, Initializer, callable
    Membrane time constant.
  x_initializer: ArrayType, Initializer, callable
    The initializer of :math:`x`.
  noise: ArrayType, Initializer, callable
    The noise added onto the membrane potential
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
      x_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),
      noise: Union[float, ArrayType, Initializer, Callable] = None,
      input_var: bool = False,
      name: str = None,
      mode: bm.Mode = None,
      method: str = 'exp_auto',
  ):
    super().__init__(size=size,
                     mode=mode,
                     keep_size=keep_size,
                     name=name)
    is_subclass(self.mode, (bm.TrainingMode, bm.NonBatchingMode))

    # parameters
    self.tau = parameter(tau, self.varshape, allow_none=False)
    self.noise = init_noise(noise, self.varshape)
    self.input_var = input_var

    # initializers
    self._x_initializer = is_initializer(x_initializer)

    # integral
    if self.noise is None:
      self.integral = odeint(method=method, f=self.derivative)
    else:
      self.integral = sdeint(method=method, f=self.derivative, g=self.noise)

    # variables
    self.reset_state(self.mode)

  def derivative(self, V, t, I_ext):
    return (-V + I_ext) / self.tau

  def reset_state(self, batch_size=None):
    self.x = variable_(self._x_initializer, self.varshape, batch_size)
    if self.input_var:
      self.input = variable_(bm.zeros, self.varshape, batch_size)

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    if self.input_var:
      if x is not None:
        self.input += x
      x = self.input.value
    else:
      x = 0. if x is None else x
    self.x.value = self.integral(self.x.value, t, I_ext=x, dt=dt)
    return self.x.value

  def clear_input(self):
    if self.input_var:
      self.input[:] = 0.


class LeakyIntegrator(NeuGroupNS):
  r"""Leaky Integrator Model.

  **Model Descriptions**

  This class implements a leaky integrator model, in which its dynamics is
  given by:

  .. math::

     \tau \frac{dV}{dt} = - (V(t) - V_{rest}) + RI(t)

  where :math:`V` is the membrane potential, :math:`V_{rest}` is the resting
  membrane potential, :math:`\tau` is the time constant, and :math:`R` is the
  resistance.

  Parameters
  ----------
  size: sequence of int, int
    The size of the neuron group.
  V_rest: float, ArrayType, Initializer, callable
    Resting membrane potential.
  R: float, ArrayType, Initializer, callable
    Membrane resistance.
  tau: float, ArrayType, Initializer, callable
    Membrane time constant.
  V_initializer: ArrayType, Initializer, callable
    The initializer of membrane potential.
  noise: ArrayType, Initializer, callable
    The noise added onto the membrane potential
  method: str
    The numerical integration method.
  name: str
    The group name.
  """

  def __init__(
      self,

      # neuron group size
      size: Shape,
      keep_size: bool = False,

      # neuron parameters
      V_rest: Union[float, ArrayType, Initializer, Callable] = 0.,
      R: Union[float, ArrayType, Initializer, Callable] = 1.,
      tau: Union[float, ArrayType, Initializer, Callable] = 10.,
      V_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),
      noise: Union[float, ArrayType, Initializer, Callable] = None,

      # other parameter
      input_var: bool = True,
      name: str = None,
      mode: bm.Mode = None,
      method: str = 'exp_auto',
  ):
    super().__init__(size=size,
                     mode=mode,
                     keep_size=keep_size,
                     name=name)
    is_subclass(self.mode, (bm.TrainingMode, bm.NonBatchingMode))

    # parameters
    self.V_rest = parameter(V_rest, self.varshape, allow_none=False)
    self.tau = parameter(tau, self.varshape, allow_none=False)
    self.R = parameter(R, self.varshape, allow_none=False)
    self.noise = init_noise(noise, self.varshape)
    self.input_var = input_var

    # initializers
    self._V_initializer = is_initializer(V_initializer)

    # integral
    if self.noise is None:
      self.integral = odeint(method=method, f=self.derivative)
    else:
      self.integral = sdeint(method=method, f=self.derivative, g=self.noise)

    # variables
    self.reset_state(self.mode)

  def derivative(self, V, t, I_ext):
    return (-V + self.V_rest + self.R * I_ext) / self.tau

  def reset_state(self, batch_size=None):
    self.V = variable_(self._V_initializer, self.varshape, batch_size)
    if self.input_var:
      self.input = variable_(bm.zeros, self.varshape, batch_size)

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    if self.input_var:
      if x is not None:
        self.input += x
      x = self.input.value
    else:
      x = 0. if x is None else x
    self.V.value = self.integral(self.V.value, t, x, dt)
    return self.V.value

  def clear_input(self):
    if self.input_var:
      self.input[:] = 0.


class LIF(NeuGroupNS):
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
      noise: Optional[Union[float, ArrayType, Initializer, Callable]] = None,

      # training parameter
      mode: Optional[bm.Mode] = None,
      spike_fun: Callable = bm.surrogate.inv_square_grad,

      # other parameters
      input_var: bool = True,
      ref_var: bool = False,
      method: str = 'exp_auto',
      name: Optional[str] = None,
  ):
    # initialization
    super().__init__(size=size,
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
    self.noise = init_noise(noise, self.varshape)
    self.spike_fun = is_callable(spike_fun, 'spike_fun')
    self.input_var = input_var
    self.ref_var = ref_var

    # initializers
    self._V_initializer = is_initializer(V_initializer)

    # variables
    self.reset_state(self.mode)

    # integral
    if self.noise is None:
      self.integral = odeint(method=method, f=self.derivative)
    else:
      self.integral = sdeint(method=method, f=self.derivative, g=self.noise)

  def derivative(self, V, t, I):
    return (-V + self.V_rest + self.R * I) / self.tau

  def reset_state(self, batch_size=None):
    self.V = variable_(self._V_initializer, self.varshape, batch_size)
    if self.input_var:
      self.input = variable_(bm.zeros, self.varshape, batch_size)
    sp_type = bm.float_ if isinstance(self.mode, bm.TrainingMode) else bool  # the gradient of spike is a float
    self.spike = variable_(lambda s: bm.zeros(s, dtype=sp_type), self.varshape, batch_size)
    if self.tau_ref is not None:
      self.t_last_spike = variable_(lambda s: bm.ones(s) * -1e7, self.varshape, batch_size)
      if self.ref_var:
        self.refractory = variable_(lambda s: bm.zeros(s, dtype=bool), self.varshape, batch_size)

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    if self.input_var:
      if x is not None:
        self.input += x
      x = self.input.value
    else:
      x = 0. if x is None else x

    # integrate membrane potential
    V = self.integral(self.V.value, t, x, dt)

    if self.tau_ref is not None:
      # refractory
      refractory = (t - self.t_last_spike) <= self.tau_ref
      if isinstance(self.mode, bm.TrainingMode):
        refractory = stop_gradient(refractory)
      V = bm.where(refractory, self.V_reset, V)

      # spike, refractory, spiking time, and membrane potential reset
      if isinstance(self.mode, bm.TrainingMode):
        spike = self.spike_fun(V - self.V_th)
        spike_no_grad = stop_gradient(spike)
        V += (self.V_reset - V) * spike_no_grad
        spike_ = spike_no_grad > 0.
        # will be used in other place, like Delta Synapse, so stop its gradient
        if self.ref_var:
          self.refractory.value = stop_gradient(bm.logical_or(refractory, spike_).value)
        t_last_spike = stop_gradient(bm.where(spike_, t, self.t_last_spike.value))
      else:
        spike = V >= self.V_th
        V = bm.where(spike, self.V_reset, V)
        if self.ref_var:
          self.refractory.value = bm.logical_or(refractory, spike)
        t_last_spike = bm.where(spike, t, self.t_last_spike.value)
      self.V.value = V
      self.spike.value = spike
      self.t_last_spike.value = t_last_spike

    else:
      # spike, spiking time, and membrane potential reset
      if isinstance(self.mode, bm.TrainingMode):
        spike = self.spike_fun(V - self.V_th)
        spike_no_grad = stop_gradient(spike)
        V += (self.V_reset - V) * spike_no_grad
      else:
        spike = V >= self.V_th
        V = bm.where(spike, self.V_reset, V)
      self.V.value = V
      self.spike.value = spike
    return spike

  def clear_input(self):
    if self.input_var:
      self.input[:] = 0.


class ExpIF(NeuGroupNS):
  r"""Exponential integrate-and-fire neuron model.

  **Model Descriptions**

  In the exponential integrate-and-fire model [1]_, the differential
  equation for the membrane potential is given by

  .. math::

      \tau\frac{d V}{d t}= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} + RI(t), \\
      \text{after} \, V(t) \gt V_{th}, V(t) = V_{reset} \, \text{last} \, \tau_{ref} \, \text{ms}

  This equation has an exponential nonlinearity with "sharpness" parameter :math:`\Delta_{T}`
  and "threshold" :math:`\vartheta_{rh}`.

  The moment when the membrane potential reaches the numerical threshold :math:`V_{th}`
  defines the firing time :math:`t^{(f)}`. After firing, the membrane potential is reset to
  :math:`V_{rest}` and integration restarts at time :math:`t^{(f)}+\tau_{\rm ref}`,
  where :math:`\tau_{\rm ref}` is an absolute refractory time.
  If the numerical threshold is chosen sufficiently high, :math:`V_{th}\gg v+\Delta_T`,
  its exact value does not play any role. The reason is that the upswing of the action
  potential for :math:`v\gg v +\Delta_{T}` is so rapid, that it goes to infinity in
  an incredibly short time. The threshold :math:`V_{th}` is introduced mainly for numerical
  convenience. For a formal mathematical analysis of the model, the threshold can be pushed
  to infinity.

  The model was first introduced by Nicolas Fourcaud-Trocmé, David Hansel, Carl van Vreeswijk
  and Nicolas Brunel [1]_. The exponential nonlinearity was later confirmed by Badel et al. [3]_.
  It is one of the prominent examples of a precise theoretical prediction in computational
  neuroscience that was later confirmed by experimental neuroscience.

  Two important remarks:

  - (i) The right-hand side of the above equation contains a nonlinearity
    that can be directly extracted from experimental data [3]_. In this sense the exponential
    nonlinearity is not an arbitrary choice but directly supported by experimental evidence.
  - (ii) Even though it is a nonlinear model, it is simple enough to calculate the firing
    rate for constant input, and the linear response to fluctuations, even in the presence
    of input noise [4]_.

  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> group = bp.neurons.ExpIF(1)
    >>> runner = bp.DSRunner(group, monitors=['V'], inputs=('input', 10.))
    >>> runner.run(300., )
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='V', show=True)


  **Model Parameters**

  ============= ============== ======== ===================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ---------------------------------------------------
  V_rest        -65            mV       Resting potential.
  V_reset       -68            mV       Reset potential after spike.
  V_th          -30            mV       Threshold potential of spike.
  V_T           -59.9          mV       Threshold potential of generating action potential.
  delta_T       3.48           \        Spike slope factor.
  R             1              \        Membrane resistance.
  tau           10             \        Membrane time constant. Compute by R * C.
  tau_ref       1.7            \        Refractory period length.
  ============= ============== ======== ===================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                  0                 Membrane potential.
  input              0                 External and synaptic input current.
  spike              False             Flag to mark whether the neuron is spiking.
  refractory         False             Flag to mark whether the neuron is in refractory period.
  t_last_spike       -1e7              Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
         mechanisms determine the neuronal response to fluctuating
         inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
  .. [2] Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014).
         Neuronal dynamics: From single neurons to networks and models
         of cognition. Cambridge University Press.
  .. [3] Badel, Laurent, Sandrine Lefort, Romain Brette, Carl CH Petersen,
         Wulfram Gerstner, and Magnus JE Richardson. "Dynamic IV curves
         are reliable predictors of naturalistic pyramidal-neuron voltage
         traces." Journal of Neurophysiology 99, no. 2 (2008): 656-666.
  .. [4] Richardson, Magnus JE. "Firing-rate response of linear and nonlinear
         integrate-and-fire neurons to modulated current-based and
         conductance-based synaptic drive." Physical Review E 76, no. 2 (2007): 021919.
  .. [5] https://en.wikipedia.org/wiki/Exponential_integrate-and-fire
  """

  def __init__(
      self,
      size: Shape,
      V_rest: Union[float, ArrayType, Initializer, Callable] = -65.,
      V_reset: Union[float, ArrayType, Initializer, Callable] = -68.,
      V_th: Union[float, ArrayType, Initializer, Callable] = -30.,
      V_T: Union[float, ArrayType, Initializer, Callable] = -59.9,
      delta_T: Union[float, ArrayType, Initializer, Callable] = 3.48,
      R: Union[float, ArrayType, Initializer, Callable] = 1.,
      tau: Union[float, ArrayType, Initializer, Callable] = 10.,
      tau_ref: Union[float, ArrayType, Initializer, Callable] = None,
      V_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),
      noise: Union[float, ArrayType, Initializer, Callable] = None,
      keep_size: bool = False,
      input_var: bool = True,
      ref_var: bool = False,
      mode: bm.Mode = None,
      method: str = 'exp_auto',
      name: str = None
  ):
    # initialize
    super(ExpIF, self).__init__(size=size,
                                name=name,
                                mode=mode,
                                keep_size=keep_size, )
    is_subclass(self.mode, (bm.TrainingMode, bm.NonBatchingMode))

    # parameters
    self.V_rest = parameter(V_rest, self.varshape, allow_none=False)
    self.V_reset = parameter(V_reset, self.varshape, allow_none=False)
    self.V_th = parameter(V_th, self.varshape, allow_none=False)
    self.V_T = parameter(V_T, self.varshape, allow_none=False)
    self.delta_T = parameter(delta_T, self.varshape, allow_none=False)
    self.tau_ref = parameter(tau_ref, self.varshape, allow_none=True)
    self.tau = parameter(tau, self.varshape, allow_none=False)
    self.R = parameter(R, self.varshape, allow_none=False)
    self.noise = init_noise(noise, self.varshape)
    self.input_var = input_var
    self.ref_var = ref_var

    # initializers
    self._V_initializer = is_initializer(V_initializer)

    # variables
    self.reset_state(self.mode)

    # integral
    if self.noise is None:
      self.integral = odeint(method=method, f=self.derivative)
    else:
      self.integral = sdeint(method=method, f=self.derivative, g=self.noise)

  def reset_state(self, batch_size=None):
    self.V = variable_(self._V_initializer, self.varshape, batch_size)
    if self.input_var:
      self.input = variable_(bm.zeros, self.varshape, batch_size)
    sp_type = bm.float_ if isinstance(self.mode, bm.TrainingMode) else bool
    self.spike = variable_(lambda s: bm.zeros(s, dtype=sp_type), self.varshape, batch_size)
    if self.tau_ref is not None:
      self.t_last_spike = variable_(lambda s: bm.ones(s) * -1e7, self.varshape, batch_size)
      if self.ref_var:
        self.refractory = variable_(lambda s: bm.zeros(s, dtype=bool), self.varshape, batch_size)

  def derivative(self, V, t, I_ext):
    exp_v = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
    dvdt = (- (V - self.V_rest) + exp_v + self.R * I_ext) / self.tau
    return dvdt

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    if self.input_var:
      if x is not None:
        self.input += x
      x = self.input.value
    else:
      x = 0. if x is None else x
    V = self.integral(self.V.value, t, x, dt)
    if self.tau_ref is not None:
      refractory = (t - self.t_last_spike) <= self.tau_ref
      V = bm.where(refractory, self.V_reset, V)
      spike = self.V_th <= V
      V = bm.where(spike, self.V_reset, V)
      self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)
      if self.ref_var:
        self.refractory.value = bm.logical_or(refractory, spike)
    else:
      spike = self.V_th <= V
      V = bm.where(spike, self.V_reset, V)
    self.V.value = V
    self.spike.value = spike
    return spike

  def clear_input(self):
    if self.input_var:
      self.input[:] = 0.


class AdExIF(NeuGroupNS):
  r"""Adaptive exponential integrate-and-fire neuron model.

  **Model Descriptions**

  The **adaptive exponential integrate-and-fire model**, also called AdEx, is a
  spiking neuron model with two variables [1]_ [2]_.

  .. math::

      \begin{aligned}
      \tau_m\frac{d V}{d t} &= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} - Rw + RI(t), \\
      \tau_w \frac{d w}{d t} &=a(V-V_{rest}) - w
      \end{aligned}

  once the membrane potential reaches the spike threshold,

  .. math::

      V \rightarrow V_{reset}, \\
      w \rightarrow w+b.

  The first equation describes the dynamics of the membrane potential and includes
  an activation term with an exponential voltage dependence. Voltage is coupled to
  a second equation which describes adaptation. Both variables are reset if an action
  potential has been triggered. The combination of adaptation and exponential voltage
  dependence gives rise to the name Adaptive Exponential Integrate-and-Fire model.

  The adaptive exponential integrate-and-fire model is capable of describing known
  neuronal firing patterns, e.g., adapting, bursting, delayed spike initiation,
  initial bursting, fast spiking, and regular spiking.

  **Model Examples**

  - `Examples for different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Gerstner_2005_AdExIF_model.html>`_

  **Model Parameters**

  ============= ============== ======== ========================================================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
  V_rest        -65            mV       Resting potential.
  V_reset       -68            mV       Reset potential after spike.
  V_th          -30            mV       Threshold potential of spike and reset.
  V_T           -59.9          mV       Threshold potential of generating action potential.
  delta_T       3.48           \        Spike slope factor.
  a             1              \        The sensitivity of the recovery variable :math:`u` to the sub-threshold fluctuations of the membrane potential :math:`v`
  b             1              \        The increment of :math:`w` produced by a spike.
  R             1              \        Membrane resistance.
  tau           10             ms       Membrane time constant. Compute by R * C.
  tau_w         30             ms       Time constant of the adaptation current.
  tau_ref       0.             ms       Refractory time. 
  ============= ============== ======== ========================================================================================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                   0                 Membrane potential.
  w                   0                 Adaptation current.
  input               0                 External and synaptic input current.
  spike               False             Flag to mark whether the neuron is spiking.
  refractory          False             Flag to mark whether the neuron is in refractory period.
  t_last_spike        -1e7              Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
         mechanisms determine the neuronal response to fluctuating
         inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
  .. [2] http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model
  """

  def __init__(
      self,
      size: Shape,
      V_rest: Union[float, ArrayType, Initializer, Callable] = -65.,
      V_reset: Union[float, ArrayType, Initializer, Callable] = -68.,
      V_th: Union[float, ArrayType, Initializer, Callable] = -30.,
      V_T: Union[float, ArrayType, Initializer, Callable] = -59.9,
      delta_T: Union[float, ArrayType, Initializer, Callable] = 3.48,
      a: Union[float, ArrayType, Initializer, Callable] = 1.,
      b: Union[float, ArrayType, Initializer, Callable] = 1.,
      tau: Union[float, ArrayType, Initializer, Callable] = 10.,
      tau_w: Union[float, ArrayType, Initializer, Callable] = 30.,
      tau_ref: Optional[Union[float, ArrayType, Initializer, Callable]] = None,
      R: Union[float, ArrayType, Initializer, Callable] = 1.,
      V_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),
      w_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),
      noise: Optional[Union[float, ArrayType, Initializer, Callable]] = None,
      method: str = 'exp_auto',
      keep_size: bool = False,
      input_var: bool = True,
      mode: bm.Mode = None,
      name: Optional[str] = None
  ):
    super(AdExIF, self).__init__(size=size,
                                 keep_size=keep_size,
                                 name=name,
                                 mode=mode, )
    is_subclass(self.mode, (bm.TrainingMode, bm.NonBatchingMode))

    # parameters
    self.V_rest = parameter(V_rest, self.varshape, allow_none=False)
    self.V_reset = parameter(V_reset, self.varshape, allow_none=False)
    self.V_th = parameter(V_th, self.varshape, allow_none=False)
    self.V_T = parameter(V_T, self.varshape, allow_none=False)
    self.a = parameter(a, self.varshape, allow_none=False)
    self.b = parameter(b, self.varshape, allow_none=False)
    self.R = parameter(R, self.varshape, allow_none=False)
    self.tau = parameter(tau, self.varshape, allow_none=False)
    self.tau_w = parameter(tau_w, self.varshape, allow_none=False)
    self.tau_ref = parameter(tau_ref, self.varshape, allow_none=True)
    self.delta_T = parameter(delta_T, self.varshape, allow_none=False)
    self.noise = init_noise(noise, self.varshape, num_vars=2)
    self.input_var = input_var

    # initializers
    self._V_initializer = is_initializer(V_initializer)
    self._w_initializer = is_initializer(w_initializer)

    # variables
    self.reset_state(self.mode)

    # functions
    if self.noise is None:
      self.integral = odeint(method=method, f=self.derivative)
    else:
      self.integral = sdeint(method=method, f=self.derivative, g=self.noise)

  def reset_state(self, batch_size=None):
    self.V = variable_(self._V_initializer, self.varshape, batch_size)
    self.w = variable_(self._w_initializer, self.varshape, batch_size)
    if self.input_var:
      self.input = variable_(bm.zeros, self.varshape, batch_size)
    sp_type = bm.float_ if isinstance(self.mode, bm.BatchingMode) else bool
    self.spike = variable_(lambda s: bm.zeros(s, dtype=sp_type), self.varshape, batch_size)
    if self.tau_ref is not None:
      self.refractory = variable_(partial(bm.zeros, dtype=bool), self.varshape, batch_size)
      self.t_last_spike = variable_(lambda s: bm.ones(s) * -1e8, self.varshape, batch_size)

  def dV(self, V, t, w, I_ext):
    exp = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
    dVdt = (- V + self.V_rest + exp - self.R * w + self.R * I_ext) / self.tau
    return dVdt

  def dw(self, w, t, V):
    dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
    return dwdt

  @property
  def derivative(self):
    return JointEq([self.dV, self.dw])

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    if self.input_var:
      if x is not None:
        self.input += x
      x = self.input.value
    else:
      x = 0. if x is None else x
    V, w = self.integral(self.V.value, self.w.value, t, x, dt)
    if self.tau_ref is not None:
      refractory = (t - self.t_last_spike) <= self.tau_ref
      V = bm.where(refractory, self.V_reset, V)
    spike = V >= self.V_th
    self.V.value = bm.where(spike, self.V_reset, V)
    self.w.value = bm.where(spike, w + self.b, w)
    self.spike.value = spike
    if self.tau_ref is not None:
      self.refractory.value = bm.logical_or(refractory, spike)
      self.t_last_spike.value = bm.where(spike, t, self.t_last_spike.value)
    return spike

  def clear_input(self):
    if self.input_var:
      self.input[:] = 0.


class QuaIF(NeuGroupNS):
  r"""Quadratic Integrate-and-Fire neuron model.

  **Model Descriptions**

  In contrast to physiologically accurate but computationally expensive
  neuron models like the Hodgkin–Huxley model, the QIF model [1]_ seeks only
  to produce **action potential-like patterns** and ignores subtleties
  like gating variables, which play an important role in generating action
  potentials in a real neuron. However, the QIF model is incredibly easy
  to implement and compute, and relatively straightforward to study and
  understand, thus has found ubiquitous use in computational neuroscience.

  .. math::

      \tau \frac{d V}{d t}=c(V-V_{rest})(V-V_c) + RI(t)

  where the parameters are taken to be :math:`c` =0.07, and :math:`V_c = -50 mV` (Latham et al., 2000).

  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>>
    >>> group = bp.neurons.QuaIF(1,)
    >>>
    >>> runner = bp.DSRunner(group, monitors=['V'], inputs=('input', 20.))
    >>> runner.run(duration=200.)
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.V, show=True)


  **Model Parameters**

  ============= ============== ======== ========================================================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
  V_rest        -65            mV       Resting potential.
  V_reset       -68            mV       Reset potential after spike.
  V_th          -30            mV       Threshold potential of spike and reset.
  V_c           -50            mV       Critical voltage for spike initiation. Must be larger than V_rest.
  c             .07            \        Coefficient describes membrane potential update. Larger than 0.
  R             1              \        Membrane resistance.
  tau           10             ms       Membrane time constant. Compute by R * C.
  tau_ref       0              ms       Refractory period length.
  ============= ============== ======== ========================================================================================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                   0                 Membrane potential.
  input               0                 External and synaptic input current.
  spike               False             Flag to mark whether the neuron is spiking.
  refractory          False             Flag to mark whether the neuron is in refractory period.
  t_last_spike       -1e7               Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1]  P. E. Latham, B.J. Richmond, P. Nelson and S. Nirenberg
          (2000) Intrinsic dynamics in neuronal networks. I. Theory.
          J. Neurophysiology 83, pp. 808–827.
  """

  def __init__(
      self,
      size: Shape,
      V_rest: Union[float, ArrayType, Initializer, Callable] = -65.,
      V_reset: Union[float, ArrayType, Initializer, Callable] = -68.,
      V_th: Union[float, ArrayType, Initializer, Callable] = -30.,
      V_c: Union[float, ArrayType, Initializer, Callable] = -50.0,
      c: Union[float, ArrayType, Initializer, Callable] = .07,
      R: Union[float, ArrayType, Initializer, Callable] = 1.,
      tau: Union[float, ArrayType, Initializer, Callable] = 10.,
      tau_ref: Union[float, ArrayType, Initializer, Callable] = None,
      V_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),
      noise: Union[float, ArrayType, Initializer, Callable] = None,
      keep_size: bool = False,
      input_var: bool = True,
      mode: bm.Mode = None,
      method: str = 'exp_auto',
      name: str = None
  ):
    # initialization
    super(QuaIF, self).__init__(size=size,
                                keep_size=keep_size,
                                name=name,
                                mode=mode)
    is_subclass(self.mode, (bm.TrainingMode, bm.NonBatchingMode))

    # parameters
    self.V_rest = parameter(V_rest, self.varshape, allow_none=False)
    self.V_reset = parameter(V_reset, self.varshape, allow_none=False)
    self.V_th = parameter(V_th, self.varshape, allow_none=False)
    self.V_c = parameter(V_c, self.varshape, allow_none=False)
    self.c = parameter(c, self.varshape, allow_none=False)
    self.R = parameter(R, self.varshape, allow_none=False)
    self.tau = parameter(tau, self.varshape, allow_none=False)
    self.tau_ref = parameter(tau_ref, self.varshape, allow_none=True)
    self.noise = init_noise(noise, self.varshape, num_vars=1)
    self.input_var = input_var

    # initializers
    self._V_initializer = is_initializer(V_initializer)

    # variables
    self.reset_state(self.mode)

    # integral
    if self.noise is None:
      self.integral = odeint(method=method, f=self.derivative)
    else:
      self.integral = sdeint(method=method, f=self.derivative, g=self.noise)

  def reset_state(self, batch_size=None):
    self.V = variable_(self._V_initializer, self.varshape, batch_size)
    if self.input_var:
      self.input = variable_(bm.zeros, self.varshape, batch_size)
    sp_type = bm.float_ if isinstance(self.mode, bm.TrainingMode) else bool
    self.spike = variable_(lambda s: bm.zeros(s, dtype=sp_type), self.varshape, batch_size)
    if self.tau_ref is not None:
      self.t_last_spike = variable_(lambda s: bm.ones(s) * -1e7, self.varshape, batch_size)
      self.refractory = variable_(lambda s: bm.zeros(s, dtype=bool), self.varshape, batch_size)

  def derivative(self, V, t, I_ext):
    dVdt = (self.c * (V - self.V_rest) * (V - self.V_c) + self.R * I_ext) / self.tau
    return dVdt

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    if self.input_var:
      if x is not None:
        self.input += x
      x = self.input.value
    else:
      x = 0. if x is None else x
    V = self.integral(self.V.value, t, x, dt)
    if self.tau_ref is not None:
      refractory = (t - self.t_last_spike) <= self.tau_ref
      V = bm.where(refractory, self.V_reset, V)
      spike = self.V_th <= V
      t_last_spike = bm.where(spike, t, self.t_last_spike.value)
      V = bm.where(spike, self.V_reset, V)
      self.refractory.value = bm.logical_or(refractory, spike)
      self.t_last_spike.value = t_last_spike
    else:
      spike = self.V_th <= V
      V = bm.where(spike, self.V_reset, V)
    self.V.value = V
    self.spike.value = spike

  def clear_input(self):
    if self.input_var:
      self.input[:] = 0.


class AdQuaIF(NeuGroupNS):
  r"""Adaptive quadratic integrate-and-fire neuron model.

  **Model Descriptions**

  The adaptive quadratic integrate-and-fire neuron model [1]_ is given by:

  .. math::

      \begin{aligned}
      \tau_m \frac{d V}{d t}&=c(V-V_{rest})(V-V_c) - w + I(t), \\
      \tau_w \frac{d w}{d t}&=a(V-V_{rest}) - w,
      \end{aligned}

  once the membrane potential reaches the spike threshold,

  .. math::

      V \rightarrow V_{reset}, \\
      w \rightarrow w+b.

  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> group = bp.neurons.AdQuaIF(1, )
    >>> runner = bp.DSRunner(group, monitors=['V', 'w'], inputs=('input', 30.))
    >>> runner.run(300)
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
    >>> fig.add_subplot(gs[0, 0])
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='V')
    >>> fig.add_subplot(gs[1, 0])
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.w, ylabel='w', show=True)

  **Model Parameters**

  ============= ============== ======== =======================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -------------------------------------------------------
  V_rest         -65            mV       Resting potential.
  V_reset        -68            mV       Reset potential after spike.
  V_th           -30            mV       Threshold potential of spike and reset.
  V_c            -50            mV       Critical voltage for spike initiation. Must be larger
                                         than :math:`V_{rest}`.
  a               1              \       The sensitivity of the recovery variable :math:`u` to
                                         the sub-threshold fluctuations of the membrane
                                         potential :math:`v`
  b              .1             \        The increment of :math:`w` produced by a spike.
  c              .07             \       Coefficient describes membrane potential update.
                                         Larger than 0.
  tau            10             ms       Membrane time constant.
  tau_w          10             ms       Time constant of the adaptation current.
  ============= ============== ======== =======================================================

  **Model Variables**

  ================== ================= ==========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ----------------------------------------------------------
  V                   0                 Membrane potential.
  w                   0                 Adaptation current.
  input               0                 External and synaptic input current.
  spike               False             Flag to mark whether the neuron is spiking.
  t_last_spike        -1e7              Last spike time stamp.
  ================== ================= ==========================================================

  **References**

  .. [1] Izhikevich, E. M. (2004). Which model to use for cortical spiking
         neurons?. IEEE transactions on neural networks, 15(5), 1063-1070.
  .. [2] Touboul, Jonathan. "Bifurcation analysis of a general class of
         nonlinear integrate-and-fire neurons." SIAM Journal on Applied
         Mathematics 68, no. 4 (2008): 1045-1079.
  """

  def __init__(
      self,
      size: Shape,
      V_rest: Union[float, ArrayType, Initializer, Callable] = -65.,
      V_reset: Union[float, ArrayType, Initializer, Callable] = -68.,
      V_th: Union[float, ArrayType, Initializer, Callable] = -30.,
      V_c: Union[float, ArrayType, Initializer, Callable] = -50.0,
      a: Union[float, ArrayType, Initializer, Callable] = 1.,
      b: Union[float, ArrayType, Initializer, Callable] = .1,
      c: Union[float, ArrayType, Initializer, Callable] = .07,
      tau: Union[float, ArrayType, Initializer, Callable] = 10.,
      tau_w: Union[float, ArrayType, Initializer, Callable] = 10.,
      V_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),
      w_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),
      noise: Union[float, ArrayType, Initializer, Callable] = None,
      method: str = 'exp_auto',
      keep_size: bool = False,
      input_var: bool = True,
      mode: bm.Mode = None,
      name: str = None
  ):
    super(AdQuaIF, self).__init__(size=size,
                                  keep_size=keep_size,
                                  name=name,
                                  mode=mode, )
    is_subclass(self.mode, (bm.TrainingMode, bm.NonBatchingMode))

    # parameters
    self.V_rest = parameter(V_rest, self.varshape, allow_none=False)
    self.V_reset = parameter(V_reset, self.varshape, allow_none=False)
    self.V_th = parameter(V_th, self.varshape, allow_none=False)
    self.V_c = parameter(V_c, self.varshape, allow_none=False)
    self.c = parameter(c, self.varshape, allow_none=False)
    self.a = parameter(a, self.varshape, allow_none=False)
    self.b = parameter(b, self.varshape, allow_none=False)
    self.tau = parameter(tau, self.varshape, allow_none=False)
    self.tau_w = parameter(tau_w, self.varshape, allow_none=False)
    self.noise = init_noise(noise, self.varshape, num_vars=2)
    self.input_var = input_var

    # initializers
    self._V_initializer = is_initializer(V_initializer)
    self._w_initializer = is_initializer(w_initializer)

    # variables
    self.reset_state(self.mode)

    # integral
    if self.noise is None:
      self.integral = odeint(method=method, f=self.derivative)
    else:
      self.integral = sdeint(method=method, f=self.derivative, g=self.noise)

  def reset_state(self, batch_size=None):
    self.V = variable_(self._V_initializer, self.varshape, batch_size)
    self.w = variable_(self._w_initializer, self.varshape, batch_size)
    sp_type = bm.float_ if isinstance(self.mode, bm.TrainingMode) else bool
    self.spike = variable_(lambda s: bm.zeros(s, dtype=sp_type), self.varshape, batch_size)
    if self.input_var:
      self.input = variable_(bm.zeros, self.varshape, batch_size)

  def dV(self, V, t, w, I_ext):
    dVdt = (self.c * (V - self.V_rest) * (V - self.V_c) - w + I_ext) / self.tau
    return dVdt

  def dw(self, w, t, V):
    dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
    return dwdt

  @property
  def derivative(self):
    return JointEq([self.dV, self.dw])

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    if self.input_var:
      if x is not None:
        self.input += x
      x = self.input.value
    else:
      x = 0. if x is None else x
    V, w = self.integral(self.V.value, self.w.value, t, x, dt)
    spike = self.V_th <= V
    self.V.value = bm.where(spike, self.V_reset, V)
    self.w.value = bm.where(spike, w + self.b, w)
    self.spike.value = spike
    return spike

  def clear_input(self):
    if self.input_var:
      self.input[:] = 0.


class GIF(NeuGroupNS):
  r"""Generalized Integrate-and-Fire model.

  **Model Descriptions**

  The generalized integrate-and-fire model [1]_ is given by

  .. math::

      &\frac{d I_j}{d t} = - k_j I_j

      &\frac{d V}{d t} = ( - (V - V_{rest}) + R\sum_{j}I_j + RI) / \tau

      &\frac{d V_{th}}{d t} = a(V - V_{rest}) - b(V_{th} - V_{th\infty})

  When :math:`V` meet :math:`V_{th}`, Generalized IF neuron fires:

  .. math::

      &I_j \leftarrow R_j I_j + A_j

      &V \leftarrow V_{reset}

      &V_{th} \leftarrow max(V_{th_{reset}}, V_{th})

  Note that :math:`I_j` refers to arbitrary number of internal currents.

  **Model Examples**

  - `Detailed examples to reproduce different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Niebur_2009_GIF.html>`_

  **Model Parameters**

  ============= ============== ======== ====================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- --------------------------------------------------------------------
  V_rest        -70            mV       Resting potential.
  V_reset       -70            mV       Reset potential after spike.
  V_th_inf      -50            mV       Target value of threshold potential :math:`V_{th}` updating.
  V_th_reset    -60            mV       Free parameter, should be larger than :math:`V_{reset}`.
  R             20             \        Membrane resistance.
  tau           20             ms       Membrane time constant. Compute by :math:`R * C`.
  a             0              \        Coefficient describes the dependence of
                                        :math:`V_{th}` on membrane potential.
  b             0.01           \        Coefficient describes :math:`V_{th}` update.
  k1            0.2            \        Constant pf :math:`I1`.
  k2            0.02           \        Constant of :math:`I2`.
  R1            0              \        Free parameter.
                                        Describes dependence of :math:`I_1` reset value on
                                        :math:`I_1` value before spiking.
  R2            1              \        Free parameter.
                                        Describes dependence of :math:`I_2` reset value on
                                        :math:`I_2` value before spiking.
  A1            0              \        Free parameter.
  A2            0              \        Free parameter.
  ============= ============== ======== ====================================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                  -70               Membrane potential.
  input              0                 External and synaptic input current.
  spike              False             Flag to mark whether the neuron is spiking.
  V_th               -50               Spiking threshold potential.
  I1                 0                 Internal current 1.
  I2                 0                 Internal current 2.
  t_last_spike       -1e7              Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] Mihalaş, Ştefan, and Ernst Niebur. "A generalized linear
         integrate-and-fire neural model produces diverse spiking
         behaviors." Neural computation 21.3 (2009): 704-718.
  .. [2] Teeter, Corinne, Ramakrishnan Iyer, Vilas Menon, Nathan
         Gouwens, David Feng, Jim Berg, Aaron Szafer et al. "Generalized
         leaky integrate-and-fire models classify multiple neuron types."
         Nature communications 9, no. 1 (2018): 1-15.
  """

  def __init__(
      self,
      size: Shape,
      V_rest: Union[float, ArrayType, Initializer, Callable] = -70.,
      V_reset: Union[float, ArrayType, Initializer, Callable] = -70.,
      V_th_inf: Union[float, ArrayType, Initializer, Callable] = -50.,
      V_th_reset: Union[float, ArrayType, Initializer, Callable] = -60.,
      R: Union[float, ArrayType, Initializer, Callable] = 20.,
      tau: Union[float, ArrayType, Initializer, Callable] = 20.,
      a: Union[float, ArrayType, Initializer, Callable] = 0.,
      b: Union[float, ArrayType, Initializer, Callable] = 0.01,
      k1: Union[float, ArrayType, Initializer, Callable] = 0.2,
      k2: Union[float, ArrayType, Initializer, Callable] = 0.02,
      R1: Union[float, ArrayType, Initializer, Callable] = 0.,
      R2: Union[float, ArrayType, Initializer, Callable] = 1.,
      A1: Union[float, ArrayType, Initializer, Callable] = 0.,
      A2: Union[float, ArrayType, Initializer, Callable] = 0.,
      V_initializer: Union[Initializer, Callable, ArrayType] = OneInit(-70.),
      I1_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),
      I2_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),
      Vth_initializer: Union[Initializer, Callable, ArrayType] = OneInit(-50.),
      noise: Union[float, ArrayType, Initializer, Callable] = None,
      method: str = 'exp_auto',
      keep_size: bool = False,
      input_var: bool = True,
      name: str = None,

      # parameter for training
      mode: bm.Mode = None,
      spike_fun: Callable = bm.surrogate.sigmoid,
  ):
    # initialization
    super().__init__(size=size,
                     keep_size=keep_size,
                     name=name,
                     mode=mode)
    is_subclass(self.mode, (bm.TrainingMode, bm.NonBatchingMode))

    # params
    self.V_rest = parameter(V_rest, self.varshape, allow_none=False)
    self.V_reset = parameter(V_reset, self.varshape, allow_none=False)
    self.V_th_inf = parameter(V_th_inf, self.varshape, allow_none=False)
    self.V_th_reset = parameter(V_th_reset, self.varshape, allow_none=False)
    self.R = parameter(R, self.varshape, allow_none=False)
    self.tau = parameter(tau, self.varshape, allow_none=False)
    self.a = parameter(a, self.varshape, allow_none=False)
    self.b = parameter(b, self.varshape, allow_none=False)
    self.k1 = parameter(k1, self.varshape, allow_none=False)
    self.k2 = parameter(k2, self.varshape, allow_none=False)
    self.R1 = parameter(R1, self.varshape, allow_none=False)
    self.R2 = parameter(R2, self.varshape, allow_none=False)
    self.A1 = parameter(A1, self.varshape, allow_none=False)
    self.A2 = parameter(A2, self.varshape, allow_none=False)
    self.noise = init_noise(noise, self.varshape, num_vars=4)
    self.spike_fun = is_callable(spike_fun, 'spike_fun')
    self.input_var = input_var

    # initializers
    self._V_initializer = is_initializer(V_initializer, 'V_initializer')
    self._I1_initializer = is_initializer(I1_initializer, 'I1_initializer')
    self._I2_initializer = is_initializer(I2_initializer, 'I2_initializer')
    self._Vth_initializer = is_initializer(Vth_initializer, 'Vth_initializer')

    # variables
    self.reset_state(self.mode)

    # integral
    if self.noise is None:
      self.integral = odeint(method=method, f=self.derivative)
    else:
      self.integral = sdeint(method=method, f=self.derivative, g=self.noise)

  def reset_state(self, batch_size=None):
    self.V = variable_(self._V_initializer, self.varshape, batch_size)
    self.I1 = variable_(self._I1_initializer, self.varshape, batch_size)
    self.I2 = variable_(self._I2_initializer, self.varshape, batch_size)
    self.V_th = variable_(self._Vth_initializer, self.varshape, batch_size)
    if self.input_var:
      self.input = variable_(bm.zeros, self.varshape, batch_size)
    sp_type = bm.float_ if self.mode.is_a(bm.TrainingMode) else bool
    self.spike = variable_(lambda s: bm.zeros(s, dtype=sp_type), self.varshape, batch_size)

  def dI1(self, I1, t):
    return - self.k1 * I1

  def dI2(self, I2, t):
    return - self.k2 * I2

  def dVth(self, V_th, t, V):
    return self.a * (V - self.V_rest) - self.b * (V_th - self.V_th_inf)

  def dV(self, V, t, I1, I2, I_ext):
    return (- (V - self.V_rest) + self.R * (I_ext + I1 + I2)) / self.tau

  @property
  def derivative(self):
    return JointEq(self.dI1, self.dI2, self.dVth, self.dV)

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    if self.input_var:
      if x is not None:
        self.input += x
      x = self.input.value
    else:
      x = 0. if x is None else x
    I1, I2, V_th, V = self.integral(self.I1.value, self.I2.value, self.V_th.value, self.V.value, t, x, dt)

    # spike and resets
    if isinstance(self.mode, bm.TrainingMode):
      spike = self.spike_fun(V - self.V_th)
      V += (self.V_reset - V) * spike
      I1 += spike * (self.R1 * I1 + self.A1 - I1)
      I2 += spike * (self.R2 * I2 + self.A2 - I2)
      reset_th = self.spike_fun(self.V_th_reset - V_th) * spike
      V_th += reset_th * (self.V_th_reset - V_th)
    else:
      spike = self.V_th <= V
      V = bm.where(spike, self.V_reset, V)
      I1 = bm.where(spike, self.R1 * I1 + self.A1, I1)
      I2 = bm.where(spike, self.R2 * I2 + self.A2, I2)
      V_th = bm.where(spike, bm.maximum(self.V_th_reset, V_th), V_th)
    self.spike.value = spike
    self.I1.value = I1
    self.I2.value = I2
    self.V_th.value = V_th
    self.V.value = V
    return spike

  def clear_input(self):
    if self.input_var:
      self.input[:] = 0.


class ALIFBellec2020(NeuGroupNS):
  r"""Leaky Integrate-and-Fire model with SFA [1]_.

  This model is similar to the GLIF2 model in the Technical White Paper
  on generalized LIF (GLIF) models from AllenInstitute [2]_.

  Formally, this model is given by:

  .. math::

     \tau \dot{V} = -(V - V_{\mathrm{rest}}) + R*I \\
     \tau_a \dot{a} = -a

  Once a spike is induced by :math:`V(t) > V_{\mathrm{th}} + \beta a`, then

  .. math::

     V \gets V - V_{\mathrm{th}} \\
     a \gets a + 1


  References
  ----------
  .. [1] Bellec, Guillaume, et al. "A solution to the learning dilemma for
         recurrent networks of spiking neurons."
         Nature communications 11.1 (2020): 1-15.
  .. [2] Allen Institute: Cell Types Database. © 2018 Allen Institute for
         Brain Science. Allen Cell Types Database, cell feature search.
         Available from: celltypes.brain-map.org/data (2018).
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,

      # model parameters
      V_rest: Union[float, ArrayType, Initializer, Callable] = -70.,
      V_th: Union[float, ArrayType, Initializer, Callable] = -60.,
      R: Union[float, ArrayType, Initializer, Callable] = 1.,
      beta: Union[float, ArrayType, Initializer, Callable] = 1.6,
      tau: Union[float, ArrayType, Initializer, Callable] = 20.,
      tau_a: Union[float, ArrayType, Initializer, Callable] = 2000.,
      tau_ref: Union[float, ArrayType, Initializer, Callable] = None,
      noise: Union[float, ArrayType, Initializer, Callable] = None,

      # initializers
      V_initializer: Union[Initializer, Callable, ArrayType] = OneInit(-70.),
      a_initializer: Union[Initializer, Callable, ArrayType] = OneInit(-50.),

      # parameter for training
      spike_fun: Callable = bm.surrogate.relu_grad,
      input_var: bool = True,

      # other parameters
      method: str = 'exp_auto',
      name: str = None,
      mode: bm.Mode = None,
      eprop: bool = False
  ):
    super().__init__(name=name,
                     size=size,
                     keep_size=keep_size,
                     mode=mode)
    is_subclass(self.mode, (bm.TrainingMode, bm.NonBatchingMode))

    # parameters
    self.V_rest = parameter(V_rest, self.varshape, allow_none=False)
    self.V_th = parameter(V_th, self.varshape, allow_none=False)
    self.R = parameter(R, self.varshape, allow_none=False)
    self.beta = parameter(beta, self.varshape, allow_none=False)
    self.tau = parameter(tau, self.varshape, allow_none=False)
    self.tau_a = parameter(tau_a, self.varshape, allow_none=False)
    self.tau_ref = parameter(tau_ref, self.varshape, allow_none=True)
    self.noise = init_noise(noise, self.varshape, num_vars=2)
    self.spike_fun = is_callable(spike_fun, 'spike_fun')
    self.eprop = eprop
    self.input_var = input_var

    # initializers
    self._V_initializer = is_initializer(V_initializer, 'V_initializer')
    self._a_initializer = is_initializer(a_initializer, 'a_initializer')

    # variables
    self.reset_state(self.mode)

    # integral
    if self.noise is None:
      self.integral = odeint(method=method, f=self.derivative)
    else:
      self.integral = sdeint(method=method, f=self.derivative, g=self.noise)

  def da(self, a, t):
    return -a / self.tau_a

  def dV(self, V, t, I_ext):
    return (- (V - self.V_rest) + self.R * I_ext) / self.tau

  @property
  def derivative(self):
    return JointEq([self.dV, self.da])

  def reset_state(self, batch_size=None):
    self.a = variable_(self._a_initializer, self.varshape, batch_size)
    self.V = variable_(self._V_initializer, self.varshape, batch_size)
    if self.input_var:
      self.input = variable_(bm.zeros, self.varshape, batch_size)
    sp_type = bm.float_ if isinstance(self.mode, bm.TrainingMode) else bool
    self.spike = variable_(lambda s: bm.zeros(s, dtype=sp_type), self.varshape, batch_size)
    if self.tau_ref is not None:
      self.t_last_spike = variable_(lambda s: bm.ones(s) * -1e7, self.varshape, batch_size)
      self.refractory = variable_(lambda s: bm.zeros(s, dtype=bool), self.varshape, batch_size)

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    if self.input_var:
      if x is not None:
        self.input += x
      x = self.input.value
    else:
      x = 0. if x is None else x
    V, a = self.integral(self.V, self.a, t, x, dt)

    if self.tau_ref is not None:
      # refractory
      refractory = (t - self.t_last_spike) <= self.tau_ref
      if isinstance(self.mode, bm.TrainingMode):
        refractory = stop_gradient(refractory)
      V = bm.where(refractory, self.V_reset, V)
      # spike and reset
      if isinstance(self.mode, bm.TrainingMode):
        spike = self.spike_fun((V - self.V_th - self.beta * self.a) / self.V_th)
        V -= self.V_th * (stop_gradient(spike) if self.eprop else spike)
        # will be used in other place, like Delta Synapse, so stop its gradient
        spike_ = spike > 0.
        refractory = stop_gradient(bm.logical_or(refractory, spike_))
        t_last_spike = stop_gradient(bm.where(spike_, t, self.t_last_spike.value))
      else:
        spike = V >= (self.V_th + self.beta * self.a)
        refractory = bm.logical_or(refractory, spike)
        t_last_spike = bm.where(spike, t, self.t_last_spike.value)
        V -= self.V_th * spike
      self.refractory.value = refractory
      self.t_last_spike.value = t_last_spike

    else:
      # spike and reset
      if isinstance(self.mode, bm.TrainingMode):
        spike = self.spike_fun((V - self.V_th - self.beta * self.a) / self.V_th)
        V -= self.V_th * (stop_gradient(spike) if self.eprop else spike)
      else:
        spike = V >= (self.V_th + self.beta * self.a)
        V -= self.V_th * spike
    self.spike.value = spike
    self.V.value = V
    self.a.value = a + spike
    return spike

  def clear_input(self):
    if self.input_var:
      self.input[:] = 0.


class Izhikevich(NeuGroupNS):
  r"""The Izhikevich neuron model.

  **Model Descriptions**

  The dynamics of the Izhikevich neuron model [1]_ [2]_ is given by:

  .. math ::

      \frac{d V}{d t} &= 0.04 V^{2}+5 V+140-u+I

      \frac{d u}{d t} &=a(b V-u)

  .. math ::

      \text{if}  v \geq 30  \text{mV}, \text{then}
      \begin{cases} v \leftarrow c \\
      u \leftarrow u+d \end{cases}

  **Model Examples**

  - `Detailed examples to reproduce different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Izhikevich_2003_Izhikevich_model.html>`_

  **Model Parameters**

  ============= ============== ======== ================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- --------------------------------------------------------------------------------
  a             0.02           \        It determines the time scale of
                                        the recovery variable :math:`u`.
  b             0.2            \        It describes the sensitivity of the
                                        recovery variable :math:`u` to
                                        the sub-threshold fluctuations of the
                                        membrane potential :math:`v`.
  c             -65            \        It describes the after-spike reset value
                                        of the membrane potential :math:`v` caused by
                                        the fast high-threshold :math:`K^{+}`
                                        conductance.
  d             8              \        It describes after-spike reset of the
                                        recovery variable :math:`u`
                                        caused by slow high-threshold
                                        :math:`Na^{+}` and :math:`K^{+}` conductance.
  tau_ref       0              ms       Refractory period length. [ms]
  V_th          30             mV       The membrane potential threshold.
  ============= ============== ======== ================================================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                          -65        Membrane potential.
  u                          1          Recovery variable.
  input                      0          External and synaptic input current.
  spike                      False      Flag to mark whether the neuron is spiking.
  refractory                False       Flag to mark whether the neuron is in refractory period.
  t_last_spike               -1e7       Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] Izhikevich, Eugene M. "Simple model of spiking neurons." IEEE
         Transactions on neural networks 14.6 (2003): 1569-1572.

  .. [2] Izhikevich, Eugene M. "Which model to use for cortical spiking neurons?."
         IEEE transactions on neural networks 15.5 (2004): 1063-1070.
  """

  def __init__(
      self,
      size: Shape,
      a: Union[float, ArrayType, Initializer, Callable] = 0.02,
      b: Union[float, ArrayType, Initializer, Callable] = 0.20,
      c: Union[float, ArrayType, Initializer, Callable] = -65.,
      d: Union[float, ArrayType, Initializer, Callable] = 8.,
      V_th: Union[float, ArrayType, Initializer, Callable] = 30.,
      tau_ref: Union[float, ArrayType, Initializer, Callable] = None,
      V_initializer: Union[Initializer, Callable, ArrayType] = None,
      u_initializer: Union[Initializer, Callable, ArrayType] = None,
      noise: Union[float, ArrayType, Initializer, Callable] = None,
      method: str = 'exp_auto',
      mode: bm.Mode = None,
      spike_fun: Callable = bm.surrogate.inv_square_grad,
      keep_size: bool = False,
      input_var: bool = True,
      ref_var: bool = False,
      name: str = None
  ):
    # initialization
    super().__init__(size=size,
                     keep_size=keep_size,
                     name=name,
                     mode=mode)
    is_subclass(self.mode, (bm.TrainingMode, bm.NonBatchingMode))

    # params
    self.a = parameter(a, self.varshape, allow_none=False)
    self.b = parameter(b, self.varshape, allow_none=False)
    self.c = parameter(c, self.varshape, allow_none=False)
    self.d = parameter(d, self.varshape, allow_none=False)
    self.V_th = parameter(V_th, self.varshape, allow_none=False)
    self.tau_ref = parameter(tau_ref, self.varshape, allow_none=True)
    self.noise = init_noise(noise, self.varshape, num_vars=2)
    self.spike_fun = is_callable(spike_fun, 'spike_fun')
    self.input_var = input_var
    self.ref_var = ref_var

    # initializers
    self._V_initializer = is_initializer(V_initializer, allow_none=True)
    self._u_initializer = is_initializer(u_initializer, allow_none=True)

    # variables
    self.reset_state(self.mode)

    # functions
    if self.noise is None:
      self.integral = odeint(method=method, f=JointEq([self.dV, self.du]))
    else:
      self.integral = sdeint(method=method, f=JointEq([self.dV, self.du]), g=self.noise)

  def reset_state(self, batch_size=None):
    v_init = OneInit(-70.) if self._V_initializer is None else self._V_initializer
    self.V = variable_(v_init, self.varshape, batch_size)
    u_init = OneInit(self.b * self.V) if self._u_initializer is None else self._u_initializer
    self.u = variable_(u_init, self.varshape, batch_size)
    if self.input_var:
      self.input = variable_(bm.zeros, self.varshape, batch_size)
    sp_type = bm.float_ if isinstance(self.mode, bm.TrainingMode) else bool
    self.spike = variable_(lambda s: bm.zeros(s, dtype=sp_type), self.varshape, batch_size)
    if self.tau_ref is not None:
      self.t_last_spike = variable_(lambda s: bm.ones(s) * -1e7, self.varshape, batch_size)
      if self.ref_var:
        self.refractory = variable_(lambda s: bm.zeros(s, dtype=bool), self.varshape, batch_size)

  def dV(self, V, t, u, I_ext):
    dVdt = 0.04 * V * V + 5 * V + 140 - u + I_ext
    return dVdt

  def du(self, u, t, V):
    dudt = self.a * (self.b * V - u)
    return dudt

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    if self.input_var:
      if x is not None:
        self.input += x
      x = self.input.value
    else:
      x = 0. if x is None else x
    V, u = self.integral(self.V.value, self.u.value, t, x, dt)

    if self.tau_ref is not None:
      refractory = bm.as_jax((t - self.t_last_spike) <= self.tau_ref)
      refractory = stop_gradient(refractory)
      V = bm.where(refractory, self.V_reset, V)

      # spike, refractory, and reset membrane potential
      if isinstance(self.mode, bm.TrainingMode):
        spike = self.spike_fun(V - self.V_th)
        spike_no_grad = stop_gradient(spike)
        V += spike_no_grad * (self.c - self.V_th)
        u += spike_no_grad * self.d
        t_last_spike = stop_gradient(bm.where(spike_no_grad, t, self.t_last_spike.value))
        if self.ref_var:
          self.refractory.value = stop_gradient(bm.logical_or(refractory, spike_no_grad > 0.))
      else:
        spike = self.V_th <= V
        V = bm.where(spike, self.c, V)
        u = bm.where(spike, u + self.d, u)
        t_last_spike = bm.where(spike, t, self.t_last_spike.value)
        if self.ref_var:
          self.refractory.value = bm.logical_or(refractory, spike)
      self.t_last_spike.value = t_last_spike

    else:
      # spike, refractory, and reset membrane potential
      if isinstance(self.mode, bm.TrainingMode):
        spike = self.spike_fun(V - self.V_th)
        spike_no_grad = stop_gradient(spike)
        V += spike_no_grad * (self.c - self.V_th)
        u += spike_no_grad * self.d
      else:
        spike = self.V_th <= V
        V = bm.where(spike, self.c, V)
        u = bm.where(spike, u + self.d, u)

    # finally
    self.V.value = V
    self.u.value = u
    self.spike.value = spike
    return spike

  def clear_input(self):
    if self.input_var:
      self.input[:] = 0.


class HindmarshRose(NeuGroupNS):
  r"""Hindmarsh-Rose neuron model.

  **Model Descriptions**

  The Hindmarsh–Rose model [1]_ [2]_ of neuronal activity is aimed to study the
  spiking-bursting behavior of the membrane potential observed in experiments
  made with a single neuron.

  The model has the mathematical form of a system of three nonlinear ordinary
  differential equations on the dimensionless dynamical variables :math:`x(t)`,
  :math:`y(t)`, and :math:`z(t)`. They read:

  .. math::

     \begin{aligned}
     \frac{d V}{d t} &= y - a V^3 + b V^2 - z + I \\
     \frac{d y}{d t} &= c - d V^2 - y \\
     \frac{d z}{d t} &= r (s (V - V_{rest}) - z)
     \end{aligned}

  where :math:`a, b, c, d` model the working of the fast ion channels,
  :math:`I` models the slow ion channels.

  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy.math as bm
    >>> import brainpy as bp
    >>> import matplotlib.pyplot as plt
    >>>
    >>> bp.math.set_dt(dt=0.01)
    >>> bp.ode.set_default_odeint('rk4')
    >>>
    >>> types = ['quiescence', 'spiking', 'bursting', 'irregular_spiking', 'irregular_bursting']
    >>> bs = bm.array([1.0, 3.5, 2.5, 2.95, 2.8])
    >>> Is = bm.array([2.0, 5.0, 3.0, 3.3, 3.7])
    >>>
    >>> # define neuron type
    >>> group = bp.neurons.HindmarshRose(len(types), b=bs)
    >>> runner = bp.DSRunner(group, monitors=['V'], inputs=['input', Is],)
    >>> runner.run(1e3)
    >>>
    >>> fig, gs = bp.visualize.get_figure(row_num=3, col_num=2, row_len=3, col_len=5)
    >>> for i, mode in enumerate(types):
    >>>     fig.add_subplot(gs[i // 2, i % 2])
    >>>     plt.plot(runner.mon.ts, runner.mon.V[:, i])
    >>>     plt.title(mode)
    >>>     plt.xlabel('Time [ms]')
    >>> plt.show()

  **Model Parameters**

  ============= ============== ========= ============================================================
  **Parameter** **Init Value** **Unit**  **Explanation**
  ------------- -------------- --------- ------------------------------------------------------------
  a             1              \         Model parameter.
                                         Fixed to a value best fit neuron activity.
  b             3              \         Model parameter.
                                         Allows the model to switch between bursting
                                         and spiking, controls the spiking frequency.
  c             1              \         Model parameter.
                                         Fixed to a value best fit neuron activity.
  d             5              \         Model parameter.
                                         Fixed to a value best fit neuron activity.
  r             0.01           \         Model parameter.
                                         Controls slow variable z's variation speed.
                                         Governs spiking frequency when spiking, and affects the
                                         number of spikes per burst when bursting.
  s             4              \         Model parameter. Governs adaption.
  ============= ============== ========= ============================================================

  **Model Variables**

  =============== ================= =====================================
  **Member name** **Initial Value** **Explanation**
  --------------- ----------------- -------------------------------------
  V               -1.6              Membrane potential.
  y               -10               Gating variable.
  z               0                 Gating variable.
  spike           False             Whether generate the spikes.
  input           0                 External and synaptic input current.
  t_last_spike    -1e7              Last spike time stamp.
  =============== ================= =====================================

  **References**

  .. [1] Hindmarsh, James L., and R. M. Rose. "A model of neuronal bursting using
        three coupled first order differential equations." Proceedings of the
        Royal society of London. Series B. Biological sciences 221.1222 (1984):
        87-102.
  .. [2] Storace, Marco, Daniele Linaro, and Enno de Lange. "The Hindmarsh–Rose
        neuron model: bifurcation analysis and piecewise-linear approximations."
        Chaos: An Interdisciplinary Journal of Nonlinear Science 18.3 (2008):
        033128.
  """

  def __init__(
      self,
      size: Shape,
      a: Union[float, ArrayType, Initializer, Callable] = 1.,
      b: Union[float, ArrayType, Initializer, Callable] = 3.,
      c: Union[float, ArrayType, Initializer, Callable] = 1.,
      d: Union[float, ArrayType, Initializer, Callable] = 5.,
      r: Union[float, ArrayType, Initializer, Callable] = 0.01,
      s: Union[float, ArrayType, Initializer, Callable] = 4.,
      V_rest: Union[float, ArrayType, Initializer, Callable] = -1.6,
      V_th: Union[float, ArrayType, Initializer, Callable] = 1.0,
      V_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),
      y_initializer: Union[Initializer, Callable, ArrayType] = OneInit(-10.),
      z_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),
      noise: Union[float, ArrayType, Initializer, Callable] = None,
      method: str = 'exp_auto',
      keep_size: bool = False,
      input_var: bool = True,
      name: str = None,

      # parameters for training
      mode: bm.Mode = None,
      spike_fun: Callable = bm.surrogate.inv_square_grad,
  ):
    # initialization
    super(HindmarshRose, self).__init__(size=size,
                                        keep_size=keep_size,
                                        name=name,
                                        mode=mode)
    is_subclass(self.mode, (bm.TrainingMode, bm.NonBatchingMode))

    # parameters
    self.a = parameter(a, self.varshape, allow_none=False)
    self.b = parameter(b, self.varshape, allow_none=False)
    self.c = parameter(c, self.varshape, allow_none=False)
    self.d = parameter(d, self.varshape, allow_none=False)
    self.r = parameter(r, self.varshape, allow_none=False)
    self.s = parameter(s, self.varshape, allow_none=False)
    self.V_th = parameter(V_th, self.varshape, allow_none=False)
    self.V_rest = parameter(V_rest, self.varshape, allow_none=False)
    self.noise = init_noise(noise, self.varshape, num_vars=3)
    self.spike_fun = is_callable(spike_fun, 'spike_fun')
    self.input_var = input_var

    # variables
    is_initializer(V_initializer, 'V_initializer', allow_none=False)
    is_initializer(y_initializer, 'y_initializer', allow_none=False)
    is_initializer(z_initializer, 'z_initializer', allow_none=False)
    self._V_initializer = V_initializer
    self._y_initializer = y_initializer
    self._z_initializer = z_initializer

    # variables
    self.reset_state(self.mode)

    # integral
    if self.noise is None:
      self.integral = odeint(method=method, f=self.derivative)
    else:
      self.integral = sdeint(method=method, f=self.derivative, g=self.noise)

  def reset_state(self, batch_size=None):
    self.V = variable_(self._V_initializer, self.varshape, batch_size)
    self.y = variable_(self._y_initializer, self.varshape, batch_size)
    self.z = variable_(self._z_initializer, self.varshape, batch_size)
    if self.input_var:
      self.input = variable_(bm.zeros, self.varshape, batch_size)
    sp_type = bm.float_ if isinstance(self.mode, bm.TrainingMode) else bool
    self.spike = variable_(lambda s: bm.zeros(s, dtype=sp_type), self.varshape, batch_size)

  def dV(self, V, t, y, z, I_ext):
    return y - self.a * V * V * V + self.b * V * V - z + I_ext

  def dy(self, y, t, V):
    return self.c - self.d * V * V - y

  def dz(self, z, t, V):
    return self.r * (self.s * (V - self.V_rest) - z)

  @property
  def derivative(self):
    return JointEq([self.dV, self.dy, self.dz])

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    if self.input_var:
      if x is not None:
        self.input += x
      x = self.input.value
    else:
      x = 0. if x is None else x
    V, y, z = self.integral(self.V, self.y, self.z, t, x, dt=dt)
    if isinstance(self.mode, bm.TrainingMode):
      self.spike.value = self.spike_fun(V - self.V_th, self.V - self.V_th)
    else:
      self.spike.value = bm.logical_and(V >= self.V_th, self.V < self.V_th)
    self.V.value = V
    self.y.value = y
    self.z.value = z
    return self.spike.value

  def clear_input(self):
    if self.input_var:
      self.input[:] = 0.


class FHN(NeuGroupNS):
  r"""FitzHugh-Nagumo neuron model.

  **Model Descriptions**

  The FitzHugh–Nagumo model (FHN), named after Richard FitzHugh (1922–2007)
  who suggested the system in 1961 [1]_ and J. Nagumo et al. who created the
  equivalent circuit the following year, describes a prototype of an excitable
  system (e.g., a neuron).

  The motivation for the FitzHugh-Nagumo model was to isolate conceptually
  the essentially mathematical properties of excitation and propagation from
  the electrochemical properties of sodium and potassium ion flow. The model
  consists of

  - a *voltage-like variable* having cubic nonlinearity that allows regenerative
    self-excitation via a positive feedback, and
  - a *recovery variable* having a linear dynamics that provides a slower negative feedback.

  .. math::

     \begin{aligned}
     {\dot {v}} &=v-{\frac {v^{3}}{3}}-w+RI_{\rm {ext}},  \\
     \tau {\dot  {w}}&=v+a-bw.
     \end{aligned}

  The FHN Model is an example of a relaxation oscillator
  because, if the external stimulus :math:`I_{\text{ext}}`
  exceeds a certain threshold value, the system will exhibit
  a characteristic excursion in phase space, before the
  variables :math:`v` and :math:`w` relax back to their rest values.
  This behaviour is typical for spike generations (a short,
  nonlinear elevation of membrane voltage :math:`v`,
  diminished over time by a slower, linear recovery variable
  :math:`w`) in a neuron after stimulation by an external
  input current.

  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> fhn = bp.neurons.FHN(1)
    >>> runner = bp.DSRunner(fhn, inputs=('input', 1.), monitors=['V', 'w'])
    >>> runner.run(100.)
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.w, legend='w')
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', show=True)

  **Model Parameters**

  ============= ============== ======== ========================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------
  a             1              \        Positive constant
  b             1              \        Positive constant
  tau           10             ms       Membrane time constant.
  V_th          1.8            mV       Threshold potential of spike.
  ============= ============== ======== ========================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                   0                 Membrane potential.
  w                   0                 A recovery variable which represents
                                        the combined effects of sodium channel
                                        de-inactivation and potassium channel
                                        deactivation.
  input               0                 External and synaptic input current.
  spike               False             Flag to mark whether the neuron is spiking.
  t_last_spike       -1e7               Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] FitzHugh, Richard. "Impulses and physiological states in theoretical models of nerve membrane." Biophysical journal 1.6 (1961): 445-466.
  .. [2] https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model
  .. [3] http://www.scholarpedia.org/article/FitzHugh-Nagumo_model

  """

  def __init__(
      self,
      size: Shape,
      a: Union[float, ArrayType, Initializer, Callable] = 0.7,
      b: Union[float, ArrayType, Initializer, Callable] = 0.8,
      tau: Union[float, ArrayType, Initializer, Callable] = 12.5,
      Vth: Union[float, ArrayType, Initializer, Callable] = 1.8,
      V_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),
      w_initializer: Union[Initializer, Callable, ArrayType] = ZeroInit(),
      noise: Union[float, ArrayType, Initializer, Callable] = None,
      method: str = 'exp_auto',
      keep_size: bool = False,
      input_var: bool = True,
      name: str = None,

      # parameters for training
      mode: bm.Mode = None,
  ):
    # initialization
    super(FHN, self).__init__(size=size,
                              keep_size=keep_size,
                              name=name,
                              mode=mode)
    is_subclass(self.mode, (bm.NonBatchingMode,))

    # parameters
    self.a = parameter(a, self.varshape, allow_none=False)
    self.b = parameter(b, self.varshape, allow_none=False)
    self.tau = parameter(tau, self.varshape, allow_none=False)
    self.Vth = parameter(Vth, self.varshape, allow_none=False)
    self.noise = init_noise(noise, self.varshape, num_vars=2)
    self.input_var = input_var

    # initializers
    is_initializer(V_initializer, 'V_initializer')
    is_initializer(w_initializer, 'w_initializer')
    self._V_initializer = V_initializer
    self._w_initializer = w_initializer

    # variables
    self.reset_state(self.mode)

    # integral
    if self.noise is None:
      self.integral = odeint(method=method, f=self.derivative)
    else:
      self.integral = sdeint(method=method, f=self.derivative, g=self.noise)

  def reset_state(self, batch_size=None):
    self.V = variable_(self._V_initializer, self.varshape, batch_size)
    self.w = variable_(self._w_initializer, self.varshape, batch_size)
    if self.input_var:
      self.input = variable_(bm.zeros, self.varshape, batch_size)
    self.spike = variable_(lambda s: bm.zeros(s, dtype=bool), self.varshape, batch_size)

  def dV(self, V, t, w, I_ext):
    return V - V * V * V / 3 - w + I_ext

  def dw(self, w, t, V):
    return (V + self.a - self.b * w) / self.tau

  @property
  def derivative(self):
    return JointEq([self.dV, self.dw])

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    if self.input_var:
      if x is not None:
        self.input += x
      x = self.input.value
    else:
      x = 0. if x is None else x
    V, w = self.integral(self.V.value, self.w.value, t, x, dt=dt)
    self.spike.value = bm.logical_and(V >= self.Vth, self.V < self.Vth)
    self.V.value = V
    self.w.value = w
    return self.spike.value

  def clear_input(self):
    if self.input_var:
      self.input[:] = 0.


class LIF_SFA_Bellec2020(NeuGroupNS):
  r"""Leaky Integrate-and-Fire model with SFA [1]_.

  This model is similar to the GLIF2 model in the Technical White Paper
  on generalized LIF (GLIF) models from AllenInstitute [2]_.

  Formally, this model is given by:

  .. math::

     \tau \dot{V} = -(V - V_{\mathrm{rest}}) + R*I \\
     \tau_a \dot{a} = -a

  Once a spike is induced by :math:`V(t) > V_{\mathrm{th}} + \beta a`, then

  .. math::

     V \gets V - V_{\mathrm{th}} \\
     a \gets a + 1


  References
  ----------
  .. [1] Bellec, Guillaume, et al. "A solution to the learning dilemma for
         recurrent networks of spiking neurons."
         Nature communications 11.1 (2020): 1-15.
  .. [2] Allen Institute: Cell Types Database. © 2018 Allen Institute for
         Brain Science. Allen Cell Types Database, cell feature search.
         Available from: celltypes.brain-map.org/data (2018).
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,

      # model parameters
      V_rest: Union[float, ArrayType, Initializer, Callable] = -70.,
      V_th: Union[float, ArrayType, Initializer, Callable] = -60.,
      R: Union[float, ArrayType, Initializer, Callable] = 1.,
      beta: Union[float, ArrayType, Initializer, Callable] = 1.6,
      tau: Union[float, ArrayType, Initializer, Callable] = 20.,
      tau_a: Union[float, ArrayType, Initializer, Callable] = 2000.,
      tau_ref: Union[float, ArrayType, Initializer, Callable] = None,

      # initializers
      V_initializer: Union[Initializer, Callable, ArrayType] = OneInit(-70.),
      a_initializer: Union[Initializer, Callable, ArrayType] = OneInit(-50.),

      # parameter for training
      spike_fun: Callable = bm.surrogate.relu_grad,

      # other parameters
      method: str = 'exp_auto',
      name: str = None,
      mode: bm.Mode = None,
  ):
    super().__init__(name=name, size=size, keep_size=keep_size, mode=mode)
    is_subclass(self.mode, (bm.TrainingMode, bm.NonBatchingMode))

    # parameters
    self.V_rest = parameter(V_rest, self.varshape, allow_none=False)
    self.V_th = parameter(V_th, self.varshape, allow_none=False)
    self.R = parameter(R, self.varshape, allow_none=False)
    self.beta = parameter(beta, self.varshape, allow_none=False)
    self.tau = parameter(tau, self.varshape, allow_none=False)
    self.tau_a = parameter(tau_a, self.varshape, allow_none=False)
    self.tau_ref = parameter(tau_ref, self.varshape, allow_none=True)
    self.spike_fun = is_callable(spike_fun, 'spike_fun')

    # initializers
    self._V_initializer = is_initializer(V_initializer)
    self._a_initializer = is_initializer(a_initializer)

    # variables
    self.reset_state(self.mode)

    # integral
    self.integral = odeint(method=method, f=self.derivative)

  def da(self, a, t):
    return -a / self.tau_a

  def dV(self, V, t, I_ext):
    return (- (V - self.V_rest) + self.R * I_ext) / self.tau

  @property
  def derivative(self):
    return JointEq([self.dV, self.da])

  def reset_state(self, batch_size=None):
    self.a = variable_(self._a_initializer, self.varshape, batch_size)
    self.V = variable_(self._V_initializer, self.varshape, batch_size)
    self.spike = variable_(bm.zeros, self.varshape, batch_size)
    if self.tau_ref is not None:
      self.t_last_spike = variable_(OneInit(-1e7), self.varshape, batch_size)

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    x = 0. if x is None else x

    # integral
    V, a = self.integral(self.V, self.a, t, x, dt)

    if self.tau_ref is not None:
      # refractory
      refractory = stop_gradient((t - self.t_last_spike) <= self.tau_ref)
      V = bm.where(refractory, self.V_reset, V)
      # spike and reset
      spike = self.spike_fun((V - self.V_th - self.beta * self.a) / self.V_th)
      V -= self.V_th * spike
      t_last_spike = stop_gradient(bm.where(spike, t, self.t_last_spike.value))
      self.t_last_spike.value = t_last_spike

    else:
      # spike and reset
      spike = self.spike_fun((V - self.V_th - self.beta * self.a) / self.V_th)
      V -= self.V_th * spike
    self.spike.value = spike
    self.V.value = V
    self.a.value = a + spike
    return spike
