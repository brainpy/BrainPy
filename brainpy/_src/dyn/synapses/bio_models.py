from typing import Union, Sequence, Callable, Optional

import jax.numpy
from brainpy import math as bm
from brainpy._src.context import share
from brainpy._src.dyn._docs import pneu_doc
from brainpy._src.dyn.base import SynDyn
from brainpy._src.integrators.joint_eq import JointEq
from brainpy._src.integrators.ode.generic import odeint
from brainpy.types import ArrayType

__all__ = [
  'AMPA',
  'GABAa',
  'BioNMDA',
]


class AMPA(SynDyn):
  r"""AMPA synapse model.

  **Model Descriptions**

  AMPA receptor is an ionotropic receptor, which is an ion channel.
  When it is bound by neurotransmitters, it will immediately open the
  ion channel, causing the change of membrane potential of postsynaptic neurons.

  A classical model is to use the Markov process to model ion channel switch.
  Here :math:`g` represents the probability of channel opening, :math:`1-g`
  represents the probability of ion channel closing, and :math:`\alpha` and
  :math:`\beta` are the transition probability. Because neurotransmitters can
  open ion channels, the transfer probability from :math:`1-g` to :math:`g`
  is affected by the concentration of neurotransmitters. We denote the concentration
  of neurotransmitters as :math:`[T]` and get the following Markov process.

  .. image:: ../../../_static/synapse_markov.png
      :align: center

  We obtained the following formula when describing the process by a differential equation.

  .. math::

      \frac{ds}{dt} =\alpha[T](1-g)-\beta g

  where :math:`\alpha [T]` denotes the transition probability from state :math:`(1-g)`
  to state :math:`(g)`; and :math:`\beta` represents the transition probability of
  the other direction. :math:`\alpha` is the binding constant. :math:`\beta` is the
  unbinding constant. :math:`[T]` is the neurotransmitter concentration, and
  has the duration of 0.5 ms.

  Moreover, the post-synaptic current on the post-synaptic neuron is formulated as

  .. math::

      I_{syn} = g_{max} g (V-E)

  where :math:`g_{max}` is the maximum conductance, and `E` is the reverse potential.

  .. [1] Vijayan S, Kopell N J. Thalamic model of awake alpha oscillations
         and implications for stimulus processing[J]. Proceedings of the
         National Academy of Sciences, 2012, 109(45): 18553-18558.

  Args:
    alpha: float, ArrayType, Callable. Binding constant.
    beta: float, ArrayType, Callable. Unbinding constant.
    T: float, ArrayType, Callable. Transmitter concentration when synapse is triggered by
      a pre-synaptic spike.. Default 1 [mM].
    T_dur: float, ArrayType, Callable. Transmitter concentration duration time after being triggered. Default 1 [ms]
    %s
  """

  supported_modes = (bm.NonBatchingMode, bm.BatchingMode)

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      sharding: Optional[Sequence[str]] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,

      # synapse parameters
      alpha: Union[float, ArrayType, Callable] = 0.98,
      beta: Union[float, ArrayType, Callable] = 0.18,
      T: Union[float, ArrayType, Callable] = 0.5,
      T_dur: Union[float, ArrayType, Callable] = 0.5,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.alpha = self.init_param(alpha)
    self.beta = self.init_param(beta)
    self.T = self.init_param(T)
    self.T_duration = self.init_param(T_dur)

    # functions
    self.integral = odeint(method=method, f=self.dg)

    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.g = self.init_variable(bm.zeros, batch_size)
    self.spike_arrival_time = self.init_variable(bm.ones, batch_size)
    self.spike_arrival_time.fill(-1e7)

  def dg(self, g, t, TT):
    return self.alpha * TT * (1 - g) - self.beta * g

  def update(self, pre_spike):
    t = share.load('t')
    dt = share.load('dt')
    self.spike_arrival_time.value = bm.where(pre_spike, t, self.spike_arrival_time)
    TT = ((t - self.spike_arrival_time) < self.T_duration) * self.T
    self.g.value = self.integral(self.g, t, TT, dt)
    return self.g.value

  def return_info(self):
    return self.g


AMPA.__doc__ = AMPA.__doc__ % (pneu_doc,)


class GABAa(AMPA):
  r"""GABAa synapse model.

  **Model Descriptions**

  GABAa synapse model has the same equation with the `AMPA synapse <./brainmodels.synapses.AMPA.rst>`_,

  .. math::

      \frac{d g}{d t}&=\alpha[T](1-g) - \beta g \\
      I_{syn}&= - g_{max} g (V - E)

  but with the difference of:

  - Reversal potential of synapse :math:`E` is usually low, typically -80. mV
  - Activating rate constant :math:`\alpha=0.53`
  - De-activating rate constant :math:`\beta=0.18`
  - Transmitter concentration :math:`[T]=1\,\mu ho(\mu S)` when synapse is
    triggered by a pre-synaptic spike, with the duration of 1. ms.

  .. [1] Destexhe, Alain, and Denis Paré. "Impact of network activity
         on the integrative properties of neocortical pyramidal neurons
         in vivo." Journal of neurophysiology 81.4 (1999): 1531-1547.

  Args:
    alpha: float, ArrayType, Callable. Binding constant. Default 0.062
    beta: float, ArrayType, Callable. Unbinding constant. Default 3.57
    T: float, ArrayType, Callable. Transmitter concentration when synapse is triggered by
      a pre-synaptic spike.. Default 1 [mM].
    T_dur: float, ArrayType, Callable. Transmitter concentration duration time
      after being triggered. Default 1 [ms]
    %s
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      sharding: Optional[Sequence[str]] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,

      # synapse parameters
      alpha: Union[float, ArrayType, Callable] = 0.53,
      beta: Union[float, ArrayType, Callable] = 0.18,
      T: Union[float, ArrayType, Callable] = 1.,
      T_dur: Union[float, ArrayType, Callable] = 1.,
  ):
    super().__init__(alpha=alpha,
                     beta=beta,
                     T=T,
                     T_dur=T_dur,
                     method=method,
                     name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)


GABAa.__doc__ = GABAa.__doc__ % (pneu_doc,)


class BioNMDA(SynDyn):
  r"""Biological NMDA synapse model.

  **Model Descriptions**

  The NMDA receptor is a glutamate receptor and ion channel found in neurons.
  The NMDA receptor is one of three types of ionotropic glutamate receptors,
  the other two being AMPA and kainate receptors.

  The NMDA receptor mediated conductance depends on the postsynaptic voltage.
  The voltage dependence is due to the blocking of the pore of the NMDA receptor
  from the outside by a positively charged magnesium ion. The channel is
  nearly completely blocked at resting potential, but the magnesium block is
  relieved if the cell is depolarized. The fraction of channels :math:`g_{\infty}`
  that are not blocked by magnesium can be fitted to

  .. math::

      g_{\infty}(V,[{Mg}^{2+}]_{o}) = (1+{e}^{-a V}
      \frac{[{Mg}^{2+}]_{o}} {b})^{-1}

  Here :math:`[{Mg}^{2+}]_{o}` is the extracellular magnesium concentration,
  usually 1 mM. Thus, the channel acts as a
  "coincidence detector" and only once both of these conditions are met, the
  channel opens and it allows positively charged ions (cations) to flow through
  the cell membrane [2]_.

  If we make the approximation that the magnesium block changes
  instantaneously with voltage and is independent of the gating of the channel,
  the net NMDA receptor-mediated synaptic current is given by

  .. math::

      I_{syn} = g_\mathrm{NMDA}(t) (V(t)-E) \cdot g_{\infty}

  where :math:`V(t)` is the post-synaptic neuron potential, :math:`E` is the
  reversal potential.

  Simultaneously, the kinetics of synaptic state :math:`g` is determined by a 2nd-order kinetics [1]_:

  .. math::

      & \frac{d g}{dt} = \alpha_1 x (1 - g) - \beta_1 g \\
      & \frac{d x}{dt} = \alpha_2 [T] (1 - x) - \beta_2 x

  where :math:`\alpha_1, \beta_1` refers to the conversion rate of variable g and
  :math:`\alpha_2, \beta_2` refers to the conversion rate of variable x.

  The NMDA receptor has been thought to be very important for controlling
  synaptic plasticity and mediating learning and memory functions [3]_.

  .. [1] Devaney A J . Mathematical Foundations of Neuroscience[M].
         Springer New York, 2010: 162.
  .. [2] Furukawa, Hiroyasu, Satinder K. Singh, Romina Mancusso, and
         Eric Gouaux. "Subunit arrangement and function in NMDA receptors."
         Nature 438, no. 7065 (2005): 185-192.
  .. [3] Li, F. and Tsien, J.Z., 2009. Memory and the NMDA receptors. The New
         England journal of medicine, 361(3), p.302.
  .. [4] https://en.wikipedia.org/wiki/NMDA_receptor


  Args:
    alpha1: float, ArrayType, Callable. The conversion rate of g from inactive to active. Default 2 ms^-1.
    beta1: float, ArrayType, Callable. The conversion rate of g from active to inactive. Default 0.01 ms^-1.
    alpha2: float, ArrayType, Callable. The conversion rate of x from inactive to active. Default 1 ms^-1.
    beta2: float, ArrayType, Callable. The conversion rate of x from active to inactive. Default 0.5 ms^-1.
    T: float, ArrayType, Callable. Transmitter concentration when synapse is
      triggered by a pre-synaptic spike. Default 1 [mM].
    T_dur: float, ArrayType, Callable. Transmitter concentration duration time after being triggered. Default 1 [ms]
    %s
  """
  supported_modes = (bm.NonBatchingMode, bm.BatchingMode)

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      sharding: Optional[Sequence[str]] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,

      # synapse parameters
      alpha1: Union[float, ArrayType, Callable] = 2.,
      beta1: Union[float, ArrayType, Callable] = 0.01,
      alpha2: Union[float, ArrayType, Callable] = 1.,
      beta2: Union[float, ArrayType, Callable] = 0.5,
      T: Union[float, ArrayType, Callable] = 1.,
      T_dur: Union[float, ArrayType, Callable] = 0.5,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.beta1 = self.init_param(beta1)
    self.beta2 = self.init_param(beta2)
    self.alpha1 = self.init_param(alpha1)
    self.alpha2 = self.init_param(alpha2)
    self.T = self.init_param(T)
    self.T_dur = self.init_param(T_dur)

    # integral
    self.integral = odeint(method=method, f=JointEq([self.dg, self.dx]))

    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.g = self.init_variable(bm.zeros, batch_size)
    self.x = self.init_variable(bm.zeros, batch_size)
    self.spike_arrival_time = self.init_variable(bm.ones, batch_size)
    self.spike_arrival_time.fill(-1e7)

  def dg(self, g, t, x):
    return self.alpha1 * x * (1 - g) - self.beta1 * g

  def dx(self, x, t, T):
    return self.alpha2 * T * (1 - x) - self.beta2 * x

  def update(self, pre_spike):
    t = share.load('t')
    dt = share.load('dt')
    self.spike_arrival_time.value = bm.where(pre_spike, t, self.spike_arrival_time)
    T = ((t - self.spike_arrival_time) < self.T_dur) * self.T
    self.g.value, self.x.value = self.integral(self.g, self.x, t, T, dt)
    return self.g.value

  def return_info(self):
    return self.g


BioNMDA.__doc__ = BioNMDA.__doc__ % (pneu_doc,)
