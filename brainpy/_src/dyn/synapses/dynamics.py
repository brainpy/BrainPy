from typing import Union, Sequence, Callable, Optional

from brainpy import math as bm
from brainpy._src.context import share
from brainpy._src.dyn._docs import pneu_doc
from brainpy._src.dyn.base import SynDyn
from brainpy._src.integrators.joint_eq import JointEq
from brainpy._src.integrators.ode.generic import odeint
from brainpy._src.mixin import AlignPost, ReturnInfo
from brainpy.types import ArrayType

__all__ = [
  'Expon',
  'DualExpon',
  'Alpha',
  'NMDA',
  'STD',
  'STP',
  'AMPA',
  'GABAa',
  'BioNMDA',
]



class Expon(SynDyn, AlignPost):
  r"""Exponential decay synapse model.

  **Model Descriptions**

  The single exponential decay synapse model assumes the release of neurotransmitter,
  its diffusion across the cleft, the receptor binding, and channel opening all happen
  very quickly, so that the channels instantaneously jump from the closed to the open state.
  Therefore, its expression is given by

  .. math::

      g_{\mathrm{syn}}(t)=g_{\mathrm{max}} e^{-\left(t-t_{0}\right) / \tau}

  where :math:`\tau_{delay}` is the time constant of the synaptic state decay,
  :math:`t_0` is the time of the pre-synaptic spike,
  :math:`g_{\mathrm{max}}` is the maximal conductance.

  Accordingly, the differential form of the exponential synapse is given by

  .. math::

      \begin{aligned}
       & \frac{d g}{d t} = -\frac{g}{\tau_{decay}}+\sum_{k} \delta(t-t_{j}^{k}).
       \end{aligned}

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.

  Args:
    tau: float, ArrayType, Callable. The time constant of decay. [ms]
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
      tau: Union[float, ArrayType, Callable] = 8.0,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.tau = self.init_param(tau)

    # function
    self.integral = odeint(self.derivative, method=method)

    self.reset_state(self.mode)

  def derivative(self, g, t):
    return -g / self.tau

  def reset_state(self, batch_size=None):
    self.g = self.init_variable(bm.zeros, batch_size)

  def update(self, x=None):
    t = share.load('t')
    dt = share.load('dt')
    self.g.value = self.integral(self.g.value, t, dt)
    if x is not None:
      self.g.value += x
    return self.g.value

  def add_current(self, x):
    self.g.value += x

  def return_info(self):
    return self.g


Expon.__doc__ = Expon.__doc__ % (pneu_doc,)


class DualExpon(SynDyn):
  r"""Dual exponential synapse model.

  **Model Descriptions**

  The dual exponential synapse model [1]_, also named as *difference of two exponentials* model,
  is given by:

  .. math::

    g_{\mathrm{syn}}(t)=g_{\mathrm{max}} \frac{\tau_{1} \tau_{2}}{
        \tau_{1}-\tau_{2}}\left(\exp \left(-\frac{t-t_{0}}{\tau_{1}}\right)
        -\exp \left(-\frac{t-t_{0}}{\tau_{2}}\right)\right)

  where :math:`\tau_1` is the time constant of the decay phase, :math:`\tau_2`
  is the time constant of the rise phase, :math:`t_0` is the time of the pre-synaptic
  spike, :math:`g_{\mathrm{max}}` is the maximal conductance.

  However, in practice, this formula is hard to implement. The equivalent solution is
  two coupled linear differential equations [2]_:

  .. math::

      \begin{aligned}
      &\frac{d g}{d t}=-\frac{g}{\tau_{\mathrm{decay}}}+h \\
      &\frac{d h}{d t}=-\frac{h}{\tau_{\text {rise }}}+ \delta\left(t_{0}-t\right),
      \end{aligned}

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
         "The Synapse." Principles of Computational Modelling in Neuroscience.
         Cambridge: Cambridge UP, 2011. 172-95. Print.
  .. [2] Roth, A., & Van Rossum, M. C. W. (2009). Modeling Synapses. Computational
         Modeling Methods for Neuroscientists.

  Args:
    tau_decay: float, ArrayArray, Callable. The time constant of the synaptic decay phase. [ms]
    tau_rise: float, ArrayArray, Callable. The time constant of the synaptic rise phase. [ms]
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
      tau_decay: Union[float, ArrayType, Callable] = 10.0,
      tau_rise: Union[float, ArrayType, Callable] = 1.,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.tau_rise = self.init_param(tau_rise)
    self.tau_decay = self.init_param(tau_decay)

    # integrator
    self.integral = odeint(JointEq(self.dg, self.dh), method=method)

    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.h = self.init_variable(bm.zeros, batch_size)
    self.g = self.init_variable(bm.zeros, batch_size)

  def dh(self, h, t):
    return -h / self.tau_rise

  def dg(self, g, t, h):
    return -g / self.tau_decay + h

  def update(self, x):
    t = share.load('t')
    dt = share.load('dt')

    # update synaptic variables
    self.g.value, self.h.value = self.integral(self.g.value, self.h.value, t, dt=dt)
    self.h += x
    return self.g.value

  def return_info(self):
    return self.g


DualExpon.__doc__ = DualExpon.__doc__ % (pneu_doc,)


class Alpha(DualExpon):
  r"""Alpha synapse model.

  **Model Descriptions**

  The analytical expression of alpha synapse is given by:

  .. math::

      g_{syn}(t)= g_{max} \frac{t-t_{s}}{\tau} \exp \left(-\frac{t-t_{s}}{\tau}\right).

  While, this equation is hard to implement. So, let's try to convert it into the
  differential forms:

  .. math::

      \begin{aligned}
      &\frac{d g}{d t}=-\frac{g}{\tau}+h \\
      &\frac{d h}{d t}=-\frac{h}{\tau}+\delta\left(t_{0}-t\right)
      \end{aligned}

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.

  Args:
    tau_decay: float, ArrayType, Callable. The time constant [ms] of the synaptic decay phase.
       The name of this synaptic projection.
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
      tau_decay: Union[float, ArrayType, Callable] = 10.0,
  ):
    super().__init__(
      tau_decay=tau_decay,
      tau_rise=tau_decay,
      method=method,
      name=name,
      mode=mode,
      size=size,
      keep_size=keep_size,
      sharding=sharding
    )


Alpha.__doc__ = Alpha.__doc__ % (pneu_doc,)


class NMDA(SynDyn):
  r"""NMDA synapse model.

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

      g_{\infty}(V,[{Mg}^{2+}]_{o}) = (1+{e}^{-\alpha V}
      \frac{[{Mg}^{2+}]_{o}} {\beta})^{-1}

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

  Simultaneously, the kinetics of synaptic state :math:`g` is given by

  .. math::

      & g_\mathrm{NMDA} (t) = g_{max} g \\
      & \frac{d g}{dt} = -\frac{g} {\tau_{decay}}+a x(1-g) \\
      & \frac{d x}{dt} = -\frac{x}{\tau_{rise}}+ \sum_{k} \delta(t-t_{j}^{k})

  where the decay time of NMDA currents is usually taken to be
  :math:`\tau_{decay}` =100 ms, :math:`a= 0.5 ms^{-1}`, and :math:`\tau_{rise}` =2 ms.

  The NMDA receptor has been thought to be very important for controlling
  synaptic plasticity and mediating learning and memory functions [3]_.


  .. [1] Brunel N, Wang X J. Effects of neuromodulation in a
         cortical network model of object working memory dominated
         by recurrent inhibition[J].
         Journal of computational neuroscience, 2001, 11(1): 63-85.
  .. [2] Furukawa, Hiroyasu, Satinder K. Singh, Romina Mancusso, and
         Eric Gouaux. "Subunit arrangement and function in NMDA receptors."
         Nature 438, no. 7065 (2005): 185-192.
  .. [3] Li, F. and Tsien, J.Z., 2009. Memory and the NMDA receptors. The New
         England journal of medicine, 361(3), p.302.
  .. [4] https://en.wikipedia.org/wiki/NMDA_receptor

  Args:
    tau_decay: float, ArrayType, Callable. The time constant of the synaptic decay phase. Default 100 [ms]
    tau_rise: float, ArrayType, Callable. The time constant of the synaptic rise phase. Default 2 [ms]
    a: float, ArrayType, Callable. Default 0.5 ms^-1.
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
      a: Union[float, ArrayType, Callable] = 0.5,
      tau_decay: Union[float, ArrayType, Callable] = 100.,
      tau_rise: Union[float, ArrayType, Callable] = 2.,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.tau_decay = self.init_param(tau_decay)
    self.tau_rise = self.init_param(tau_rise)
    self.a = self.init_param(a)

    # integral
    self.integral = odeint(method=method, f=JointEq(self.dg, self.dx))

    self.reset_state(self.mode)

  def dg(self, g, t, x):
    return -g / self.tau_decay + self.a * x * (1 - g)

  def dx(self, x, t):
    return -x / self.tau_rise

  def reset_state(self, batch_size=None):
    self.g = self.init_variable(bm.zeros, batch_size)
    self.x = self.init_variable(bm.zeros, batch_size)

  def update(self, pre_spike):
    t = share.load('t')
    dt = share.load('dt')
    self.g.value, self.x.value = self.integral(self.g, self.x, t, dt=dt)
    self.x += pre_spike
    return self.g.value

  def return_info(self):
    return self.g


NMDA.__doc__ = NMDA.__doc__ % (pneu_doc,)


class STD(SynDyn):
  r"""Synaptic output with short-term depression.

  This model filters the synaptic current by the following equation:

  .. math::

     I_{syn}^+(t) = I_{syn}^-(t) * x

  where :math:`x` is the normalized variable between 0 and 1, and
  :math:`I_{syn}^-(t)` and :math:`I_{syn}^+(t)` are the synaptic currents before
  and after STD filtering.

  Moreover, :math:`x` is updated according to the dynamics of:

  .. math::

     \frac{dx}{dt} = \frac{1-x}{\tau} - U * x * \delta(t-t_{spike})

  where :math:`U` is the fraction of resources used per action potential,
  :math:`\tau` is the time constant of recovery of the synaptic vesicles.

  Args:
    tau: float, ArrayType, Callable. The time constant of recovery of the synaptic vesicles.
    U: float, ArrayType, Callable. The fraction of resources used per action potential.
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
      tau: Union[float, ArrayType, Callable] = 200.,
      U: Union[float, ArrayType, Callable] = 0.07,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.tau = self.init_param(tau)
    self.U = self.init_param(U)

    # integral function
    self.integral = odeint(lambda x, t: (1 - x) / self.tau, method=method)

    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.x = self.init_variable(bm.ones, batch_size)

  def update(self, pre_spike):
    t = share.load('t')
    dt = share.load('dt')
    x = self.integral(self.x.value, t, dt)
    self.x.value = bm.where(pre_spike, x - self.U * self.x, x)
    return self.x.value

  def return_info(self):
    return self.x


STD.__doc__ = STD.__doc__ % (pneu_doc,)


class STP(SynDyn):
  r"""Synaptic output with short-term plasticity.

  This model filters the synaptic currents according to two variables: :math:`u` and :math:`x`.

  .. math::

     I_{syn}^+(t) = I_{syn}^-(t) * x * u

  where :math:`I_{syn}^-(t)` and :math:`I_{syn}^+(t)` are the synaptic currents before
  and after STP filtering, :math:`x` denotes the fraction of resources that remain available
  after neurotransmitter depletion, and :math:`u` represents the fraction of available
  resources ready for use (release probability).

  The dynamics of :math:`u` and :math:`x` are governed by

  .. math::

     \begin{aligned}
    \frac{du}{dt} & = & -\frac{u}{\tau_f}+U(1-u^-)\delta(t-t_{sp}), \\
    \frac{dx}{dt} & = & \frac{1-x}{\tau_d}-u^+x^-\delta(t-t_{sp}), \\
    \tag{1}\end{aligned}

  where :math:`t_{sp}` denotes the spike time and :math:`U` is the increment
  of :math:`u` produced by a spike. :math:`u^-, x^-` are the corresponding
  variables just before the arrival of the spike, and :math:`u^+`
  refers to the moment just after the spike.

  Args:
    tau_f: float, ArrayType, Callable. The time constant of short-term facilitation.
    tau_d: float, ArrayType, Callable. The time constant of short-term depression.
    U: float, ArrayType, Callable. The fraction of resources used per action potential.
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
      U: Union[float, ArrayType, Callable] = 0.15,
      tau_f: Union[float, ArrayType, Callable] = 1500.,
      tau_d: Union[float, ArrayType, Callable] = 200.,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.tau_f = self.init_param(tau_f)
    self.tau_d = self.init_param(tau_d)
    self.U = self.init_param(U)
    self.method = method

    # integral function
    self.integral = odeint(self.derivative, method=self.method)

    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.x = self.init_variable(bm.ones, batch_size)
    self.u = self.init_variable(bm.ones, batch_size)
    self.u.fill_(self.U)

  @property
  def derivative(self):
    du = lambda u, t: self.U - u / self.tau_f
    dx = lambda x, t: (1 - x) / self.tau_d
    return JointEq([du, dx])

  def update(self, pre_spike):
    t = share.load('x')
    dt = share.load('dt')
    u, x = self.integral(self.u.value, self.x.value, t, dt)
    u = bm.where(pre_spike, u + self.U * (1 - self.u), u)
    x = bm.where(pre_spike, x - u * self.x, x)
    self.x.value = x
    self.u.value = u
    return u * x

  def return_info(self):
    return ReturnInfo(size=self.varshape,
                      batch_or_mode=self.mode,
                      axis_names=self.sharding,
                      init=bm.zeros)


STP.__doc__ = STP.__doc__ % (pneu_doc,)


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

  supported_modes = (bm.NonBatchingMode,)

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
  supported_modes = (bm.NonBatchingMode,)

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
