from typing import Union, Sequence, Callable, Optional

import jax.numpy
from brainpy import math as bm
from brainpy._src.context import share
from brainpy._src.dyn._docs import pneu_doc
from brainpy._src.dyn.base import SynDyn
from brainpy._src.integrators.joint_eq import JointEq
from brainpy._src.integrators.ode.generic import odeint
from brainpy._src.mixin import AlignPost, ReturnInfo
from brainpy._src.initialize import Constant
from brainpy.types import ArrayType

__all__ = [
  'Delta',
  'Expon',
  'DualExpon',
  'DualExponV2',
  'Alpha',
  'NMDA',
  'STD',
  'STP',
]


class Delta(SynDyn, AlignPost):
  r"""Delta synapse model.

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
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      sharding: Optional[Sequence[str]] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.g = self.init_variable(bm.zeros, batch_size)

  def update(self, x=None):
    if x is not None:
      self.g.value += x
    return self.g.value

  def add_current(self, x):
    self.g.value += x

  def return_info(self):
    return self.g


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
    self.g.value = self.integral(self.g.value, share['t'], share['dt'])
    if x is not None:
      self.add_current(x)
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
    # update synaptic variables
    self.g.value, self.h.value = self.integral(self.g.value, self.h.value, share['t'], dt=share['dt'])
    self.h += x
    return self.g.value

  def return_info(self):
    return self.g


DualExpon.__doc__ = DualExpon.__doc__ % (pneu_doc,)


class DualExponV2(SynDyn, AlignPost):
  r"""Dual exponential synapse model.

  The dual exponential synapse model [1]_, also named as *difference of two exponentials* model,
  is given by:

  .. math::

    g_{\mathrm{syn}}(t)=g_{\mathrm{max}} \frac{\tau_{1} \tau_{2}}{
        \tau_{1}-\tau_{2}}\left(\exp \left(-\frac{t-t_{0}}{\tau_{1}}\right)
        -\exp \left(-\frac{t-t_{0}}{\tau_{2}}\right)\right)

  where :math:`\tau_1` is the time constant of the decay phase, :math:`\tau_2`
  is the time constant of the rise phase, :math:`t_0` is the time of the pre-synaptic
  spike, :math:`g_{\mathrm{max}}` is the maximal conductance.

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
    self.coeff = self.tau_rise * self.tau_decay / (self.tau_decay - self.tau_rise)

    # integrator
    self.integral = odeint(lambda g, t, tau: -g / tau, method=method)

    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.g_rise = self.init_variable(bm.zeros, batch_size)
    self.g_decay = self.init_variable(bm.zeros, batch_size)

  def update(self, x=None):
    self.g_rise.value = self.integral(self.g_rise.value, share['t'], self.tau_rise, share['dt'])
    self.g_decay.value = self.integral(self.g_decay.value, share['t'], self.tau_decay, share['dt'])
    if x is not None:
      self.add_current(x)
    return self.coeff * (self.g_decay - self.g_rise)

  def add_current(self, inp):
    self.g_rise += inp
    self.g_decay += inp

  def return_info(self):
    return ReturnInfo(self.varshape, self.sharding, self.mode,
                      lambda shape: self.coeff * (self.g_decay - self.g_rise))


DualExponV2.__doc__ = DualExponV2.__doc__ % (pneu_doc,)


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
    return JointEq(du, dx)

  def update(self, pre_spike):
    t = share.load('t')
    dt = share.load('dt')
    u, x = self.integral(self.u.value, self.x.value, t, dt)
    # if pre_spike.dtype == jax.numpy.bool_:
    #   u = bm.where(pre_spike, u + self.U * (1 - self.u), u)
    #   x = bm.where(pre_spike, x - u * self.x, x)
    # else:
    # u = pre_spike * (u + self.U * (1 - self.u)) + (1 - pre_spike) * u
    # x = pre_spike * (x - u * self.x) + (1 - pre_spike) * x
    u = pre_spike * self.U * (1 - self.u) + u
    x = pre_spike * -u * self.x + x
    self.x.value = x
    self.u.value = u
    return u * x

  def return_info(self):
    return ReturnInfo(self.varshape, self.sharding, self.mode,
                      lambda shape: self.u * self.x)


STP.__doc__ = STP.__doc__ % (pneu_doc,)
