# -*- coding: utf-8 -*-

import warnings
from typing import Union, Dict, Callable, Optional

from jax import vmap
from jax.lax import stop_gradient

import brainpy.math as bm
from brainpy.connect import TwoEndConnector, All2All, One2One
from brainpy.dyn.base import NeuGroup, TwoEndConn, SynSTP, SynOut
from brainpy.dyn.synouts import COBA, MgBlock
from brainpy.initialize import Initializer, variable
from brainpy.integrators import odeint, JointEq
from brainpy.types import Array
from brainpy.modes import Mode, BatchingMode, TrainingMode, normal, batching, training

__all__ = [
  'AMPA',
  'GABAa',
  'BioNMDA',
]


class AMPA(TwoEndConn):
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

  .. image:: ../../../../_static/synapse_markov.png
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

  **Model Examples**


  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> from brainpy.dyn import neurons, synapses
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = neurons.HH(1)
    >>> neu2 = neurons.HH(1)
    >>> syn1 = synapses.AMPA(neu1, neu2, bp.connect.All2All())
    >>> net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.dyn.DSRunner(net, inputs=[('pre.input', 5.)], monitors=['pre.V', 'post.V', 'syn.g'])
    >>> runner.run(150.)
    >>>
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
    >>> fig.add_subplot(gs[0, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
    >>> plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
    >>> plt.legend()
    >>>
    >>> fig.add_subplot(gs[1, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g')
    >>> plt.legend()
    >>> plt.show()

  Parameters
  ----------
  pre: NeuGroup
    The pre-synaptic neuron group.
  post: NeuGroup
    The post-synaptic neuron group.
  conn: optional, ndarray, JaxArray, dict of (str, ndarray), TwoEndConnector
    The synaptic connections.
  comp_method: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `dense`.
  delay_step: int, ndarray, JaxArray, Initializer, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  E: float, JaxArray, ndarray
    The reversal potential for the synaptic current. [mV]

    .. deprecated:: 2.1.13
       `E` is deprecated in AMPA model. Please define `E` with brainpy.dyn.synouts.COBA.
       This parameter will be removed since 2.2.0

  g_max: float, ndarray, JaxArray, Initializer, Callable
    The synaptic strength (the maximum conductance). Default is 1.
  alpha: float, JaxArray, ndarray
    Binding constant.
  beta: float, JaxArray, ndarray
    Unbinding constant.
  T: float, JaxArray, ndarray
    Transmitter concentration when synapse is triggered by
    a pre-synaptic spike.. Default 1 [mM].
  T_duration: float, JaxArray, ndarray
    Transmitter concentration duration time after being triggered. Default 1 [ms]
  name: str
    The name of this synaptic projection.
  method: str
    The numerical integration methods.

  References
  ----------

  .. [1] Vijayan S, Kopell N J. Thalamic model of awake alpha oscillations
         and implications for stimulus processing[J]. Proceedings of the
         National Academy of Sciences, 2012, 109(45): 18553-18558.
  """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Array, Dict[str, Array]],
      output: SynOut = COBA(E=0.),
      stp: Optional[SynSTP] = None,
      comp_method: str = 'dense',
      g_max: Union[float, Array, Initializer, Callable] = 0.42,
      delay_step: Union[int, Array, Initializer, Callable] = None,
      alpha: float = 0.98,
      beta: float = 0.18,
      T: float = 0.5,
      T_duration: float = 0.5,
      method: str = 'exp_auto',

      # other parameters
      name: str = None,
      mode: Mode = normal,
      stop_spike_gradient: bool = False,
  ):
    super(AMPA, self).__init__(pre=pre,
                               post=post,
                               conn=conn,
                               output=output,
                               stp=stp,
                               name=name,
                               mode=mode)

    # parameters
    self.stop_spike_gradient = stop_spike_gradient
    self.comp_method = comp_method
    self.alpha = alpha
    self.beta = beta
    self.T = T
    self.T_duration = T_duration
    if bm.size(alpha) != 1:
      raise ValueError(f'"alpha" must be a scalar or a tensor with size of 1. But we got {alpha}')
    if bm.size(beta) != 1:
      raise ValueError(f'"beta" must be a scalar or a tensor with size of 1. But we got {beta}')
    if bm.size(T) != 1:
      raise ValueError(f'"T" must be a scalar or a tensor with size of 1. But we got {T}')
    if bm.size(T_duration) != 1:
      raise ValueError(f'"T_duration" must be a scalar or a tensor with size of 1. But we got {T_duration}')

    # connection
    self.g_max, self.conn_mask = self.init_weights(g_max, comp_method, sparse_data='ij')

    # variables
    self.g = variable(bm.zeros, mode, self.pre.num)
    self.spike_arrival_time = variable(lambda s: bm.ones(s) * -1e7, mode, self.pre.num)
    self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)

    # functions
    self.integral = odeint(method=method, f=self.dg)

  def reset_state(self, batch_size=None):
    self.g = variable(bm.zeros, batch_size, self.pre.num)
    self.spike_arrival_time = variable(lambda s: bm.ones(s) * -1e7, batch_size, self.pre.num)
    self.output.reset_state(batch_size)
    if self.stp is not None: self.stp.reset_state(batch_size)

  def dg(self, g, t, TT):
    dg = self.alpha * TT * (1 - g) - self.beta * g
    return dg

  def update(self, tdi, pre_spike=None):
    t, dt = tdi['t'], tdi['dt']

    # delays
    if pre_spike is None:
      pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)
    if self.stop_spike_gradient:
      pre_spike = pre_spike.value if isinstance(pre_spike, bm.JaxArray) else pre_spike
      pre_spike = stop_gradient(pre_spike)

    # update sub-components
    self.output.update(tdi)
    if self.stp is not None: self.stp.update(tdi, pre_spike)

    # update synaptic variables
    self.spike_arrival_time.value = bm.where(pre_spike, t, self.spike_arrival_time)
    if isinstance(self.mode, TrainingMode):
      self.spike_arrival_time.value = stop_gradient(self.spike_arrival_time.value)
    TT = ((t - self.spike_arrival_time) < self.T_duration) * self.T
    self.g.value = self.integral(self.g, t, TT, dt)

    # post-synaptic values
    syn_value = self.g.value
    if self.stp is not None: syn_value = self.stp(syn_value)
    if isinstance(self.conn, All2All):
      post_vs = self.syn2post_with_all2all(syn_value, self.g_max)
    elif isinstance(self.conn, One2One):
      post_vs = self.syn2post_with_one2one(syn_value, self.g_max)
    else:
      if self.comp_method == 'sparse':
        f = lambda s: bm.pre2post_sum(s, self.post.num, *self.conn_mask)
        if isinstance(self.mode, BatchingMode): f = vmap(f)
        post_vs = f(syn_value)
      else:
        post_vs = self.syn2post_with_dense(syn_value, self.g_max, self.conn_mask)

    # output
    return self.output(post_vs)


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

  **Model Examples**

  - `Gamma oscillation network model <https://brainpy-examples.readthedocs.io/en/latest/oscillation_synchronization/Wang_1996_gamma_oscillation.html>`_


  Parameters
  ----------
  pre: NeuGroup
    The pre-synaptic neuron group.
  post: NeuGroup
    The post-synaptic neuron group.
  conn: optional, ndarray, JaxArray, dict of (str, ndarray), TwoEndConnector
    The synaptic connections.
  comp_method: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `dense`.
  delay_step: int, ndarray, JaxArray, Initializer, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  E: float, JaxArray, ndarray
    The reversal potential for the synaptic current. [mV]

    .. deprecated:: 2.1.13
       `E` is deprecated in AMPA model. Please define `E` with brainpy.dyn.synouts.COBA.
       This parameter will be removed since 2.2.0

  g_max: float, ndarray, JaxArray, Initializer, Callable
    The synaptic strength (the maximum conductance). Default is 1.
  alpha: float, JaxArray, ndarray
    Binding constant. Default 0.062
  beta: float, JaxArray, ndarray
    Unbinding constant. Default 3.57
  T: float, JaxArray, ndarray
    Transmitter concentration when synapse is triggered by
    a pre-synaptic spike.. Default 1 [mM].
  T_duration: float, JaxArray, ndarray
    Transmitter concentration duration time after being triggered. Default 1 [ms]
  name: str
    The name of this synaptic projection.
  method: str
    The numerical integration methods.

  References
  ----------
  .. [1] Destexhe, Alain, and Denis Paré. "Impact of network activity
         on the integrative properties of neocortical pyramidal neurons
         in vivo." Journal of neurophysiology 81.4 (1999): 1531-1547.
  """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Array, Dict[str, Array]],
      output: SynOut = COBA(E=-80.),
      stp: Optional[SynSTP] = None,
      comp_method: str = 'dense',
      g_max: Union[float, Array, Initializer, Callable] = 0.04,
      delay_step: Union[int, Array, Initializer, Callable] = None,
      alpha: Union[float, Array] = 0.53,
      beta: Union[float, Array] = 0.18,
      T: Union[float, Array] = 1.,
      T_duration: Union[float, Array] = 1.,
      method: str = 'exp_auto',

      # other parameters
      name: str = None,
      mode: Mode = normal,
      stop_spike_gradient: bool = False,

      # deprecated
      E: Union[float, Array] = None,
  ):
    super(GABAa, self).__init__(pre=pre,
                                post=post,
                                conn=conn,
                                output=output,
                                stp=stp,
                                comp_method=comp_method,
                                delay_step=delay_step,
                                g_max=g_max,
                                alpha=alpha,
                                beta=beta,
                                T=T,
                                T_duration=T_duration,
                                method=method,
                                name=name,
                                mode=mode,
                                stop_spike_gradient=stop_spike_gradient, )


class BioNMDA(TwoEndConn):
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

      g_{\infty}(V,[{Mg}^{2+}]_{o}) = (1+{e}^{-\a V}
      \frac{[{Mg}^{2+}]_{o}} {\b})^{-1}

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

      & g_\mathrm{NMDA} (t) = g_{max} g \\
      & \frac{d g}{dt} = \alpha_1 x (1 - g) - \beta_1 g \\
      & \frac{d x}{dt} = \alpha_2 [T] (1 - x) - \beta_2 x

  where :math:`\alpha_1, \beta_1` refers to the conversion rate of variable g and
  :math:`\alpha_2, \beta_2` refers to the conversion rate of variable x.

  The NMDA receptor has been thought to be very important for controlling
  synaptic plasticity and mediating learning and memory functions [3]_.

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> from brainpy.dyn import neurons, synapses
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = neurons.HH(1)
    >>> neu2 = neurons.HH(1)
    >>> syn1 = synapses.BioNMDA(neu1, neu2, bp.connect.All2All(), E=0.)
    >>> net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.dyn.DSRunner(net, inputs=[('pre.input', 5.)], monitors=['pre.V', 'post.V', 'syn.g', 'syn.x'])
    >>> runner.run(150.)
    >>>
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
    >>> fig.add_subplot(gs[0, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
    >>> plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
    >>> plt.legend()
    >>>
    >>> fig.add_subplot(gs[1, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g')
    >>> plt.plot(runner.mon.ts, runner.mon['syn.x'], label='x')
    >>> plt.legend()
    >>> plt.show()

  Parameters
  ----------
  pre: NeuGroup
    The pre-synaptic neuron group.
  post: NeuGroup
    The post-synaptic neuron group.
  conn: optional, ndarray, JaxArray, dict of (str, ndarray), TwoEndConnector
    The synaptic connections.
  comp_method: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `dense`.
  delay_step: int, ndarray, JaxArray, Initializer, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  g_max: float, ndarray, JaxArray, Initializer, Callable
    The synaptic strength (the maximum conductance). Default is 1.
  alpha1: float, JaxArray, ndarray
    The conversion rate of g from inactive to active. Default 2 ms^-1.
  beta1: float, JaxArray, ndarray
    The conversion rate of g from active to inactive. Default 0.01 ms^-1.
  alpha2: float, JaxArray, ndarray
    The conversion rate of x from inactive to active. Default 1 ms^-1.
  beta2: float, JaxArray, ndarray
    The conversion rate of x from active to inactive. Default 0.5 ms^-1.
  name: str
    The name of this synaptic projection.
  method: str
    The numerical integration methods.

  References
  ----------

  .. [1] Devaney A J . Mathematical Foundations of Neuroscience[M].
         Springer New York, 2010: 162.
  .. [2] Furukawa, Hiroyasu, Satinder K. Singh, Romina Mancusso, and
         Eric Gouaux. "Subunit arrangement and function in NMDA receptors."
         Nature 438, no. 7065 (2005): 185-192.
  .. [3] Li, F. and Tsien, J.Z., 2009. Memory and the NMDA receptors. The New
         England journal of medicine, 361(3), p.302.
  .. [4] https://en.wikipedia.org/wiki/NMDA_receptor

  """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Array, Dict[str, Array]],
      output: SynOut = MgBlock(E=0.),
      stp: Optional[SynSTP] = None,
      comp_method: str = 'dense',
      g_max: Union[float, Array, Initializer, Callable] = 0.15,
      delay_step: Union[int, Array, Initializer, Callable] = None,
      alpha1: Union[float, Array] = 2.,
      beta1: Union[float, Array] = 0.01,
      alpha2: Union[float, Array] = 1.,
      beta2: Union[float, Array] = 0.5,
      T_0: Union[float, Array] = 1.,
      T_dur: Union[float, Array] = 0.5,
      method: str = 'exp_auto',

      # other parameters
      mode: Mode = normal,
      name: str = None,
      stop_spike_gradient: bool = False,
  ):
    super(BioNMDA, self).__init__(pre=pre,
                                  post=post,
                                  conn=conn,
                                  output=output,
                                  stp=stp,
                                  name=name,
                                  mode=mode)

    # parameters
    self.beta1 = beta1
    self.beta2 = beta2
    self.alpha1 = alpha1
    self.alpha2 = alpha2
    self.T_0 = T_0
    self.T_dur = T_dur
    if bm.size(alpha1) != 1:
      raise ValueError(f'"alpha1" must be a scalar or a tensor with size of 1. But we got {alpha1}')
    if bm.size(beta1) != 1:
      raise ValueError(f'"beta1" must be a scalar or a tensor with size of 1. But we got {beta1}')
    if bm.size(alpha2) != 1:
      raise ValueError(f'"alpha2" must be a scalar or a tensor with size of 1. But we got {alpha2}')
    if bm.size(beta2) != 1:
      raise ValueError(f'"beta2" must be a scalar or a tensor with size of 1. But we got {beta2}')
    if bm.size(T_0) != 1:
      raise ValueError(f'"T_0" must be a scalar or a tensor with size of 1. But we got {T_0}')
    if bm.size(T_dur) != 1:
      raise ValueError(f'"T_dur" must be a scalar or a tensor with size of 1. But we got {T_dur}')
    self.comp_method = comp_method
    self.stop_spike_gradient = stop_spike_gradient

    # connections and weights
    self.g_max, self.conn_mask = self.init_weights(g_max, comp_method, sparse_data='ij')

    # variables
    self.g = variable(bm.zeros, mode, self.pre.num)
    self.x = variable(bm.zeros, mode, self.pre.num)
    self.spike_arrival_time = variable(lambda s: bm.ones(s) * -1e7, mode, self.pre.num)
    self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)

    # integral
    self.integral = odeint(method=method, f=JointEq([self.dg, self.dx]))

  def reset_state(self, batch_size=None):
    self.g = variable(bm.zeros, batch_size, self.pre.num)
    self.x = variable(bm.zeros, batch_size, self.pre.num)
    self.spike_arrival_time = variable(lambda s: bm.ones(s) * -1e7, batch_size, self.pre.num)
    self.output.reset_state(batch_size)
    if self.stp is not None: self.stp.reset_state(batch_size)

  def dg(self, g, t, x):
    return self.alpha1 * x * (1 - g) - self.beta1 * g

  def dx(self, x, t, T):
    return self.alpha2 * T * (1 - x) - self.beta2 * x

  def update(self, tdi, pre_spike=None):
    t, dt = tdi['t'], tdi['dt']

    # pre-synaptic spikes
    if pre_spike is None:
      pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)
    if self.stop_spike_gradient:
      pre_spike = pre_spike.value if isinstance(pre_spike, bm.JaxArray) else pre_spike
      pre_spike = stop_gradient(pre_spike)

    # update sub-components
    self.output.update(tdi)
    if self.stp is not None: self.stp.update(tdi, pre_spike)

    # update synapse variables
    self.spike_arrival_time.value = bm.where(pre_spike, t, self.spike_arrival_time)
    if isinstance(self.mode, TrainingMode):
      self.spike_arrival_time.value = stop_gradient(self.spike_arrival_time.value)
    T = ((t - self.spike_arrival_time) < self.T_dur) * self.T_0
    self.g.value, self.x.value = self.integral(self.g, self.x, t, T, dt)

    # post-synaptic value
    syn_value = self.g.value
    if self.stp is not None: syn_value = self.stp(syn_value)
    if isinstance(self.conn, All2All):
      post_vs = self.syn2post_with_all2all(syn_value, self.g_max)
    elif isinstance(self.conn, One2One):
      post_vs = self.syn2post_with_one2one(syn_value, self.g_max)
    else:
      if self.comp_method == 'sparse':
        f = lambda s: bm.pre2post_sum(s, self.post.num, *self.conn_mask)
        if isinstance(self.mode, BatchingMode): f = vmap(f)
        post_vs = f(syn_value)
      else:
        post_vs = self.syn2post_with_dense(syn_value, self.g_max, self.conn_mask)

    # output
    return self.output(post_vs)
