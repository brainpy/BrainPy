# -*- coding: utf-8 -*-

from typing import Union, Dict, Callable

import brainpy.math as bm
from brainpy.connect import TwoEndConnector, All2All, One2One
from brainpy.dyn.base import NeuGroup, TwoEndConn
from brainpy.initialize import Initializer, init_param
from brainpy.integrators import odeint, JointEq
from brainpy.types import Tensor

__all__ = [
  'AMPA',
  'GABAa',
  'BioNMDA',
]


class AMPA(TwoEndConn):
  r"""AMPA conductance-based synapse model.

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
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = bp.dyn.HH(1)
    >>> neu2 = bp.dyn.HH(1)
    >>> syn1 = bp.dyn.AMPA(neu1, neu2, bp.connect.All2All())
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
  conn_type: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `dense`.
  delay_step: int, ndarray, JaxArray, Initializer, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  E: float, JaxArray, ndarray
    The reversal potential for the synaptic current. [mV]
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
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      conn_type: str = 'dense',
      g_max: Union[float, Tensor, Initializer, Callable] = 0.42,
      delay_step: Union[int, Tensor, Initializer, Callable] = None,
      E: Union[float, Tensor] = 0.,
      alpha: Union[float, Tensor] = 0.98,
      beta: Union[float, Tensor] = 0.18,
      T: Union[float, Tensor] = 0.5,
      T_duration: Union[float, Tensor] = 0.5,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(AMPA, self).__init__(pre=pre, post=post, conn=conn, name=name)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # parameters
    self.E = E
    self.alpha = alpha
    self.beta = beta
    self.T = T
    self.T_duration = T_duration
    if bm.size(E) != 1:
      raise ValueError(f'"E" must be a scalar or a tensor with size of 1. But we got {E}')
    if bm.size(alpha) != 1:
      raise ValueError(f'"alpha" must be a scalar or a tensor with size of 1. But we got {alpha}')
    if bm.size(beta) != 1:
      raise ValueError(f'"beta" must be a scalar or a tensor with size of 1. But we got {beta}')
    if bm.size(T) != 1:
      raise ValueError(f'"T" must be a scalar or a tensor with size of 1. But we got {T}')
    if bm.size(T_duration) != 1:
      raise ValueError(f'"T_duration" must be a scalar or a tensor with size of 1. But we got {T_duration}')

    # connection
    self.conn_type = conn_type
    if conn_type not in ['sparse', 'dense']:
      raise ValueError(f'"conn_type" must be in "sparse" and "dense", but we got {conn_type}')
    if self.conn is None:
      raise ValueError(f'Must provide "conn" when initialize the model {self.name}')
    if isinstance(self.conn, One2One):
      self.g_max = init_param(g_max, (self.pre.num,), allow_none=False)
      self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
    elif isinstance(self.conn, All2All):
      self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
      if bm.size(self.g_max) != 1:
        self.weight_type = 'heter'
        bm.fill_diagonal(self.g_max, 0.)
      else:
        self.weight_type = 'homo'
    else:
      if conn_type == 'sparse':
        self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')
        self.g_max = init_param(g_max, self.post_ids.shape, allow_none=False)
        self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
      elif conn_type == 'dense':
        self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
        self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
        if self.weight_type == 'homo':
          self.conn_mat = self.conn.require('conn_mat')
      else:
        raise ValueError(f'Unknown connection type: {conn_type}')

    # variables
    self.g = bm.Variable(bm.zeros(self.pre.num))
    self.spike_arrival_time = bm.Variable(bm.ones(self.pre.num) * -1e7)
    self.delay_step = self.register_delay(f"{self.pre.name}.spike",
                                          delay_step=delay_step,
                                          delay_target=self.pre.spike)

    # functions
    self.integral = odeint(method=method, f=self.dg)

  def reset(self):
    self.g.value = bm.zeros(self.pre.num)

  def dg(self, g, t, TT):
    dg = self.alpha * TT * (1 - g) - self.beta * g
    return dg

  def update(self, t, dt):
    # delays
    pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)

    # spike arrival time
    self.spike_arrival_time.value = bm.where(pre_spike, t, self.spike_arrival_time)

    # post-synaptic values
    TT = ((t - self.spike_arrival_time) < self.T_duration) * self.T
    self.g.value = self.integral(self.g, t, TT, dt=dt)
    if isinstance(self.conn, One2One):
      post_g = self.g_max * self.g
    elif isinstance(self.conn, All2All):
      if self.weight_type == 'homo':
        post_g = bm.sum(self.g)
        if not self.conn.include_self:
          post_g = post_g - self.g
        post_g = post_g * self.g_max
      else:
        post_g = self.g @ self.g_max
    else:
      if self.conn_type == 'sparse':
        post_g = bm.pre2post_sum(self.g, self.post.num, self.post_ids, self.pre_ids)
      else:
        if self.weight_type == 'homo':
          post_g = (self.g_max * self.g) @ self.conn_mat
        else:
          post_g = self.g @ self.g_max

    # output
    self.post.input -= post_g * (self.post.V - self.E)


class GABAa(AMPA):
  r"""GABAa conductance-based synapse model.

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
  conn_type: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `dense`.
  delay_step: int, ndarray, JaxArray, Initializer, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  E: float, JaxArray, ndarray
    The reversal potential for the synaptic current. [mV]
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
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      conn_type: str = 'dense',
      g_max: Union[float, Tensor, Initializer, Callable] = 0.04,
      delay_step: Union[int, Tensor, Initializer, Callable] = None,
      E: Union[float, Tensor] = -80.,
      alpha: Union[float, Tensor] = 0.53,
      beta: Union[float, Tensor] = 0.18,
      T: Union[float, Tensor] = 1.,
      T_duration: Union[float, Tensor] = 1.,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(GABAa, self).__init__(pre, post, conn,
                                conn_type=conn_type,
                                delay_step=delay_step,
                                g_max=g_max,
                                E=E,
                                alpha=alpha,
                                beta=beta,
                                T=T,
                                T_duration=T_duration,
                                method=method,
                                name=name)


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
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = bp.dyn.HH(1)
    >>> neu2 = bp.dyn.HH(1)
    >>> syn1 = bp.dyn.BioNMDA(neu1, neu2, bp.connect.All2All(), E=0.)
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
  conn_type: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `dense`.
  delay_step: int, ndarray, JaxArray, Initializer, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  g_max: float, ndarray, JaxArray, Initializer, Callable
    The synaptic strength (the maximum conductance). Default is 1.
  E: float, JaxArray, ndarray
    The reversal potential for the synaptic current. [mV]
  a: float, JaxArray, ndarray
    Binding constant. Default 0.062
  b: float, JaxArray, ndarray
    Unbinding constant. Default 3.57
  cc_Mg: float, JaxArray, ndarray
    Concentration of Magnesium ion. Default 1.2 [mM].
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
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      conn_type: str = 'dense',
      g_max: Union[float, Tensor, Initializer, Callable] = 0.15,
      delay_step: Union[int, Tensor, Initializer, Callable] = None,
      E: Union[float, Tensor] = 0.,
      cc_Mg: Union[float, Tensor] = 1.2,
      a: Union[float, Tensor] = 0.062,
      b: Union[float, Tensor] = 3.57,
      alpha1: Union[float, Tensor] = 2.,
      beta1: Union[float, Tensor] = 0.01,
      alpha2: Union[float, Tensor] = 1.,
      beta2: Union[float, Tensor] = 0.5,
      T_0: Union[float, Tensor] = 1.,
      T_dur: Union[float, Tensor] = 0.5,
      method: str = 'exp_auto',
      name: str = None,
  ):
    super(BioNMDA, self).__init__(pre=pre, post=post, conn=conn, name=name)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # parameters
    self.E = E
    self.alpha = a
    self.beta = b
    self.cc_Mg = cc_Mg
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
    if bm.size(E) != 1:
      raise ValueError(f'"E" must be a scalar or a tensor with size of 1. But we got {E}')
    if bm.size(a) != 1:
      raise ValueError(f'"a" must be a scalar or a tensor with size of 1. But we got {a}')
    if bm.size(b) != 1:
      raise ValueError(f'"b" must be a scalar or a tensor with size of 1. But we got {b}')
    if bm.size(cc_Mg) != 1:
      raise ValueError(f'"cc_Mg" must be a scalar or a tensor with size of 1. But we got {cc_Mg}')
    if bm.size(T_0) != 1:
      raise ValueError(f'"T_0" must be a scalar or a tensor with size of 1. But we got {T_0}')
    if bm.size(T_dur) != 1:
      raise ValueError(f'"T_dur" must be a scalar or a tensor with size of 1. But we got {T_dur}')

    # connections and weights
    self.conn_type = conn_type
    if conn_type not in ['sparse', 'dense']:
      raise ValueError(f'"conn_type" must be in "sparse" and "dense", but we got {conn_type}')
    if self.conn is None:
      raise ValueError(f'Must provide "conn" when initialize the model {self.name}')
    if isinstance(self.conn, One2One):
      self.g_max = init_param(g_max, (self.pre.num,), allow_none=False)
      self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
    elif isinstance(self.conn, All2All):
      self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
      if bm.size(self.g_max) != 1:
        self.weight_type = 'heter'
        bm.fill_diagonal(self.g_max, 0.)
      else:
        self.weight_type = 'homo'
    else:
      if conn_type == 'sparse':
        self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')
        self.g_max = init_param(g_max, self.post_ids.shape, allow_none=False)
        self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
      elif conn_type == 'dense':
        self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
        self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
        if self.weight_type == 'homo':
          self.conn_mat = self.conn.require('conn_mat')
      else:
        raise ValueError(f'Unknown connection type: {conn_type}')

    # variables
    self.g = bm.Variable(bm.zeros(self.pre.num, dtype=bm.float_))
    self.x = bm.Variable(bm.zeros(self.pre.num, dtype=bm.float_))
    self.spike_arrival_time = bm.Variable(bm.ones(self.pre.num, dtype=bm.float_) * -1e7)
    self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)

    # integral
    self.integral = odeint(method=method, f=JointEq([self.dg, self.dx]))

  def reset(self):
    self.g.value = bm.zeros(self.pre.num)
    self.x.value = bm.zeros(self.pre.num)
    self.spike_arrival_time.value = bm.ones(self.pre.num) * -1e7

  def dg(self, g, t, x):
    return self.alpha1 * x * (1 - g) - self.beta1 * g

  def dx(self, x, t, T):
    return self.alpha2 * T * (1 - x) - self.beta2 * x

  def update(self, t, dt):
    # delays
    delayed_pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)

    # update synapse variables
    self.spike_arrival_time.value = bm.where(delayed_pre_spike, t, self.spike_arrival_time)
    T = ((t - self.spike_arrival_time) < self.T_dur) * self.T_0
    self.g.value, self.x.value = self.integral(self.g, self.x, t, T, dt=dt)

    # post-synaptic value
    assert self.weight_type in ['homo', 'heter']
    assert self.conn_type in ['sparse', 'dense']
    if isinstance(self.conn, All2All):
      if self.weight_type == 'homo':
        post_g = bm.sum(self.g)
        if not self.conn.include_self:
          post_g = post_g - self.g
        post_g = post_g * self.g_max
      else:
        post_g = self.g @ self.g_max
    elif isinstance(self.conn, One2One):
      post_g = self.g_max * self.g
    else:
      if self.conn_type == 'sparse':
        post_g = bm.pre2post_sum(self.g, self.post.num, self.post_ids, self.pre_ids)
      else:
        if self.weight_type == 'homo':
          post_g = (self.g_max * self.g) @ self.conn_mat
        else:
          post_g = self.g @ self.g_max

    # output
    g_inf = 1 + self.cc_Mg / self.beta * bm.exp(-self.alpha * self.post.V)
    self.post.input += post_g * (self.E - self.post.V) / g_inf
