# -*- coding: utf-8 -*-

from typing import Union, Dict, Callable

import brainpy.math as bm
from brainpy.connect import TwoEndConnector, All2All, One2One
from brainpy.dyn.base import NeuGroup, TwoEndConn
from brainpy.initialize import Initializer, init_param
from brainpy.integrators import odeint
from brainpy.types import Tensor

__all__ = [
  'AMPA',
  'GABAa',
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

  **Model Parameters**

  ============= ============== ======== ================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------------------------------
  delay         0              ms       The decay length of the pre-synaptic spikes.
  g_max         .42            µmho(µS) Maximum conductance.
  E             0              mV       The reversal potential for the synaptic current.
  alpha         .98            \        Binding constant.
  beta          .18            \        Unbinding constant.
  T             .5             mM       Neurotransmitter concentration.
  T_duration    .5             ms       Duration of the neurotransmitter concentration.
  ============= ============== ======== ================================================


  **Model Variables**

  ================== ================== ==================================================
  **Member name**    **Initial values** **Explanation**
  ------------------ ------------------ --------------------------------------------------
  g                  0                  Synapse gating variable.
  pre_spike          False              The history of pre-synaptic neuron spikes.
  spike_arrival_time -1e7               The arrival time of the pre-synaptic neuron spike.
  ================== ================== ==================================================

  **References**

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
      E: float = 0.,
      alpha: float = 0.98,
      beta: float = 0.18,
      T: float = 0.5,
      T_duration: float = 0.5,
      delay_step: Union[int, Tensor, Initializer, Callable] = None,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(AMPA, self).__init__(pre=pre, post=post, conn=conn, name=name)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # parameters
    self.g_max = g_max
    self.E = E
    self.alpha = alpha
    self.beta = beta
    self.T = T
    self.T_duration = T_duration

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
    self.delay_step = self.register_delay(self.pre.name + '.spike', delay_step, self.pre.spike)

    # functions
    self.integral = odeint(method=method, f=self.dg)

  def dg(self, g, t, TT):
    dg = self.alpha * TT * (1 - g) - self.beta * g
    return dg

  def update(self, _t, _dt):
    # delays
    if self.delay_step is None:
      pre_spike = self.pre.spike
    else:
      pre_spike = self.get_delay(self.pre.name + '.spike', self.delay_step)
      self.update_delay(self.pre.name + '.spike', self.pre.spike)

    # spike arrival time
    self.spike_arrival_time.value = bm.where(pre_spike, _t, self.spike_arrival_time)

    # post-synaptic values
    TT = ((_t - self.spike_arrival_time) < self.T_duration) * self.T
    self.g.value = self.integral(self.g, _t, TT, dt=_dt)
    if isinstance(self.conn, One2One):
      post_g = self.g_max * self.g
    elif isinstance(self.conn, All2All):
      if self.weight_type == 'homo':
        post_g = self.g.sum() * self.g_max
        if not self.conn.include_self:
          post_g = post_g - self.g
      else:
        post_g = bm.expand_dims(self.g, 1) * self.g_max
        post_g = post_g.sum(axis=0)
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
  E: float
    The reversal potential for the synaptic current. [mV]
  g_max: float, ndarray, JaxArray, Initializer, Callable
    The synaptic strength (the maximum conductance). Default is 1.
  alpha: float
    Binding constant. Default 0.062
  beta: float
    Unbinding constant. Default 3.57
  T: float
    Transmitter concentration when synapse is triggered by
    a pre-synaptic spike.. Default 1 [mM].
  T_duration: float
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
      E: float = -80.,
      alpha: float = 0.53,
      beta: float = 0.18,
      T: float = 1.,
      T_duration: float = 1.,
      delay_step: Union[int, Tensor, Initializer, Callable] = None,
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
