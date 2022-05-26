# -*- coding: utf-8 -*-
import warnings
from typing import Union, Dict, Callable, Optional

import brainpy.math as bm
from brainpy.connect import TwoEndConnector, All2All, One2One
from brainpy.dyn.base import NeuGroup, SynapseOutput, SynapsePlasticity, TwoEndConn
from brainpy.initialize import Initializer, init_param
from brainpy.integrators import odeint, JointEq
from brainpy.types import Tensor
from ..synouts import CUBA, MgBlock

__all__ = [
  'Delta',
  'Exponential',
  'DualExponential',
  'Alpha',
  'NMDA',
]


class Delta(TwoEndConn):
  r"""Voltage Jump Synapse Model, or alias of Delta Synapse Model.

  **Model Descriptions**

  .. math::

      I_{syn} (t) = \sum_{j\in C} w \delta(t-t_j-D)

  where :math:`w` denotes the chemical synaptic strength, :math:`t_j` the spiking
  moment of the presynaptic neuron :math:`j`, :math:`C` the set of neurons connected
  to the post-synaptic neuron, and :math:`D` the transmission delay of chemical
  synapses. For simplicity, the rise and decay phases of post-synaptic currents are
  omitted in this model.

  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> from brainpy.dyn import synapses, neurons
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = neurons.LIF(1)
    >>> neu2 = neurons.LIF(1)
    >>> syn1 = synapses.Alpha(neu1, neu2, bp.connect.All2All(), weights=5.)
    >>> net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.dyn.DSRunner(net, inputs=[('pre.input', 25.), ('post.input', 10.)], monitors=['pre.V', 'post.V', 'pre.spike'])
    >>> runner.run(150.)
    >>>
    >>> fig, gs = bp.visualize.get_figure(1, 1, 3, 8)
    >>> plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
    >>> plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
    >>> plt.xlim(40, 150)
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
    `sparse` and `dense`. The default is `sparse`.
  delay_step: int, ndarray, JaxArray, Initializer, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  g_max: float, ndarray, JaxArray, Initializer, Callable
    The synaptic strength. Default is 1.
  post_key: str
    The key of the post variable. It should be a string. The key should
    be the attribute of the post-synaptic neuron group.
  post_has_ref: bool
    Whether the post-synaptic group has refractory period.
  """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      output: Optional[SynapseOutput] = None,
      plasticity: Optional[SynapsePlasticity] = None,
      conn_type: str = 'sparse',
      g_max: Union[float, Tensor, Initializer, Callable] = 1.,
      delay_step: Union[float, Tensor, Initializer, Callable] = None,
      post_key: str = 'V',
      post_has_ref: bool = False,
      name: str = None,
  ):
    super(Delta, self).__init__(pre=pre,
                                post=post,
                                conn=conn,
                                output=CUBA() if output is None else output,
                                plasticity=plasticity,
                                name=name)
    self.check_pre_attrs('spike')

    # parameters
    self.post_key = post_key
    self.check_post_attrs(post_key)
    self.post_has_ref = post_has_ref
    if post_has_ref:
      self.check_post_attrs('refractory')

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
        self.pre2post = self.conn.require('pre2post')
        self.g_max = init_param(g_max, self.pre2post[1].shape, allow_none=False)
        self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
      elif conn_type == 'dense':
        self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
        self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
        if self.weight_type == 'homo':
          self.conn_mat = self.conn.require('conn_mat')
      else:
        raise ValueError(f'Unknown connection type: {conn_type}')

    # variables
    self.delay_step = self.register_delay(f"{self.pre.name}.spike",
                                          delay_step=delay_step,
                                          delay_target=self.pre.spike)

  def reset(self):
    self.output.reset()
    self.plasticity.reset()

  def update(self, t, dt):
    # delayed pre-synaptic spikes
    pre_spike = self.get_delay_data(f"{self.pre.name}.spike", delay_step=self.delay_step)

    # update sub-components
    self.output.update(t, dt)
    self.plasticity.update(t, dt, pre_spike, self.post.spike)

    # post values
    pre_spike = self.plasticity.filter(pre_spike.astype(bm.float_))

    assert self.weight_type in ['homo', 'heter']
    assert self.conn_type in ['sparse', 'dense']
    if isinstance(self.conn, All2All):
      if self.weight_type == 'homo':
        post_vs = bm.sum(pre_spike)
        if not self.conn.include_self:
          post_vs = post_vs - pre_spike
        post_vs *= self.g_max
      else:
        post_vs = pre_spike @ self.g_max
    elif isinstance(self.conn, One2One):
      post_vs = pre_spike * self.g_max
    else:
      if self.conn_type == 'sparse':
        post_vs = bm.pre2post_event_sum(pre_spike,
                                        self.pre2post,
                                        self.post.num,
                                        self.g_max)
      else:
        if self.weight_type == 'homo':
          post_vs = self.g_max * (pre_spike @ self.conn_mat)
        else:
          post_vs = pre_spike @ self.g_max

    # update outputs
    target = getattr(self.post, self.post_key)
    if self.post_has_ref:
      post_vs = post_vs * bm.logical_not(self.post.refractory)
    target += self.output.filter(post_vs)


class Exponential(TwoEndConn):
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
       & g_{\mathrm{syn}}(t) = g_{max} g \\
       & \frac{d g}{d t} = -\frac{g}{\tau_{decay}}+\sum_{k} \delta(t-t_{j}^{k}).
       \end{aligned}

  **Model Examples**

  - `(Brunel & Hakim, 1999) Fast Global Oscillation <https://brainpy-examples.readthedocs.io/en/latest/oscillation_synchronization/Brunel_Hakim_1999_fast_oscillation.html>`_
  - `(Vreeswijk & Sompolinsky, 1996) E/I balanced network <https://brainpy-examples.readthedocs.io/en/latest/ei_nets/Vreeswijk_1996_EI_net.html>`_
  - `(Brette, et, al., 2007) CUBA <https://brainpy-examples.readthedocs.io/en/latest/ei_nets/Brette_2007_CUBA.html>`_
  - `(Tian, et al., 2020) E/I Net for fast response <https://brainpy-examples.readthedocs.io/en/latest/ei_nets/Tian_2020_EI_net_for_fast_response.html>`_

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> from brainpy.dyn import neurons, synapses, synouts
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = neurons.LIF(1)
    >>> neu2 = neurons.LIF(1)
    >>> syn1 = synapses.Exponential(neu1, neu2, bp.conn.All2All(),
    >>>                             g_max=5., output=synouts.CUBA())
    >>> net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.dyn.DSRunner(net, inputs=[('pre.input', 25.)], monitors=['pre.V', 'post.V', 'syn.g'])
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
    `sparse` and `dense`. The default is `sparse`.
  delay_step: int, ndarray, JaxArray, Initializer, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  tau: float, JaxArray, ndarray
    The time constant of decay. [ms]
  g_max: float, ndarray, JaxArray, Initializer, Callable
    The synaptic strength (the maximum conductance). Default is 1.
  name: str
    The name of this synaptic projection.
  method: str
    The numerical integration methods.

  References
  ----------

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.

  """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      output: SynapseOutput = None,
      plasticity: Optional[SynapsePlasticity] = None,
      conn_type: str = 'sparse',
      g_max: Union[float, Tensor, Initializer, Callable] = 1.,
      delay_step: Union[int, Tensor, Initializer, Callable] = None,
      tau: Union[float, Tensor] = 8.0,
      name: str = None,
      method: str = 'exp_auto',
  ):
    super(Exponential, self).__init__(pre=pre,
                                      post=post,
                                      conn=conn,
                                      output=CUBA() if output is None else output,
                                      plasticity=plasticity,
                                      name=name)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # parameters
    self.tau = tau
    if bm.size(self.tau) != 1:
      raise ValueError(f'"tau" must be a scalar or a tensor with size of 1. '
                       f'But we got {self.tau}')

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
        self.pre2post = self.conn.require('pre2post')
        self.g_max = init_param(g_max, self.pre2post[1].shape, allow_none=False)
        self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
      elif conn_type == 'dense':
        self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
        self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
        if self.weight_type == 'homo':
          self.conn_mat = self.conn.require('conn_mat')
      else:
        raise ValueError(f'Unknown connection type: {conn_type}')

    # variables
    self.g = bm.Variable(bm.zeros(self.post.num))
    self.delay_step = self.register_delay(f"{self.pre.name}.spike",
                                          delay_step,
                                          self.pre.spike)

    # function
    self.integral = odeint(lambda g, t: -g / self.tau, method=method)

  def reset(self):
    self.g.value = bm.zeros(self.post.num)
    self.output.reset()

  def update(self, t, dt):
    # delays
    pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)

    # update sub-components
    self.output.update(t, dt)
    self.plasticity.update(t, dt, pre_spike, self.post.spike)

    # post values
    assert self.weight_type in ['homo', 'heter']
    assert self.conn_type in ['sparse', 'dense']
    if isinstance(self.conn, All2All):
      pre_spike = pre_spike.astype(bm.float_)
      if self.weight_type == 'homo':
        post_vs = bm.sum(pre_spike)
        if not self.conn.include_self:
          post_vs = post_vs - pre_spike
        post_vs = self.g_max * post_vs
      else:
        post_vs = pre_spike @ self.g_max
    elif isinstance(self.conn, One2One):
      pre_spike = pre_spike.astype(bm.float_)
      post_vs = pre_spike * self.g_max
    else:
      if self.conn_type == 'sparse':
        post_vs = bm.pre2post_event_sum(pre_spike,
                                        self.pre2post,
                                        self.post.num,
                                        self.g_max)
      else:
        pre_spike = pre_spike.astype(bm.float_)
        if self.weight_type == 'homo':
          post_vs = self.g_max * (pre_spike @ self.conn_mat)
        else:
          post_vs = pre_spike @ self.g_max

    # updates
    self.g.value = self.integral(self.g.value, t, dt=dt) + post_vs

    # output
    self.post.input += self.output.filter(self.g)


class DualExponential(TwoEndConn):
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
        &g_{\mathrm{syn}}(t)=g_{\mathrm{max}} g \\
        &\frac{d g}{d t}=-\frac{g}{\tau_{\mathrm{decay}}}+h \\
        &\frac{d h}{d t}=-\frac{h}{\tau_{\text {rise }}}+ \delta\left(t_{0}-t\right),
        \end{aligned}

    **Model Examples**


    .. plot::
      :include-source: True

      >>> import brainpy as bp
      >>> from brainpy.dyn import neurons, synapses, synouts
      >>> import matplotlib.pyplot as plt
      >>>
      >>> neu1 = neurons.LIF(1)
      >>> neu2 = neurons.LIF(1)
      >>> syn1 = synapses.DualExponential(neu1, neu2, bp.connect.All2All(), output=synouts.CUBA())
      >>> net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)
      >>>
      >>> runner = bp.dyn.DSRunner(net, inputs=[('pre.input', 25.)], monitors=['pre.V', 'post.V', 'syn.g', 'syn.h'])
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
      >>> plt.plot(runner.mon.ts, runner.mon['syn.h'], label='h')
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
      `sparse` and `dense`. The default is `sparse`.
    delay_step: int, ndarray, JaxArray, Initializer, Callable
      The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
    tau_decay: float, JaxArray, JaxArray, ndarray
      The time constant of the synaptic decay phase. [ms]
    tau_rise: float, JaxArray, JaxArray, ndarray
      The time constant of the synaptic rise phase. [ms]
    g_max: float, ndarray, JaxArray, Initializer, Callable
      The synaptic strength (the maximum conductance). Default is 1.
    name: str
      The name of this synaptic projection.
    method: str
      The numerical integration methods.

    References
    ----------

    .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
           "The Synapse." Principles of Computational Modelling in Neuroscience.
           Cambridge: Cambridge UP, 2011. 172-95. Print.
    .. [2] Roth, A., & Van Rossum, M. C. W. (2009). Modeling Synapses. Computational
           Modeling Methods for Neuroscientists.

    """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      plasticity: Optional[SynapsePlasticity] = None,
      output: SynapseOutput = None,
      conn_type: str = 'dense',
      g_max: Union[float, Tensor, Initializer, Callable] = 1.,
      tau_decay: Union[float, Tensor] = 10.0,
      tau_rise: Union[float, Tensor] = 1.,
      delay_step: Union[int, Tensor, Initializer, Callable] = None,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(DualExponential, self).__init__(pre=pre,
                                          post=post,
                                          conn=conn,
                                          output=CUBA() if output is None else output,
                                          plasticity=plasticity,
                                          name=name)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input')

    # parameters
    self.tau_rise = tau_rise
    self.tau_decay = tau_decay
    if bm.size(self.tau_rise) != 1:
      raise ValueError(f'"tau_rise" must be a scalar or a tensor with size of 1. '
                       f'But we got {self.tau_rise}')
    if bm.size(self.tau_decay) != 1:
      raise ValueError(f'"tau_decay" must be a scalar or a tensor with size of 1. '
                       f'But we got {self.tau_decay}')

    # connections
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
    self.h = bm.Variable(bm.zeros(self.pre.num))
    self.g = bm.Variable(bm.zeros(self.pre.num))
    self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)

    # integral
    self.integral = odeint(method=method, f=JointEq([self.dg, self.dh]))

  def reset(self):
    self.h.value = bm.zeros(self.pre.num)
    self.g.value = bm.zeros(self.pre.num)
    self.output.reset()

  def dh(self, h, t):
    return -h / self.tau_rise

  def dg(self, g, t, h):
    return -g / self.tau_decay + h

  def update(self, t, dt):
    # delays
    pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)

    # update sub-components
    self.output.update(t, dt)
    self.plasticity.update(t, dt, pre_spike, self.post.spike)

    # update synaptic variables
    self.g.value, self.h.value = self.integral(self.g, self.h, t, dt)
    self.h += pre_spike

    # post-synaptic values
    syn_value = self.plasticity.filter(self.g)

    if isinstance(self.conn, All2All):
      if self.weight_type == 'homo':
        post_vs = bm.sum(syn_value)
        if not self.conn.include_self:
          post_vs = post_vs - syn_value
        post_vs = self.g_max * post_vs
      else:
        post_vs = syn_value @ self.g_max
    elif isinstance(self.conn, One2One):
      post_vs = self.g_max * syn_value
    else:
      if self.conn_type == 'sparse':
        post_vs = bm.pre2post_sum(syn_value, self.post.num, self.post_ids, self.pre_ids)
      else:
        if self.weight_type == 'homo':
          post_vs = (self.g_max * syn_value) @ self.conn_mat
        else:
          post_vs = syn_value @ self.g_max

    # output
    self.post.input += self.output.filter(post_vs)


class Alpha(DualExponential):
  r"""Alpha synapse model.

  **Model Descriptions**

  The analytical expression of alpha synapse is given by:

  .. math::

      g_{syn}(t)= g_{max} \frac{t-t_{s}}{\tau} \exp \left(-\frac{t-t_{s}}{\tau}\right).

  While, this equation is hard to implement. So, let's try to convert it into the
  differential forms:

  .. math::

      \begin{aligned}
      &g_{\mathrm{syn}}(t)= g_{\mathrm{max}} g \\
      &\frac{d g}{d t}=-\frac{g}{\tau}+h \\
      &\frac{d h}{d t}=-\frac{h}{\tau}+\delta\left(t_{0}-t\right)
      \end{aligned}

  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> from brainpy.dyn import neurons, synapses, synouts
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = neurons.LIF(1)
    >>> neu2 = neurons.LIF(1)
    >>> syn1 = synapses.Alpha(neu1, neu2, bp.connect.All2All(), output=synouts.CUBA())
    >>> net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.dyn.DSRunner(net, inputs=[('pre.input', 25.)], monitors=['pre.V', 'post.V', 'syn.g', 'syn.h'])
    >>> runner.run(150.)
    >>>
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
    >>> fig.add_subplot(gs[0, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
    >>> plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
    >>> plt.legend()
    >>> fig.add_subplot(gs[1, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g')
    >>> plt.plot(runner.mon.ts, runner.mon['syn.h'], label='h')
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
    `sparse` and `dense`. The default is `sparse`.
  delay_step: int, ndarray, JaxArray, Initializer, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  tau_decay: float, JaxArray, ndarray
    The time constant of the synaptic decay phase. [ms]
  g_max: float, ndarray, JaxArray, Initializer, Callable
    The synaptic strength (the maximum conductance). Default is 1.
  name: str
    The name of this synaptic projection.
  method: str
    The numerical integration methods.

  References
  ----------

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.
  """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      output: SynapseOutput = None,
      plasticity: Optional[SynapsePlasticity] = None,
      conn_type: str = 'dense',
      g_max: Union[float, Tensor, Initializer, Callable] = 1.,
      delay_step: Union[int, Tensor, Initializer, Callable] = None,
      tau_decay: Union[float, Tensor] = 10.0,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(Alpha, self).__init__(pre=pre,
                                post=post,
                                conn=conn,
                                conn_type=conn_type,
                                delay_step=delay_step,
                                g_max=g_max,
                                tau_decay=tau_decay,
                                tau_rise=tau_decay,
                                method=method,
                                output=CUBA() if output is None else output,
                                plasticity=plasticity,
                                name=name)


class NMDA(TwoEndConn):
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


  **Model Examples**

  - `(Wang, 2002) Decision making spiking model <https://brainpy-examples.readthedocs.io/en/latest/decision_making/Wang_2002_decision_making_spiking.html>`_


  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> from brainpy.dyn import synapses, neurons
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = neurons.HH(1)
    >>> neu2 = neurons.HH(1)
    >>> syn1 = synapses.NMDA(neu1, neu2, bp.connect.All2All(), E=0.)
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
  tau_decay: float, JaxArray, ndarray
    The time constant of the synaptic decay phase. Default 100 [ms]
  tau_rise: float, JaxArray, ndarray
    The time constant of the synaptic rise phase. Default 2 [ms]
  a: float, JaxArray, ndarray
    Default 0.5 ms^-1.
  name: str
    The name of this synaptic projection.
  method: str
    The numerical integration methods.
  E: float, JaxArray, ndarray
    The reversal potential for the synaptic current. [mV]

    .. deprecated:: 2.1.13
       Parameter `E` is no longer supported. Please use :py:class:`~.MgBlock` instead.

  alpha: float, JaxArray, ndarray
    Binding constant. Default 0.062

    .. deprecated:: 2.1.13
       Parameter `alpha` is no longer supported. Please use :py:class:`~.MgBlock` instead.

  beta: float, JaxArray, ndarray
    Unbinding constant. Default 3.57

    .. deprecated:: 2.1.13
       Parameter `beta` is no longer supported. Please use :py:class:`~.MgBlock` instead.

  cc_Mg: float, JaxArray, ndarray
    Concentration of Magnesium ion. Default 1.2 [mM].

    .. deprecated:: 2.1.13
       Parameter `cc_Mg` is no longer supported. Please use :py:class:`~.MgBlock` instead.

  References
  ----------

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

  """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      output: Optional[SynapseOutput] = None,
      plasticity: Optional[SynapsePlasticity] = None,
      conn_type: str = 'dense',
      g_max: Union[float, Tensor, Initializer, Callable] = 0.15,
      delay_step: Union[int, Tensor, Initializer, Callable] = None,
      tau_decay: Union[float, Tensor] = 100.,
      a: Union[float, Tensor] = 0.5,
      tau_rise: Union[float, Tensor] = 2.,
      method: str = 'exp_auto',
      name: str = None,

      # deprecated
      alpha=None,
      beta=None,
      cc_Mg=None,
      E=None,
  ):

    if output is not None:
      if alpha is not None:
        raise ValueError(f'Please set "alpha" in "output" argument.')
      if beta is not None:
        raise ValueError(f'Please set "beta" in "output" argument.')
      if cc_Mg is not None:
        raise ValueError(f'Please set "cc_Mg" in "output" argument.')
      if E is not None:
        raise ValueError(f'Please set "E" in "output" argument.')
    else:
      if alpha is not None:
        warnings.warn('Please set "alpha" by using "output=bp.dyn.synouts.MgBlock(alpha)" instead.',
                      DeprecationWarning)
      else:
        alpha = 0.062
      if beta is not None:
        warnings.warn('Please set "beta" by using "output=bp.dyn.synouts.MgBlock(beta)" instead.',
                      DeprecationWarning)
      else:
        beta = 3.57
      if cc_Mg is not None:
        warnings.warn('Please set "cc_Mg" by using "output=bp.dyn.synouts.MgBlock(cc_Mg)" instead.',
                      DeprecationWarning)
      else:
        cc_Mg = 1.2
      if E is not None:
        warnings.warn('Please set "E" by using "output=bp.dyn.synouts.MgBlock(E)" instead.',
                      DeprecationWarning)
      else:
        E = 0.
      output = MgBlock(E=E, alpha=alpha, beta=beta, cc_Mg=cc_Mg)
    super(NMDA, self).__init__(pre=pre,
                               post=post,
                               conn=conn,
                               output=output,
                               plasticity=plasticity,
                               name=name)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # parameters
    self.tau_decay = tau_decay
    self.tau_rise = tau_rise
    self.a = a
    if bm.size(a) != 1:
      raise ValueError(f'"a" must be a scalar or a tensor with size of 1. But we got {a}')
    if bm.size(tau_decay) != 1:
      raise ValueError(f'"tau_decay" must be a scalar or a tensor with size of 1. But we got {tau_decay}')
    if bm.size(tau_rise) != 1:
      raise ValueError(f'"tau_rise" must be a scalar or a tensor with size of 1. But we got {tau_rise}')

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
    self.g = bm.Variable(bm.zeros(self.pre.num))
    self.x = bm.Variable(bm.zeros(self.pre.num))
    self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)

    # integral
    self.integral = odeint(method=method, f=JointEq(self.dg, self.dx))

  def dg(self, g, t, x):
    return -g / self.tau_decay + self.a * x * (1 - g)

  def dx(self, x, t):
    return -x / self.tau_rise

  def reset(self):
    self.g[:] = 0
    self.x[:] = 0
    self.output.reset()
    self.plasticity.reset()

  def update(self, t, dt):
    # delays
    pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)

    # update sub-components
    self.output.update(t, dt)
    self.plasticity.update(t, dt, pre_spike, self.post.spike)

    # update synapse variables
    self.g.value, self.x.value = self.integral(self.g, self.x, t, dt=dt)
    self.x += pre_spike

    # post-synaptic value
    syn_value = self.plasticity.filter(self.g)

    if isinstance(self.conn, All2All):
      if self.weight_type == 'homo':
        post_g = bm.sum(syn_value)
        if not self.conn.include_self:
          post_g = post_g - syn_value
        post_g = post_g * self.g_max
      else:
        post_g = syn_value @ self.g_max
    elif isinstance(self.conn, One2One):
      post_g = self.g_max * syn_value
    else:
      if self.conn_type == 'sparse':
        post_g = bm.pre2post_sum(syn_value, self.post.num, self.post_ids, self.pre_ids)
      else:
        if self.weight_type == 'homo':
          post_g = (self.g_max * syn_value) @ self.conn_mat
        else:
          post_g = syn_value @ self.g_max

    # output
    self.post.input += self.output.filter(post_g)
