# -*- coding: utf-8 -*-

from typing import Union, Dict, Callable, Optional

from jax import vmap
from jax.lax import stop_gradient

import brainpy.math as bm
from brainpy.connect import TwoEndConnector, All2All, One2One
from brainpy.dyn.base import NeuGroup, SynOut, SynSTP, TwoEndConn, SynConn
from brainpy.initialize import Initializer, variable_
from brainpy.integrators import odeint, JointEq
from brainpy.tools.checking import check_integer, check_float
from brainpy.modes import Mode, BatchingMode, normal, NormalMode, check_mode
from brainpy.types import Array
from ..synouts import CUBA, MgBlock


__all__ = [
  'Delta',
  'Exponential',
  'DualExponential',
  'Alpha',
  'NMDA',
  'PoissonInput',
]


class Delta(TwoEndConn):
  r"""Voltage Jump Synapse Model, or alias of Delta Synapse Model.

  **Model Descriptions**

  .. math::

      I_{syn} (t) = \sum_{j\in C} g_{\mathrm{max}} * \mathrm{STP} * \delta(t-t_j-D)

  where :math:`g_{\mathrm{max}}` denotes the chemical synaptic strength,
  :math:`t_j` the spiking moment of the presynaptic neuron :math:`j`,
  :math:`C` the set of neurons connected to the post-synaptic neuron,
  :math:`D` the transmission delay of chemical synapses,
  and :math:`\mathrm{STP}` the short-term plasticity effect.
  For simplicity, the rise and decay phases of post-synaptic currents are
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
  comp_method: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `sparse`.
  delay_step: int, ndarray, JaxArray, Initializer, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  g_max: float, ndarray, JaxArray, Initializer, Callable
    The synaptic strength. Default is 1.
  post_ref_key: str
    Whether the post-synaptic group has refractory period.
  """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Array, Dict[str, Array]],
      output: SynOut = CUBA(target_var='V'),
      stp: Optional[SynSTP] = None,
      comp_method: str = 'sparse',
      g_max: Union[float, Array, Initializer, Callable] = 1.,
      delay_step: Union[float, Array, Initializer, Callable] = None,
      post_ref_key: str = None,

      # other parameters
      name: str = None,
      mode: Mode = normal,
      stop_spike_gradient: bool = False,
  ):
    super(Delta, self).__init__(name=name,
                                pre=pre,
                                post=post,
                                conn=conn,
                                output=output,
                                stp=stp,
                                mode=mode)

    # parameters
    self.stop_spike_gradient = stop_spike_gradient
    self.post_ref_key = post_ref_key
    if post_ref_key:
      self.check_post_attrs(post_ref_key)
    self.comp_method = comp_method

    # connections and weights
    self.g_max, self.conn_mask = self.init_weights(g_max, comp_method=comp_method, sparse_data='csr')

    # register delay
    self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)

  def reset_state(self, batch_size=None):
    self.output.reset_state(batch_size)
    if self.stp is not None: self.stp.reset_state(batch_size)

  def update(self, tdi, pre_spike=None):
    # pre-synaptic spikes
    if pre_spike is None:
      pre_spike = self.get_delay_data(f"{self.pre.name}.spike", delay_step=self.delay_step)
    if self.stop_spike_gradient:
      pre_spike = pre_spike.value if isinstance(pre_spike, bm.JaxArray) else pre_spike
      pre_spike = stop_gradient(pre_spike)

    # update sub-components
    self.output.update(tdi)
    if self.stp is not None: self.stp.update(tdi, pre_spike)

    # synaptic values onto the post
    if isinstance(self.conn, All2All):
      syn_value = self.stp(bm.asarray(pre_spike, dtype=bm.dftype()))
      post_vs = self.syn2post_with_all2all(syn_value, self.g_max)
    elif isinstance(self.conn, One2One):
      syn_value = self.stp(bm.asarray(pre_spike, dtype=bm.dftype()))
      post_vs = self.syn2post_with_one2one(syn_value, self.g_max)
    else:
      if self.comp_method == 'sparse':
        f = lambda s: bm.pre2post_event_sum(s, self.conn_mask, self.post.num, self.g_max)
        if isinstance(self.mode, BatchingMode): f = vmap(f)
        post_vs = f(pre_spike)
        # if not isinstance(self.stp, _NullSynSTP):
        #   raise NotImplementedError()
        #   stp_value = self.stp(1.)
        #   f2 = lambda s: bm.pre2post_sum(s, self.post.num, *self.conn_mask)
        #   if self.trainable: f2 = vmap(f2)
        #   post_vs *= f2(stp_value)
      else:
        syn_value = self.stp(bm.asarray(pre_spike, dtype=bm.dftype()))
        post_vs = self.syn2post_with_dense(syn_value, self.g_max, self.conn_mask)
    if self.post_ref_key:
      post_vs = post_vs * (1. - getattr(self.post, self.post_ref_key))

    # update outputs
    return self.output(post_vs)


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
       & g_{\mathrm{syn}}(t) = g_{max} g * \mathrm{STP} \\
       & \frac{d g}{d t} = -\frac{g}{\tau_{decay}}+\sum_{k} \delta(t-t_{j}^{k}).
       \end{aligned}
  
  where :math:`\mathrm{STP}` is used to model the short-term plasticity effect.
  
  
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
  comp_method: str
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
      conn: Union[TwoEndConnector, Array, Dict[str, Array]],
      output: SynOut = CUBA(),
      stp: Optional[SynSTP] = None,
      comp_method: str = 'sparse',
      g_max: Union[float, Array, Initializer, Callable] = 1.,
      delay_step: Union[int, Array, Initializer, Callable] = None,
      tau: Union[float, Array] = 8.0,
      method: str = 'exp_auto',

      # other parameters
      name: str = None,
      mode: Mode = normal,
      stop_spike_gradient: bool = False,
  ):
    super(Exponential, self).__init__(pre=pre,
                                      post=post,
                                      conn=conn,
                                      output=output,
                                      stp=stp,
                                      name=name,
                                      mode=mode)
    # parameters
    self.stop_spike_gradient = stop_spike_gradient
    self.comp_method = comp_method
    self.tau = tau
    if bm.size(self.tau) != 1:
      raise ValueError(f'"tau" must be a scalar or a tensor with size of 1. But we got {self.tau}')

    # connections and weights
    self.g_max, self.conn_mask = self.init_weights(g_max, comp_method, sparse_data='csr')

    # variables
    self.g = variable_(bm.zeros, self.post.num, mode)
    self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)

    # function
    self.integral = odeint(lambda g, t: -g / self.tau, method=method)

  def reset_state(self, batch_size=None):
    self.g.value = variable_(bm.zeros, self.post.num, batch_size)
    self.output.reset_state(batch_size)
    if self.stp is not None: self.stp.reset_state(batch_size)

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

    # post values
    if isinstance(self.conn, All2All):
      syn_value = bm.asarray(pre_spike, dtype=bm.dftype())
      if self.stp is not None: syn_value = self.stp(syn_value)
      post_vs = self.syn2post_with_all2all(syn_value, self.g_max)
    elif isinstance(self.conn, One2One):
      syn_value = bm.asarray(pre_spike, dtype=bm.dftype())
      if self.stp is not None: syn_value = self.stp(syn_value)
      post_vs = self.syn2post_with_one2one(syn_value, self.g_max)
    else:
      if self.comp_method == 'sparse':
        f = lambda s: bm.pre2post_event_sum(s, self.conn_mask, self.post.num, self.g_max)
        if isinstance(self.mode, BatchingMode): f = vmap(f)
        post_vs = f(pre_spike)
        # if not isinstance(self.stp, _NullSynSTP):
        #   raise NotImplementedError()
      else:
        syn_value = bm.asarray(pre_spike, dtype=bm.dftype())
        if self.stp is not None: syn_value = self.stp(syn_value)
        post_vs = self.syn2post_with_dense(syn_value, self.g_max, self.conn_mask)
    # updates
    self.g.value = self.integral(self.g.value, t, dt) + post_vs

    # output
    return self.output(self.g)


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
        &g_{\mathrm{syn}}(t)=g_{\mathrm{max}} g * \mathrm{STP} \\
        &\frac{d g}{d t}=-\frac{g}{\tau_{\mathrm{decay}}}+h \\
        &\frac{d h}{d t}=-\frac{h}{\tau_{\text {rise }}}+ \delta\left(t_{0}-t\right),
        \end{aligned}

    where :math:`\mathrm{STP}` is used to model the short-term plasticity effect of synapses.

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
    comp_method: str
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
      conn: Union[TwoEndConnector, Array, Dict[str, Array]],
      stp: Optional[SynSTP] = None,
      output: SynOut = CUBA(),
      comp_method: str = 'dense',
      g_max: Union[float, Array, Initializer, Callable] = 1.,
      tau_decay: Union[float, Array] = 10.0,
      tau_rise: Union[float, Array] = 1.,
      delay_step: Union[int, Array, Initializer, Callable] = None,
      method: str = 'exp_auto',

      # other parameters
      name: str = None,
      mode: Mode = normal,
      stop_spike_gradient: bool = False,
  ):
    super(DualExponential, self).__init__(pre=pre,
                                          post=post,
                                          conn=conn,
                                          output=output,
                                          stp=stp,
                                          name=name,
                                          mode=mode)
    # parameters
    # self.check_pre_attrs('spike')
    self.check_post_attrs('input')
    self.stop_spike_gradient = stop_spike_gradient
    self.comp_method = comp_method
    self.tau_rise = tau_rise
    self.tau_decay = tau_decay
    if bm.size(self.tau_rise) != 1:
      raise ValueError(f'"tau_rise" must be a scalar or a tensor with size of 1. '
                       f'But we got {self.tau_rise}')
    if bm.size(self.tau_decay) != 1:
      raise ValueError(f'"tau_decay" must be a scalar or a tensor with size of 1. '
                       f'But we got {self.tau_decay}')

    # connections
    self.g_max, self.conn_mask = self.init_weights(g_max, comp_method, sparse_data='ij')

    # variables
    self.h = variable_(bm.zeros, self.pre.num, mode)
    self.g = variable_(bm.zeros, self.pre.num, mode)
    self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)

    # integral
    self.integral = odeint(method=method, f=JointEq([self.dg, self.dh]))

  def reset_state(self, batch_size=None):
    self.h.value = variable_(bm.zeros, self.pre.num, batch_size)
    self.g.value = variable_(bm.zeros, self.pre.num, batch_size)
    self.output.reset_state(batch_size)
    if self.stp is not None: self.stp.reset_state(batch_size)

  def dh(self, h, t):
    return -h / self.tau_rise

  def dg(self, g, t, h):
    return -g / self.tau_decay + h

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

    # update synaptic variables
    self.g.value, self.h.value = self.integral(self.g, self.h, t, dt)
    self.h += pre_spike

    # post values
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
  comp_method: str
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
      conn: Union[TwoEndConnector, Array, Dict[str, Array]],
      output: SynOut = CUBA(),
      stp: Optional[SynSTP] = None,
      comp_method: str = 'dense',
      g_max: Union[float, Array, Initializer, Callable] = 1.,
      delay_step: Union[int, Array, Initializer, Callable] = None,
      tau_decay: Union[float, Array] = 10.0,
      method: str = 'exp_auto',

      # other parameters
      name: str = None,
      mode: Mode = normal,
      stop_spike_gradient: bool = False,
  ):
    super(Alpha, self).__init__(pre=pre,
                                post=post,
                                conn=conn,
                                comp_method=comp_method,
                                delay_step=delay_step,
                                g_max=g_max,
                                tau_decay=tau_decay,
                                tau_rise=tau_decay,
                                method=method,
                                output=output,
                                stp=stp,
                                name=name,
                                mode=mode,
                                stop_spike_gradient=stop_spike_gradient)


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
  comp_method: str
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
      conn: Union[TwoEndConnector, Array, Dict[str, Array]],
      output: SynOut = MgBlock(E=0., alpha=0.062, beta=3.57, cc_Mg=1.2),
      stp: Optional[SynSTP] = None,
      comp_method: str = 'dense',
      g_max: Union[float, Array, Initializer, Callable] = 0.15,
      delay_step: Union[int, Array, Initializer, Callable] = None,
      tau_decay: Union[float, Array] = 100.,
      a: Union[float, Array] = 0.5,
      tau_rise: Union[float, Array] = 2.,
      method: str = 'exp_auto',

      # other parameters
      name: str = None,
      mode: Mode = normal,
      stop_spike_gradient: bool = False,
  ):
    super(NMDA, self).__init__(pre=pre,
                               post=post,
                               conn=conn,
                               output=output,
                               stp=stp,
                               name=name,
                               mode=mode)
    # parameters
    # self.check_post_attrs('input', 'V')
    self.tau_decay = tau_decay
    self.tau_rise = tau_rise
    self.a = a
    if bm.size(a) != 1:
      raise ValueError(f'"a" must be a scalar or a tensor with size of 1. But we got {a}')
    if bm.size(tau_decay) != 1:
      raise ValueError(f'"tau_decay" must be a scalar or a tensor with size of 1. But we got {tau_decay}')
    if bm.size(tau_rise) != 1:
      raise ValueError(f'"tau_rise" must be a scalar or a tensor with size of 1. But we got {tau_rise}')
    self.comp_method = comp_method
    self.stop_spike_gradient = stop_spike_gradient

    # connections and weights
    self.g_max, self.conn_mask = self.init_weights(g_max, comp_method, sparse_data='ij')

    # variables
    self.g = variable_(bm.zeros, self.pre.num, mode)
    self.x = variable_(bm.zeros, self.pre.num, mode)
    self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)

    # integral
    self.integral = odeint(method=method, f=JointEq(self.dg, self.dx))

  def dg(self, g, t, x):
    return -g / self.tau_decay + self.a * x * (1 - g)

  def dx(self, x, t):
    return -x / self.tau_rise

  def reset_state(self, batch_size=None):
    self.g.value = variable_(bm.zeros, self.pre.num, batch_size)
    self.x.value = variable_(bm.zeros, self.pre.num, batch_size)
    self.output.reset_state(batch_size)
    if self.stp is not None: self.stp.reset_state(batch_size)

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

    # update synapse variables
    self.g.value, self.x.value = self.integral(self.g, self.x, t, dt=dt)
    self.x += pre_spike

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


class PoissonInput(SynConn):
  """Poisson Input to the given `Variable`.

  Adds independent Poisson input to a target variable. For large
  numbers of inputs, this is much more efficient than creating a
  `PoissonGroup`. The synaptic events are generated randomly during the
  simulation and are not preloaded and stored in memory. All the inputs must
  target the same variable, have the same frequency and same synaptic weight.
  All neurons in the target variable receive independent realizations of
  Poisson spike trains.

  Parameters
  ----------
  target_var: Variable
    The variable that is targeted by this input.
  num_input: int
    The number of inputs.
  freq: float
    The frequency of each of the inputs. Must be a scalar.
  weight: float
    The synaptic weight. Must be a scalar.
  """

  def __init__(
      self,
      target_var: bm.Variable,
      num_input: int,
      freq: Union[int, float],
      weight: Union[int, float],
      seed: Optional[int] = None,
      mode: Mode = normal,
      name: str = None
  ):
    from ..neurons.input_groups import InputGroup, OutputGroup
    super(PoissonInput, self).__init__(InputGroup(1), OutputGroup(1), name=name, mode=mode)
    self.pre = None
    self.post = None

    # check data
    if not isinstance(target_var, bm.Variable):
      raise TypeError(f'"target_var" must be an instance of Variable. '
                      f'But we got {type(target_var)}: {target_var}')
    check_integer(num_input, 'num_input', min_bound=1)
    check_float(freq, 'freq', min_bound=0., allow_int=True)
    check_float(weight, 'weight', allow_int=True)
    check_mode(mode, (NormalMode, BatchingMode), name=self.__class__.__name__)

    # parameters
    self.target_var = target_var
    self.num_input = num_input
    self.freq = freq
    self.weight = weight
    self.seed = seed
    self.rng = bm.random.RandomState(self.seed)

  def update(self, tdi):
    p = self.freq * tdi.dt / 1e3
    a = self.num_input * p
    b = self.num_input * (1 - p)
    if isinstance(tdi.dt, (int, float)):  # dt is not in tracing
      if (a > 5) and (b > 5):
        inp = self.rng.normal(a, b * p, self.target_var.shape)
      else:
        inp = self.rng.binomial(self.num_input, p, self.target_var.shape)

    else:  # dt is in tracing
      inp = bm.cond((a > 5) * (b > 5),
                    lambda _: self.rng.normal(a, b * p, self.target_var.shape),
                    lambda _: self.rng.binomial(self.num_input, p, self.target_var.shape),
                    None)
    self.target_var += inp * self.weight

  def __repr__(self):
    names = self.__class__.__name__
    return f'{names}(name={self.name}, num_input={self.num_input}, freq={self.freq}, weight={self.weight})'

  def reset_state(self, batch_size=None):
    pass

  def reset(self, batch_size=None):
    self.rng.seed(self.seed)
    self.reset_state(batch_size)

