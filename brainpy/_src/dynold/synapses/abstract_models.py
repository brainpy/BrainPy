# -*- coding: utf-8 -*-

from typing import Union, Dict, Callable, Optional

import jax

import brainpy.math as bm
from brainpy._src.connect import TwoEndConnector, All2All, One2One
from brainpy._src.context import share
from brainpy._src.dyn import synapses
from brainpy._src.dyn.base import NeuDyn
from brainpy._src.dnn import linear
from brainpy._src.dynold.synouts import MgBlock, CUBA
from brainpy._src.initialize import Initializer, variable_
from brainpy._src.integrators.ode.generic import odeint
from brainpy._src.dyn.projections.aligns import _pre_delay_repr, _init_delay
from brainpy.types import ArrayType
from .base import TwoEndConn, _SynSTP, _SynOut, _TwoEndConnAlignPre

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
    >>> from brainpy import synapses, neurons
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = neurons.LIF(1)
    >>> neu2 = neurons.LIF(1)
    >>> syn1 = synapses.Alpha(neu1, neu2, bp.connect.All2All(), g_max=5.)
    >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.DSRunner(net, inputs=[('pre.input', 25.), ('post.input', 10.)], monitors=['pre.V', 'post.V', 'pre.spike'])
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
  pre: NeuDyn
    The pre-synaptic neuron group.
  post: NeuDyn
    The post-synaptic neuron group.
  conn: optional, ArrayType, dict of (str, ndarray), TwoEndConnector
    The synaptic connections.
  comp_method: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `sparse`.
  delay_step: int, ArrayType, Initializer, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  g_max: float, ArrayType, Initializer, Callable
    The synaptic strength. Default is 1.
  post_ref_key: str
    Whether the post-synaptic group has refractory period.
  """

  def __init__(
      self,
      pre: NeuDyn,
      post: NeuDyn,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      output: _SynOut = CUBA(target_var='V'),
      stp: Optional[_SynSTP] = None,
      comp_method: str = 'sparse',
      g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
      delay_step: Union[float, ArrayType, Initializer, Callable] = None,
      post_ref_key: str = None,
      name: str = None,
      mode: bm.Mode = None,
      stop_spike_gradient: bool = False,
  ):
    super().__init__(name=name,
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
    self.g_max, self.conn_mask = self._init_weights(g_max, comp_method=comp_method, sparse_data='csr')

    # register delay
    self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)

  def reset_state(self, batch_size=None):
    self.output.reset_state(batch_size)
    if self.stp is not None:
      self.stp.reset_state(batch_size)

  def update(self, pre_spike=None):
    # pre-synaptic spikes
    if pre_spike is None:
      pre_spike = self.get_delay_data(f"{self.pre.name}.spike", delay_step=self.delay_step)
    pre_spike = bm.as_jax(pre_spike)
    if self.stop_spike_gradient:
      pre_spike = jax.lax.stop_gradient(pre_spike)

    # update sub-components
    if self.stp is not None:
      self.stp.update(pre_spike)

    # synaptic values onto the post
    if isinstance(self.conn, All2All):
      syn_value = bm.asarray(pre_spike, dtype=bm.float_)
      if self.stp is not None:
        syn_value = self.stp(syn_value)
      post_vs = self._syn2post_with_all2all(syn_value, self.g_max)
    elif isinstance(self.conn, One2One):
      syn_value = bm.asarray(pre_spike, dtype=bm.float_)
      if self.stp is not None:
        syn_value = self.stp(syn_value)
      post_vs = self._syn2post_with_one2one(syn_value, self.g_max)
    else:
      if self.comp_method == 'sparse':
        if self.stp is not None:
          syn_value = self.stp(pre_spike)
          f = lambda s: bm.sparse.csrmv(
            self.g_max, self.conn_mask[0], self.conn_mask[1], s,
            shape=(self.pre.num, self.post.num), transpose=True
          )
        else:
          syn_value = pre_spike
          f = lambda s: bm.event.csrmv(
            self.g_max, self.conn_mask[0], self.conn_mask[1], s,
            shape=(self.pre.num, self.post.num), transpose=True
          )
        if isinstance(self.mode, bm.BatchingMode): f = jax.vmap(f)
        post_vs = f(syn_value)
      else:
        syn_value = bm.asarray(pre_spike, dtype=bm.float_)
        if self.stp is not None:
          syn_value = self.stp(syn_value)
        post_vs = self._syn2post_with_dense(syn_value, self.g_max, self.conn_mask)
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
    >>> from brainpy import neurons, synapses, synouts
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = neurons.LIF(1)
    >>> neu2 = neurons.LIF(1)
    >>> syn1 = synapses.Exponential(neu1, neu2, bp.conn.All2All(),
    >>>                             g_max=5., output=synouts.CUBA())
    >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.DSRunner(net, inputs=[('pre.input', 25.)], monitors=['pre.V', 'post.V', 'syn.g'])
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
  conn: optional, ArrayType, dict of (str, ndarray), TwoEndConnector
    The synaptic connections.
  comp_method: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `sparse`.
  delay_step: int, ArrayType, Initializer, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  tau: float, ArrayType
    The time constant of decay. [ms]
  g_max: float, ArrayType, Initializer, Callable
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
      pre: NeuDyn,
      post: NeuDyn,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      output: Optional[_SynOut] = CUBA(),
      stp: Optional[_SynSTP] = None,
      comp_method: str = 'sparse',
      g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
      delay_step: Union[int, ArrayType, Initializer, Callable] = None,
      tau: Union[float, ArrayType] = 8.0,
      method: str = 'exp_auto',

      # other parameters
      name: str = None,
      mode: bm.Mode = None,
      stop_spike_gradient: bool = False,
  ):
    super().__init__(pre=pre,
                     post=post,
                     conn=conn,
                     output=output,
                     stp=stp,
                     name=name,
                     mode=mode)
    # parameters
    self.stop_spike_gradient = stop_spike_gradient

    # synapse dynamics
    self.syn = synapses.Expon(post.varshape, tau=tau, method=method)

    # Projection
    if isinstance(conn, All2All):
      self.comm = linear.AllToAll(pre.num, post.num, g_max)
    elif isinstance(conn, One2One):
      assert post.num == pre.num
      self.comm = linear.OneToOne(pre.num, g_max)
    else:
      if comp_method == 'dense':
        self.comm = linear.MaskedLinear(conn, g_max)
      elif comp_method == 'sparse':
        if self.stp is None:
          self.comm = linear.EventCSRLinear(conn, g_max)
        else:
          self.comm = linear.CSRLinear(conn, g_max)
      else:
        raise ValueError(f'Does not support {comp_method}, only "sparse" or "dense".')

    # variables
    self.g = self.syn.g

    # delay
    self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)

  def reset_state(self, batch_size=None):
    self.syn.reset_state(batch_size)
    self.output.reset_state(batch_size)
    if self.stp is not None:
      self.stp.reset_state(batch_size)

  def update(self, pre_spike=None):
    # delays
    if pre_spike is None:
      pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)
    pre_spike = bm.as_jax(pre_spike)
    if self.stop_spike_gradient:
      pre_spike = jax.lax.stop_gradient(pre_spike)

    # update sub-components
    self.output.update()
    if self.stp is not None:
      self.stp.update(pre_spike)
      pre_spike = self.stp(pre_spike)

    # post values
    g = self.syn(self.comm(pre_spike))

    # output
    return self.output(g)


class DualExponential(_TwoEndConnAlignPre):
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
      >>> from brainpy import neurons, synapses, synouts
      >>> import matplotlib.pyplot as plt
      >>>
      >>> neu1 = neurons.LIF(1)
      >>> neu2 = neurons.LIF(1)
      >>> syn1 = synapses.DualExponential(neu1, neu2, bp.connect.All2All(), output=synouts.CUBA())
      >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
      >>>
      >>> runner = bp.DSRunner(net, inputs=[('pre.input', 25.)], monitors=['pre.V', 'post.V', 'syn.g', 'syn.h'])
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
    pre: NeuDyn
      The pre-synaptic neuron group.
    post: NeuDyn
      The post-synaptic neuron group.
    conn: optional, ArrayType, dict of (str, ndarray), TwoEndConnector
      The synaptic connections.
    comp_method: str
      The connection type used for model speed optimization. It can be
      `sparse` and `dense`. The default is `sparse`.
    delay_step: int, ArrayType, Initializer, Callable
      The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
    tau_decay: float, ArrayArray, ndarray
      The time constant of the synaptic decay phase. [ms]
    tau_rise: float, ArrayArray, ndarray
      The time constant of the synaptic rise phase. [ms]
    g_max: float, ArrayType, Initializer, Callable
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
      pre: NeuDyn,
      post: NeuDyn,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      stp: Optional[_SynSTP] = None,
      output: _SynOut = None,  # CUBA(),
      comp_method: str = 'dense',
      g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
      tau_decay: Union[float, ArrayType] = 10.0,
      tau_rise: Union[float, ArrayType] = 1.,
      delay_step: Union[int, ArrayType, Initializer, Callable] = None,
      method: str = 'exp_auto',

      # other parameters
      name: str = None,
      mode: bm.Mode = None,
      stop_spike_gradient: bool = False,
  ):

    # parameters
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

    syn = synapses.DualExpon(pre.size,
                             pre.keep_size,
                             mode=mode,
                             tau_decay=tau_decay,
                             tau_rise=tau_rise,
                             method=method, )

    super().__init__(pre=pre,
                     post=post,
                     syn=syn,
                     conn=conn,
                     output=output,
                     stp=stp,
                     comp_method=comp_method,
                     g_max=g_max,
                     delay_step=delay_step,
                     name=name,
                     mode=mode)

    self.check_post_attrs('input')
    # copy the references
    self.g = syn.g
    self.h = syn.h

  def update(self, pre_spike=None):
    return super().update(pre_spike, stop_spike_gradient=self.stop_spike_gradient)


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
    >>> from brainpy import neurons, synapses, synouts
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = neurons.LIF(1)
    >>> neu2 = neurons.LIF(1)
    >>> syn1 = synapses.Alpha(neu1, neu2, bp.connect.All2All(), output=synouts.CUBA())
    >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.DSRunner(net, inputs=[('pre.input', 25.)], monitors=['pre.V', 'post.V', 'syn.g', 'syn.h'])
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
  pre: NeuDyn
    The pre-synaptic neuron group.
  post: NeuDyn
    The post-synaptic neuron group.
  conn: optional, ArrayType, dict of (str, ndarray), TwoEndConnector
    The synaptic connections.
  comp_method: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `sparse`.
  delay_step: int, ArrayType, Initializer, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  tau_decay: float, ArrayType
    The time constant of the synaptic decay phase. [ms]
  g_max: float, ArrayType, Initializer, Callable
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
      pre: NeuDyn,
      post: NeuDyn,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      output: _SynOut = None,  # CUBA(),
      stp: Optional[_SynSTP] = None,
      comp_method: str = 'dense',
      g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
      delay_step: Union[int, ArrayType, Initializer, Callable] = None,
      tau_decay: Union[float, ArrayType] = 10.0,
      method: str = 'exp_auto',

      # other parameters
      name: str = None,
      mode: bm.Mode = None,
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


class NMDA(_TwoEndConnAlignPre):
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
    >>> from brainpy import synapses, neurons
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = neurons.HH(1)
    >>> neu2 = neurons.HH(1)
    >>> syn1 = synapses.NMDA(neu1, neu2, bp.connect.All2All())
    >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.DSRunner(net, inputs=[('pre.input', 5.)], monitors=['pre.V', 'post.V', 'syn.g', 'syn.x'])
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
  pre: NeuDyn
    The pre-synaptic neuron group.
  post: NeuDyn
    The post-synaptic neuron group.
  conn: optional, ArrayType, dict of (str, ndarray), TwoEndConnector
    The synaptic connections.
  comp_method: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `dense`.
  delay_step: int, ArrayType, Initializer, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  g_max: float, ArrayType, Initializer, Callable
    The synaptic strength (the maximum conductance). Default is 1.
  tau_decay: float, ArrayType
    The time constant of the synaptic decay phase. Default 100 [ms]
  tau_rise: float, ArrayType
    The time constant of the synaptic rise phase. Default 2 [ms]
  a: float, ArrayType
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
      pre: NeuDyn,
      post: NeuDyn,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      output: _SynOut = MgBlock(E=0., alpha=0.062, beta=3.57, cc_Mg=1.2),
      stp: Optional[_SynSTP] = None,
      comp_method: str = 'dense',
      g_max: Union[float, ArrayType, Initializer, Callable] = 0.15,
      delay_step: Union[int, ArrayType, Initializer, Callable] = None,
      tau_decay: Union[float, ArrayType] = 100.,
      a: Union[float, ArrayType] = 0.5,
      tau_rise: Union[float, ArrayType] = 2.,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
      stop_spike_gradient: bool = False,
  ):
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
    self.comp_method = comp_method
    self.stop_spike_gradient = stop_spike_gradient

    syn = synapses.NMDA(pre.size,
                        pre.keep_size,
                        mode=mode,
                        a=a,
                        tau_decay=tau_decay,
                        tau_rise=tau_rise,
                        method=method, )

    super().__init__(pre=pre,
                     post=post,
                     syn=syn,
                     conn=conn,
                     output=output,
                     stp=stp,
                     comp_method=comp_method,
                     g_max=g_max,
                     delay_step=delay_step,
                     name=name,
                     mode=mode)

    # copy the references
    self.g = syn.g
    self.x = syn.x

  def update(self, pre_spike=None):
    return super().update(pre_spike, stop_spike_gradient=self.stop_spike_gradient)
