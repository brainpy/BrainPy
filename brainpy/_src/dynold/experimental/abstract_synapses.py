# -*- coding: utf-8 -*-

from typing import Union, Dict, Callable, Optional

from jax import vmap

import brainpy.math as bm
from brainpy._src.connect import TwoEndConnector, All2All, One2One
from brainpy._src.context import share
from brainpy._src.dynold.experimental.base import SynConnNS, SynOutNS, SynSTPNS
from brainpy._src.initialize import Initializer, variable_
from brainpy._src.integrators import odeint, JointEq
from brainpy.check import is_float
from brainpy.types import ArrayType


class Exponential(SynConnNS):
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

  Parameters
  ----------
  conn: optional, ArrayType, dict of (str, ndarray), TwoEndConnector
    The synaptic connections.
  comp_method: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `sparse`.
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
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      out: Optional[SynOutNS] = None,
      stp: Optional[SynSTPNS] = None,
      comp_method: str = 'sparse',
      g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
      tau: Union[float, ArrayType] = 8.0,
      method: str = 'exp_auto',
      name: str = None,
      mode: bm.Mode = None,
  ):
    super().__init__(conn=conn,
                     out=out,
                     stp=stp,
                     name=name,
                     mode=mode)

    # parameters
    self.comp_method = comp_method
    self.tau = is_float(tau, allow_int=True)

    # connections and weights
    self.g_max, self.conn_mask = self._init_weights(g_max, comp_method, data_if_sparse='csr')

    # function
    self.integral = odeint(lambda g, t: -g / self.tau, method=method)

    # variables
    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.g = variable_(bm.zeros, self.post_num, batch_size)
    if self.out is not None:
      self.out.reset_state(batch_size)
    if self.stp is not None:
      self.stp.reset_state(batch_size)

  def update(self, pre_spike, post_v=None):
    if self.stp is not None:
      syn_value = self.stp(pre_spike) * pre_spike
    else:
      syn_value = pre_spike

    # post values
    if isinstance(self.conn, All2All):
      post_vs = self._syn2post_with_all2all(syn_value, self.g_max, self.conn.include_self)
    elif isinstance(self.conn, One2One):
      post_vs = self._syn2post_with_one2one(syn_value, self.g_max)
    else:
      if self.comp_method == 'sparse':
        if self.stp is None:
          f = lambda s: bm.event.csrmv(self.g_max,
                                       self.conn_mask[0],
                                       self.conn_mask[1],
                                       s,
                                       shape=(self.pre_num, self.post_num),
                                       transpose=True)
          if isinstance(self.mode, bm.BatchingMode):
            f = vmap(f)
        else:
          f = lambda s: bm.sparse.csrmv(self.g_max,
                                        self.conn_mask[0],
                                        self.conn_mask[1],
                                        s,
                                        shape=(self.pre_num, self.post_num),
                                        transpose=True,
                                        method='cusparse')
          if isinstance(self.mode, bm.BatchingMode):
            f = vmap(f)
        post_vs = f(pre_spike)
      else:
        post_vs = self._syn2post_with_dense(syn_value, self.g_max, self.conn_mask)

    # updates
    self.g.value = self.integral(self.g.value, share.load('t'), bm.dt) + post_vs

    # outputs
    if self.out is not None:
      return self.out(self.g.value, post_v)
    else:
      return self.g.value


class DualExponential(SynConnNS):
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

  Parameters
  ----------
  conn: optional, ArrayType, dict of (str, ndarray), TwoEndConnector
    The synaptic connections.
  comp_method: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `sparse`.
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
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      out: Optional[SynOutNS] = None,
      stp: Optional[SynSTPNS] = None,
      comp_method: str = 'dense',
      g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
      tau_decay: Union[float, ArrayType] = 10.0,
      tau_rise: Union[float, ArrayType] = 1.,
      method: str = 'exp_auto',
      name: str = None,
      mode: bm.Mode = None,
  ):
    super(DualExponential, self).__init__(conn=conn,
                                          out=out,
                                          stp=stp,
                                          name=name,
                                          mode=mode)
    # parameters
    self.comp_method = comp_method
    self.tau_rise = is_float(tau_rise, allow_int=True, allow_none=False)
    self.tau_decay = is_float(tau_decay, allow_int=True, allow_none=False)

    # connections and weights
    self.g_max, self.conn_mask = self._init_weights(g_max, comp_method, data_if_sparse='csr')

    # function
    self.integral = odeint(JointEq(self.dg, self.dh), method=method)

    # variables
    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.h = variable_(bm.zeros, self.conn.pre_num, batch_size)
    self.g = variable_(bm.zeros, self.conn.pre_num, batch_size)
    if self.out is not None:
      self.out.reset_state(batch_size)
    if self.stp is not None:
      self.stp.reset_state(batch_size)

  def dh(self, h, t):
    return -h / self.tau_rise

  def dg(self, g, t, h):
    return -g / self.tau_decay + h

  def update(self, pre_spike, post_v=None):
    t = share.load('t')
    dt = share.load('dt')

    # update synaptic variables
    self.g.value, self.h.value = self.integral(self.g.value, self.h.value, t, dt=dt)
    self.h += pre_spike

    # post values
    syn_value = self.g.value
    if self.stp is not None:
      syn_value = self.stp(syn_value)

    if isinstance(self.conn, All2All):
      post_vs = self._syn2post_with_all2all(syn_value, self.g_max, self.conn.include_self)
    elif isinstance(self.conn, One2One):
      post_vs = self._syn2post_with_one2one(syn_value, self.g_max)
    else:
      if self.comp_method == 'sparse':
        f = lambda s: bm.sparse.csrmv(
          self.g_max,
          self.conn_mask[0],
          self.conn_mask[1],
          s,
          shape=(self.conn.pre_num, self.conn.post_num),
          transpose=True,
          method='cusparse'
        )
        if isinstance(self.mode, bm.BatchingMode):
          f = vmap(f)
        post_vs = f(syn_value)
      else:
        post_vs = self._syn2post_with_dense(syn_value, self.g_max, self.conn_mask)

    # outputs
    if self.out is not None:
      return self.out(post_vs, post_v)
    else:
      return post_vs


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
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      out: Optional[SynOutNS] = None,
      stp: Optional[SynSTPNS] = None,
      comp_method: str = 'dense',
      g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
      tau_decay: Union[float, ArrayType] = 10.0,
      method: str = 'exp_auto',

      # other parameters
      name: str = None,
      mode: bm.Mode = None,
  ):
    super().__init__(conn=conn,
                     comp_method=comp_method,
                     g_max=g_max,
                     tau_decay=tau_decay,
                     tau_rise=tau_decay,
                     method=method,
                     out=out,
                     stp=stp,
                     name=name,
                     mode=mode)
