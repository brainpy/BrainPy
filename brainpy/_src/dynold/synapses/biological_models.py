# -*- coding: utf-8 -*-

from typing import Union, Dict, Callable, Optional

import brainpy.math as bm
from brainpy._src.connect import TwoEndConnector
from brainpy._src.dyn import synapses
from brainpy._src.dynold.synapses import _SynSTP, _SynOut, _TwoEndConnAlignPre
from brainpy._src.dynold.synouts import COBA, MgBlock
from brainpy._src.dyn.base import NeuDyn
from brainpy.types import ArrayType

__all__ = [
  'AMPA',
  'GABAa',
  'BioNMDA',
]


class AMPA(_TwoEndConnAlignPre):
  def __init__(
      self,
      pre: NeuDyn,
      post: NeuDyn,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      output: _SynOut = COBA(E=0.),
      stp: Optional[_SynSTP] = None,
      comp_method: str = 'dense',
      g_max: Union[float, ArrayType, Callable] = 0.42,
      delay_step: Union[int, ArrayType, Callable] = None,
      alpha: float = 0.98,
      beta: float = 0.18,
      T: float = 0.5,
      T_duration: float = 0.5,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
      stop_spike_gradient: bool = False,
  ):
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

    # AMPA
    syn = synapses.AMPA(pre.size, pre.keep_size, mode=mode, alpha=alpha, beta=beta,
                        T=T, T_dur=T_duration, method=method)

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
    self.spike_arrival_time = syn.spike_arrival_time

  def update(self, pre_spike=None):
    return super().update(pre_spike, stop_spike_gradient=self.stop_spike_gradient)


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
  pre: NeuDyn
    The pre-synaptic neuron group.
  post: NeuDyn
    The post-synaptic neuron group.
  conn: optional, ArrayType, dict of (str, ndarray), TwoEndConnector
    The synaptic connections.
  comp_method: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `dense`.
  delay_step: int, ArrayType, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  g_max: float, ArrayType, Callable
    The synaptic strength (the maximum conductance). Default is 1.
  alpha: float, ArrayType
    Binding constant. Default 0.062
  beta: float, ArrayType
    Unbinding constant. Default 3.57
  T: float, ArrayType
    Transmitter concentration when synapse is triggered by
    a pre-synaptic spike.. Default 1 [mM].
  T_duration: float, ArrayType
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
      pre: NeuDyn,
      post: NeuDyn,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      output: _SynOut = COBA(E=-80.),
      stp: Optional[_SynSTP] = None,
      comp_method: str = 'dense',
      g_max: Union[float, ArrayType, Callable] = 0.04,
      delay_step: Union[int, ArrayType, Callable] = None,
      alpha: Union[float, ArrayType] = 0.53,
      beta: Union[float, ArrayType] = 0.18,
      T: Union[float, ArrayType] = 1.,
      T_duration: Union[float, ArrayType] = 1.,
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


class BioNMDA(_TwoEndConnAlignPre):
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
    >>> from brainpy import neurons, synapses
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = neurons.HH(1)
    >>> neu2 = neurons.HH(1)
    >>> syn1 = synapses.BioNMDA(neu1, neu2, bp.connect.All2All())
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
  delay_step: int, ArrayType, Callable
    The delay length. It should be the value of :math:`\mathrm{delay\_time / dt}`.
  g_max: float, ArrayType, Callable
    The synaptic strength (the maximum conductance). Default is 1.
  alpha1: float, ArrayType
    The conversion rate of g from inactive to active. Default 2 ms^-1.
  beta1: float, ArrayType
    The conversion rate of g from active to inactive. Default 0.01 ms^-1.
  alpha2: float, ArrayType
    The conversion rate of x from inactive to active. Default 1 ms^-1.
  beta2: float, ArrayType
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
      pre: NeuDyn,
      post: NeuDyn,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      output: _SynOut = MgBlock(E=0.),
      stp: Optional[_SynSTP] = None,
      comp_method: str = 'dense',
      g_max: Union[float, ArrayType, Callable] = 0.15,
      delay_step: Union[int, ArrayType, Callable] = None,
      alpha1: Union[float, ArrayType] = 2.,
      beta1: Union[float, ArrayType] = 0.01,
      alpha2: Union[float, ArrayType] = 1.,
      beta2: Union[float, ArrayType] = 0.5,
      T_0: Union[float, ArrayType] = 1.,
      T_dur: Union[float, ArrayType] = 0.5,
      method: str = 'exp_auto',
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
      stop_spike_gradient: bool = False,
  ):

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

    syn = synapses.BioNMDA(pre.size,
                           pre.keep_size,
                           mode=mode,
                           alpha1=alpha1,
                           beta1=beta1,
                           alpha2=alpha2,
                           beta2=beta2,
                           T=T_0,
                           T_dur=T_dur,
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
    self.spike_arrival_time = syn.spike_arrival_time

  def update(self, pre_spike=None):
    return super().update(pre_spike, stop_spike_gradient=self.stop_spike_gradient)
