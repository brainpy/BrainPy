from typing import Optional, Callable, Union

from brainpy import math as bm, check
from brainpy._src.delay import DelayAccess, delay_identifier, init_delay_by_return
from brainpy._src.dyn.synapses.abstract_models import Expon
from brainpy._src.dynsys import DynamicalSystem, Projection
from brainpy._src.initialize import parameter
from brainpy._src.mixin import (JointType, ParamDescInit, SupportAutoDelay, BindCondData, AlignPost, SupportSTDP)
from brainpy.types import ArrayType
from .aligns import _AlignPost, _AlignPreMg, _get_return

__all__ = [
  'STDP_Song2000',
]


class STDP_Song2000(Projection):
  r"""Synaptic output with spike-time-dependent plasticity.

  This model filters the synaptic currents according to the variables: :math:`w`.

  .. math::

     I_{syn}^+(t) = I_{syn}^-(t) * w

  where :math:`I_{syn}^-(t)` and :math:`I_{syn}^+(t)` are the synaptic currents before
  and after STDP filtering, :math:`w` measures synaptic efficacy because each time a presynaptic neuron emits a pulse,
  the conductance of the synapse will increase w.

  The dynamics of :math:`w` is governed by the following equation:

  .. math::

    \begin{aligned}
    \frac{dw}{dt} & = & -A_{post}\delta(t-t_{sp}) + A_{pre}\delta(t-t_{sp}), \\
    \frac{dA_{pre}}{dt} & = & -\frac{A_{pre}}{\tau_s}+A_1\delta(t-t_{sp}), \\
    \frac{dA_{post}}{dt} & = & -\frac{A_{post}}{\tau_t}+A_2\delta(t-t_{sp}), \\
    \tag{1}\end{aligned}

  where :math:`t_{sp}` denotes the spike time and :math:`A_1` is the increment
  of :math:`A_{pre}`, :math:`A_2` is the increment of :math:`A_{post}` produced by a spike.

  Here is an example of the usage of this class::

    import brainpy as bp
    import brainpy.math as bm

    class STDPNet(bp.DynamicalSystem):
       def __init__(self, num_pre, num_post):
         super().__init__()
         self.pre = bp.dyn.LifRef(num_pre, name='neu1')
         self.post = bp.dyn.LifRef(num_post, name='neu2')
         self.syn = bp.dyn.STDP_Song2000(
           pre=self.pre,
           delay=1.,
           comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(1, pre=self.pre.num, post=self.post.num),
                                      weight=bp.init.Uniform(max_val=0.1)),
           syn=bp.dyn.Expon.desc(self.post.varshape, tau=5.),
           out=bp.dyn.COBA.desc(E=0.),
           post=self.post,
           tau_s=16.8,
           tau_t=33.7,
           A1=0.96,
           A2=0.53,
         )

       def update(self, I_pre, I_post):
         self.syn()
         self.pre(I_pre)
         self.post(I_post)
         conductance = self.syn.refs['syn'].g
         Apre = self.syn.refs['pre_trace'].g
         Apost = self.syn.refs['post_trace'].g
         current = self.post.sum_inputs(self.post.V)
         return self.pre.spike, self.post.spike, conductance, Apre, Apost, current, self.syn.comm.weight

    duration = 300.
    I_pre = bp.inputs.section_input([0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0],
                                    [5, 15, 15, 15, 15, 15, 100, 15, 15, 15, 15, 15, duration - 255])
    I_post = bp.inputs.section_input([0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0],
                                     [10, 15, 15, 15, 15, 15, 90, 15, 15, 15, 15, 15, duration - 250])

    net = STDPNet(1, 1)
    def run(i, I_pre, I_post):
      pre_spike, post_spike, g, Apre, Apost, current, W = net.step_run(i, I_pre, I_post)
      return pre_spike, post_spike, g, Apre, Apost, current, W

    indices = bm.arange(0, duration, bm.dt)
    pre_spike, post_spike, g, Apre, Apost, current, W = bm.for_loop(run, [indices, I_pre, I_post], jit=True)

  Args:
    tau_s: float, ArrayType, Callable. The time constant of :math:`A_{pre}`.
    tau_t: float, ArrayType, Callable. The time constant of :math:`A_{post}`.
    A1: float, ArrayType, Callable. The increment of :math:`A_{pre}` produced by a spike.
    A2: float, ArrayType, Callable. The increment of :math:`A_{post}` produced by a spike.
  """

  def __init__(
      self,
      pre: JointType[DynamicalSystem, SupportAutoDelay],
      delay: Union[None, int, float],
      syn: ParamDescInit[DynamicalSystem],
      comm: DynamicalSystem,
      out: ParamDescInit[JointType[DynamicalSystem, BindCondData]],
      post: DynamicalSystem,
      # synapse parameters
      tau_s: Union[float, ArrayType, Callable] = 16.8,
      tau_t: Union[float, ArrayType, Callable] = 33.7,
      A1: Union[float, ArrayType, Callable] = 0.96,
      A2: Union[float, ArrayType, Callable] = 0.53,
      # others
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(pre, JointType[DynamicalSystem, SupportAutoDelay])
    check.is_instance(syn, ParamDescInit[DynamicalSystem])
    check.is_instance(comm, JointType[DynamicalSystem, SupportSTDP])
    check.is_instance(out, ParamDescInit[JointType[DynamicalSystem, BindCondData]])
    check.is_instance(post, DynamicalSystem)
    self.pre_num = pre.num
    self.post_num = post.num
    self.comm = comm
    self.syn = syn

    # delay initialization
    if not pre.has_aft_update(delay_identifier):
      delay_ins = init_delay_by_return(pre.return_info())
      pre.add_aft_update(delay_identifier, delay_ins)
    delay_cls = pre.get_aft_update(delay_identifier)
    delay_cls.register_entry(self.name, delay)

    if issubclass(syn.cls, AlignPost):
      # synapse and output initialization
      self._post_repr = f'{out_label} // {syn.identifier} // {out.identifier}'
      if not post.has_bef_update(self._post_repr):
        syn_cls = syn()
        out_cls = out()
        out_name = self.name if out_label is None else f'{out_label} // {self.name}'
        post.add_inp_fun(out_name, out_cls)
        post.add_bef_update(self._post_repr, _AlignPost(syn_cls, out_cls))
      # references
      self.refs = dict(pre=pre, post=post, out=out)  # invisible to ``self.nodes()``
      self.refs['delay'] = pre.get_aft_update(delay_identifier)
      self.refs['syn'] = post.get_bef_update(self._post_repr).syn  # invisible to ``self.node()``
      self.refs['out'] = post.get_bef_update(self._post_repr).out  # invisible to ``self.node()``

    else:
      # synapse initialization
      self._syn_id = f'Delay({str(delay)}) // {syn.identifier}'
      if not delay_cls.has_bef_update(self._syn_id):
        delay_access = DelayAccess(delay_cls, delay)
        syn_cls = syn()
        delay_cls.add_bef_update(self._syn_id, _AlignPreMg(delay_access, syn_cls))
      # output initialization
      out_name = self.name if out_label is None else f'{out_label} // {self.name}'
      post.add_inp_fun(out_name, out)
      # references
      self.refs = dict(pre=pre, post=post)  # invisible to `self.nodes()`
      self.refs['delay'] = delay_cls.get_bef_update(self._syn_id)
      self.refs['syn'] = delay_cls.get_bef_update(self._syn_id).syn
      self.refs['out'] = out

    # trace initialization
    self.refs['pre_trace'] = self._init_trace(pre, delay, Expon.desc(pre.num, tau=tau_s))
    self.refs['post_trace'] = self._init_trace(post, None, Expon.desc(post.num, tau=tau_t))

    # synapse parameters
    self.tau_s = parameter(tau_s, sizes=self.pre_num)
    self.tau_t = parameter(tau_t, sizes=self.post_num)
    self.A1 = parameter(A1, sizes=self.pre_num)
    self.A2 = parameter(A2, sizes=self.post_num)

  def _init_trace(
      self,
      target: DynamicalSystem,
      delay: Union[None, int, float],
      syn: ParamDescInit[DynamicalSystem],
  ):
    """Calculate the trace of the target."""
    check.is_instance(target, DynamicalSystem)
    check.is_instance(syn, ParamDescInit[DynamicalSystem])

    # delay initialization
    if not target.has_aft_update(delay_identifier):
      delay_ins = init_delay_by_return(target.return_info())
      target.add_aft_update(delay_identifier, delay_ins)
    delay_cls = target.get_aft_update(delay_identifier)
    delay_cls.register_entry(target.name, delay)

    # synapse initialization
    _syn_id = f'Delay({str(delay)}) // {syn.identifier}'
    if not delay_cls.has_bef_update(_syn_id):
      # delay
      delay_access = DelayAccess(delay_cls, delay)
      # synapse
      syn_cls = syn()
      # add to "after_updates"
      delay_cls.add_bef_update(_syn_id, _AlignPreMg(delay_access, syn_cls))

    return delay_cls.get_bef_update(_syn_id).syn

  def update(self):
    # pre spikes, and pre-synaptic variables
    if issubclass(self.syn.cls, AlignPost):
      pre_spike = self.refs['delay'].at(self.name)
      x = pre_spike
    else:
      pre_spike = self.refs['delay'].access()
      x = _get_return(self.refs['syn'].return_info())

    # post spikes
    post_spike = self.refs['post'].spike

    # weight updates
    Apre = self.refs['pre_trace'].g
    Apost = self.refs['post_trace'].g
    delta_w = - bm.outer(pre_spike, Apost * self.A2) + bm.outer(Apre * self.A1, post_spike)
    self.comm.update_STDP(delta_w)

    # currents
    current = self.comm(x)
    if issubclass(self.syn.cls, AlignPost):
      self.refs['syn'].add_current(current)  # synapse post current
    else:
      self.refs['out'].bind_cond(current)  # align pre
    return current
