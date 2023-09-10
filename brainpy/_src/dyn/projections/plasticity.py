from typing import Optional, Callable, Union

from brainpy.types import ArrayType
from brainpy import math as bm, check
from brainpy._src.delay import Delay, DelayAccess, delay_identifier, init_delay_by_return
from brainpy._src.dynsys import DynamicalSystem, Projection
from brainpy._src.mixin import (JointType, ParamDescInit, ReturnInfo,
                                AutoDelaySupp, BindCondData, AlignPost)
from brainpy._src.initialize import parameter
from brainpy._src.dyn.synapses.abstract_models import Expon

__all__ = [
  'STDP_Song2000'
]

class _AlignPre(DynamicalSystem):
  def __init__(self, syn, delay=None):
    super().__init__()
    self.syn = syn
    self.delay = delay

  def update(self, x):
    if self.delay is None:
      return x >> self.syn
    else:
      return x >> self.syn >> self.delay


class _AlignPost(DynamicalSystem):
  def __init__(self,
               syn: Callable,
               out: JointType[DynamicalSystem, BindCondData]):
    super().__init__()
    self.syn = syn
    self.out = out

  def update(self, *args, **kwargs):
    self.out.bind_cond(self.syn(*args, **kwargs))


class _AlignPreMg(DynamicalSystem):
  def __init__(self, access, syn):
    super().__init__()
    self.access = access
    self.syn = syn

  def update(self, *args, **kwargs):
    return self.syn(self.access())


def _get_return(return_info):
  if isinstance(return_info, bm.Variable):
    return return_info.value
  elif isinstance(return_info, ReturnInfo):
    return return_info.get_data()
  else:
    raise NotImplementedError


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

  Example:




  Args:
    tau_s: float, ArrayType, Callable. The time constant of :math:`A_{pre}`.
    tau_t: float, ArrayType, Callable. The time constant of :math:`A_{post}`.
    A1: float, ArrayType, Callable. The increment of :math:`A_{pre}` produced by a spike.
    A2: float, ArrayType, Callable. The increment of :math:`A_{post}` produced by a spike.
    %s
  """
  def __init__(
      self,
      pre: JointType[DynamicalSystem, AutoDelaySupp],
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
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(pre, JointType[DynamicalSystem, AutoDelaySupp])
    check.is_instance(syn, ParamDescInit[DynamicalSystem])
    # TODO: check
    check.is_instance(comm, JointType[DynamicalSystem, SupportPlasticity])
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
        if out_label is None:
          out_name = self.name
        else:
          out_name = f'{out_label} // {self.name}'
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
        # delay
        delay_access = DelayAccess(delay_cls, delay)
        # synapse
        syn_cls = syn()
        # add to "after_updates"
        delay_cls.add_bef_update(self._syn_id, _AlignPreMg(delay_access, syn_cls))

      # output initialization
      if out_label is None:
        out_name = self.name
      else:
        out_name = f'{out_label} // {self.name}'
      post.add_inp_fun(out_name, out)

      # references
      self.refs = dict(pre=pre, post=post)  # invisible to `self.nodes()`
      self.refs['delay'] = delay_cls.get_bef_update(self._syn_id)
      self.refs['syn'] = delay_cls.get_bef_update(self._syn_id).syn
      self.refs['out'] = out

    # TODO: Expon and other can be parameters of the class
    self.refs['pre_trace'] = self.calculate_trace(pre, delay, Expon.desc(pre.num, tau=tau_s))
    self.refs['post_trace'] = self.calculate_trace(post, None, Expon.desc(post.num, tau=tau_t))
    # parameters
    self.tau_s = parameter(tau_s, sizes=self.pre_num)
    self.tau_t = parameter(tau_t, sizes=self.post_num)
    self.A1 = parameter(A1, sizes=self.pre_num)
    self.A2 = parameter(A2, sizes=self.post_num)

  def calculate_trace(
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
    delay_cls.register_entry(self.name, delay)

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
    if issubclass(self.syn.cls, AlignPost):
      pre_spike = self.refs['delay'].at(self.name)
      x = pre_spike
    else:
      pre_spike = self.refs['delay'].access()
      x = _get_return(self.refs['syn'].return_info())

    post_spike = self.refs['post'].spike

    Apre = self.refs['pre_trace'].g
    Apost = self.refs['post_trace'].g
    delta_w = - bm.outer(pre_spike, Apost * self.A2) + bm.outer(Apre * self.A1, post_spike)
    self.comm.update_weights(delta_w)

    current = self.comm(x)
    if issubclass(self.syn.cls, AlignPost):
      self.refs['syn'].add_current(current)  # synapse post current
    else:
      self.refs['out'].bind_cond(current)
    return current
