from typing import Optional, Callable, Union

from brainpy import math as bm
from brainpy._src.dynsys import DynamicalSystemNS, DynamicalSystem
from brainpy._src.pnn.delay import DataDelay, Delay
from brainpy._src.pnn.neurons.base import PNeuGroup
from brainpy._src.pnn.utils import DelayedInit
from .syn_output import PSynOut

__all__ = [
  'ProjectionAlignPre',
  'ProjectionAlignPost',
]


class _AlignPre(DynamicalSystemNS):
  def __init__(self, syn, delay=None):
    super().__init__()
    self.syn = syn
    self.delay = delay

  def update(self, x):
    if self.delay is None:
      return x >> self.syn
    else:
      return x >> self.syn >> self.delay


class ProjectionAlignPre(DynamicalSystemNS):
  """Synaptic projection which defines the synaptic computation with the dimension of presynaptic neuron group.

  Args:
    pre: The pre-synaptic neuron group.
    syn: The synaptic dynamics.
    delay: The synaptic delay.
    comm: The synaptic communication.
    out: The synaptic output.
    post: The post-synaptic neuron group.
    name: str. The projection name.
    mode: Mode. The computing mode.
  """
  def __init__(
      self,
      pre: PNeuGroup,
      syn: DelayedInit[DynamicalSystem],
      delay: Union[None, int, float, DelayedInit[Delay]],
      comm: Callable,
      out: PSynOut,
      post: PNeuGroup,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    assert isinstance(pre, PNeuGroup)
    assert isinstance(post, PNeuGroup)
    assert callable(comm)
    assert isinstance(out, PSynOut)
    assert isinstance(syn, DelayedInit)
    self.pre = pre
    self.post = post
    self.comm = comm

    # synapse and delay initialization
    self._syn_id = syn._identifier
    delay_time = None
    if self._syn_id not in pre.post_updates:
      syn_cls = syn()
      if delay is None:
        delay_cls = DataDelay(pre.varshape,
                              axis_names=pre.axis_names,
                              mode=pre.mode)
      elif isinstance(delay, (int, float)):
        delay_time = delay
        delay_cls = DataDelay(pre.varshape,
                              axis_names=pre.axis_names,
                              mode=pre.mode)
      elif isinstance(delay, DelayedInit):
        delay_time = delay.kwargs.get('time', None)
        delay_cls = delay()
      else:
        raise TypeError
      pre.post_updates[self._syn_id] = _AlignPre(syn_cls, delay_cls)
    delay_cls = pre.post_updates[self._syn_id].delay
    if delay_cls is not None:
      delay_cls.register_entry(self.name, delay_time)

    # output initialization
    post.cur_outputs[self.name] = out

  def update(self):
    current = self.comm(self.pre.post_updates[self._syn_id].delay.at(self.name))
    self.post.cur_outputs[self.name].bind_cond(current)
    return current


class _AlignPost(DynamicalSystemNS):
  def __init__(self, syn, out):
    super().__init__()
    self.syn = syn
    self.out = out

  def update(self, *args, **kwargs):
    self.out.bind_cond(self.syn(*args, **kwargs))


class ProjectionAlignPost(DynamicalSystemNS):
  """Synaptic projection which defines the synaptic computation with the dimension of postsynaptic neuron group.

  Args:
    pre: The pre-synaptic neuron group.
    delay: The synaptic delay.
    comm: The synaptic communication.
    syn: The synaptic dynamics.
    out: The synaptic output.
    post: The post-synaptic neuron group.
    name: str. The projection name.
    mode:  Mode. The computing mode.
  """
  def __init__(
      self,
      pre: PNeuGroup,
      delay: Union[None, int, float, DelayedInit[Delay]],
      comm: Callable,
      syn: DelayedInit[DynamicalSystem],
      out: DelayedInit[PSynOut],
      post: PNeuGroup,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    assert isinstance(pre, PNeuGroup)
    assert isinstance(post, PNeuGroup)
    assert isinstance(syn, DelayedInit)
    assert isinstance(out, DelayedInit)
    assert callable(comm)
    self.pre = pre
    self.post = post
    self.comm = comm

    # delay initialization
    self._delay_repr = '_*_align_pre_spk_delay_*_'
    delay_time = None
    if self._delay_repr not in self.pre.post_updates:
      if delay is None:
        delay_cls = DataDelay(pre.varshape,
                              axis_names=pre.axis_names,
                              mode=pre.mode)
      elif isinstance(delay, (int, float)):
        delay_time = delay
        delay_cls = DataDelay(pre.varshape,
                              axis_names=pre.axis_names,
                              mode=pre.mode)
      elif isinstance(delay, DelayedInit):
        delay_time = delay.kwargs.get('time', None)
        delay_cls = delay()
      else:
        raise TypeError
      self.pre.post_updates[self._delay_repr] = delay_cls
    delay_cls = pre.post_updates[self._delay_repr]
    if delay_cls is not None:
      delay_cls.register_entry(self.name, delay_time)

    # synapse and output initialization
    self._post_repr = f'{syn._identifier} // {out._identifier}'
    if self._post_repr not in self.post.pre_updates:
      syn_cls = syn()
      out_cls = out()
      self.post.cur_outputs[self.name] = out_cls
      self.post.pre_updates[self._post_repr] = _AlignPost(syn_cls, out_cls)

  def update(self):
    current = self.comm(self.pre.post_updates[self._delay_repr].at(self.name))
    self.post.pre_updates[self._post_repr].syn.add_current(current)  # synapse post current
    return current

