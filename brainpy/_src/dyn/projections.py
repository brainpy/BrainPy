from typing import Optional, Callable, Union

from brainpy import math as bm
from brainpy._src.delay import Delay, VariableDelay, DataDelay
from brainpy._src.dyn.base import NeuDyn, SynOut
from brainpy._src.dynsys import DynamicalSystemNS, DynamicalSystem
from brainpy._src.mixin import DelayedInit, ReturnInfo, ProjAutoDelay

__all__ = [
  'SynProj',
  'ProjAlignPre',
  'ProjAlignPost',
]


class SynProj(DynamicalSystemNS):
  """Synaptic projection."""
  pass


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


class _AlignPost(DynamicalSystemNS):
  def __init__(self, syn, out):
    super().__init__()
    self.syn = syn
    self.out = out

  def update(self, *args, **kwargs):
    self.out.bind_cond(self.syn(*args, **kwargs))


def _init_delay(info: Union[bm.Variable, ReturnInfo]) -> Delay:
  if isinstance(info, bm.Variable):
    return VariableDelay(info)
  elif isinstance(info, ReturnInfo):
    if isinstance(info.batch_or_mode, int):
      size = (info.batch_or_mode,) + tuple(info.size)
      batch_axis = 0
    elif isinstance(info.batch_or_mode, bm.NonBatchingMode):
      size = tuple(info.size)
      batch_axis = None
    elif isinstance(info.batch_or_mode, bm.BatchingMode):
      size = (info.batch_or_mode.batch_size,) + tuple(info.size)
      batch_axis = 0
    else:
      size = tuple(info.size)
      batch_axis = None
    target = bm.Variable(info.init(size),
                         batch_axis=batch_axis,
                         axis_names=info.axis_names)
    return DataDelay(target, target_init=info.init)
  else:
    raise TypeError


class ProjAlignPre(SynProj):
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
      pre: NeuDyn,
      syn: DelayedInit[ProjAutoDelay],
      delay: Union[None, int, float],
      comm: Callable,
      out: SynOut,
      post: NeuDyn,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    assert isinstance(pre, NeuDyn)
    assert isinstance(post, NeuDyn)
    assert callable(comm)
    assert isinstance(out, SynOut)
    assert isinstance(syn, DelayedInit) and issubclass(syn.cls, ProjAutoDelay)
    self.pre = pre
    self.post = post
    self.comm = comm

    # synapse and delay initialization
    self._syn_id = syn._identifier
    if self._syn_id not in pre.after_updates:
      syn_cls: ProjAutoDelay = syn()
      delay_cls = _init_delay(syn_cls.return_info())
      pre.after_updates[self._syn_id] = _AlignPre(syn_cls, delay_cls)
    delay_cls: Delay = pre.after_updates[self._syn_id].delay
    delay_cls.register_entry(self.name, delay)

    # output initialization
    post.cur_inputs[self.name] = out

  def update(self):
    current = self.comm(self.pre.after_updates[self._syn_id].delay.at(self.name))
    self.post.cur_inputs[self.name].bind_cond(current)
    return current


class ProjAlignPost(SynProj):
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
      pre: ProjAutoDelay,
      delay: Union[None, int, float],
      comm: Callable,
      syn: DelayedInit[DynamicalSystem],
      out: DelayedInit[SynOut],
      post: NeuDyn,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    assert isinstance(pre, NeuDyn) and isinstance(pre, ProjAutoDelay)
    assert isinstance(post, NeuDyn)
    assert isinstance(syn, DelayedInit) and issubclass(syn.cls, DynamicalSystem)
    assert isinstance(out, DelayedInit) and issubclass(out.cls, SynOut)
    assert callable(comm)
    self.pre = pre
    self.post = post
    self.comm = comm

    # delay initialization
    self._delay_repr = '_*_align_pre_spk_delay_*_'
    if self._delay_repr not in self.pre.after_updates:
      delay_cls = _init_delay(pre.return_info())
      self.pre.after_updates[self._delay_repr] = delay_cls
    delay_cls: Delay = pre.after_updates[self._delay_repr]
    delay_cls.register_entry(self.name, delay)

    # synapse and output initialization
    self._post_repr = f'{syn._identifier} // {out._identifier}'
    if self._post_repr not in self.post.before_updates:
      syn_cls = syn()
      out_cls = out()
      self.post.cur_inputs[self.name] = out_cls
      self.post.before_updates[self._post_repr] = _AlignPost(syn_cls, out_cls)

  def update(self):
    current = self.comm(self.pre.after_updates[self._delay_repr].at(self.name))
    self.post.before_updates[self._post_repr].syn.add_current(current)  # synapse post current
    return current
