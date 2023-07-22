from typing import Optional, Callable, Union

import jax

from brainpy import math as bm, check
from brainpy._src.delay import Delay, VarDelay, DataDelay, DelayAccess
from brainpy._src.dynsys import DynamicalSystem, Projection, Dynamic
from brainpy._src.mixin import JointType, ParamDescInit, ReturnInfo, AutoDelaySupp, BindCondData, AlignPost

__all__ = [
  'VanillaProj',
  'ProjAlignPostMg1', 'ProjAlignPostMg2',
  'ProjAlignPost1', 'ProjAlignPost2',
  'ProjAlignPreMg1', 'ProjAlignPreMg2',
]

_pre_delay_repr = '_*_align_pre_spk_delay_*_'


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

  def update(self):
    return self.syn(self.access())


def _init_delay(info: Union[bm.Variable, ReturnInfo]) -> Delay:
  if isinstance(info, bm.Variable):
    return VarDelay(info)
  elif isinstance(info, ReturnInfo):
    if isinstance(info.batch_or_mode, int):
      shape = (info.batch_or_mode,) + tuple(info.size)
      batch_axis = 0
    elif isinstance(info.batch_or_mode, bm.NonBatchingMode):
      shape = tuple(info.size)
      batch_axis = None
    elif isinstance(info.batch_or_mode, bm.BatchingMode):
      shape = (info.batch_or_mode.batch_size,) + tuple(info.size)
      batch_axis = 0
    else:
      shape = tuple(info.size)
      batch_axis = None
    if isinstance(info.data, Callable):
      init = info.data(shape)
    elif isinstance(info.data, (bm.Array, jax.Array)):
      init = info.data
    else:
      raise TypeError
    assert init.shape == shape
    if info.axis_names is not None:
      assert init.ndim == len(info.axis_names)
    target = bm.Variable(init, batch_axis=batch_axis, axis_names=info.axis_names)
    return DataDelay(target, data_init=info.data)
  else:
    raise TypeError


def _get_return(return_info):
  if isinstance(return_info, bm.Variable):
    return return_info.value
  elif isinstance(return_info, ReturnInfo):
    return return_info.get_data()
  else:
    raise NotImplementedError


class VanillaProj(Projection):
  """Synaptic projection which defines the synaptic computation with the dimension of pre-synaptic neuron group.

  **Code Examples**

  To simulate an E/I balanced network model:

  .. code-block::

      class EINet(bp.DynSysGroup):
        def __init__(self):
          super().__init__()
          self.N = bp.dyn.LifRef(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
          self.syn1 = bp.dyn.Expon(size=3200, tau=5.)
          self.syn2 = bp.dyn.Expon(size=800, tau=10.)
          self.E = bp.dyn.VanillaProj(comm=bp.dnn.EventJitFPHomoLinear(3200, 4000, prob=0.02, weight=0.6),
                                      out=bp.dyn.COBA(E=0.),
                                      post=self.N)
          self.I = bp.dyn.VanillaProj(comm=bp.dnn.EventJitFPHomoLinear(800, 4000, prob=0.02, weight=6.7),
                                      out=bp.dyn.COBA(E=-80.),
                                      post=self.N)

        def update(self, input):
          spk = self.delay.at('I')
          self.E(self.syn1(spk[:3200]))
          self.I(self.syn2(spk[3200:]))
          self.delay(self.N(input))
          return self.N.spike.value

      model = EINet()
      indices = bm.arange(1000)
      spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
      bp.visualize.raster_plot(indices, spks, show=True)


  Args:
    comm: The synaptic communication.
    out: The synaptic output.
    post: The post-synaptic neuron group.
    name: str. The projection name.
    mode: Mode. The computing mode.
  """

  def __init__(
      self,
      comm: DynamicalSystem,
      out: JointType[DynamicalSystem, BindCondData],
      post: Dynamic,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(out, JointType[DynamicalSystem, BindCondData])
    check.is_instance(post, Dynamic)
    self.post = post
    self.comm = comm

    # output initialization
    post.cur_inputs[self.name] = out

  def update(self, x):
    current = self.comm(x)
    self.post.cur_inputs[self.name].bind_cond(current)
    return current


class ProjAlignPostMg1(Projection):
  r"""Synaptic projection which defines the synaptic computation with the dimension of postsynaptic neuron group.

  **Code Examples**

  To define an E/I balanced network model.
  
  .. code-block:: python

    import brainpy as bp
    import brainpy.math as bm

    class EINet(bp.DynSysGroup):
      def __init__(self):
        super().__init__()
        self.N = bp.dyn.LifRef(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                               V_initializer=bp.init.Normal(-55., 2.))
        self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
        self.E = bp.dyn.ProjAlignPostMg1(comm=bp.dnn.EventJitFPHomoLinear(3200, 4000, prob=0.02, weight=0.6),
                                         syn=bp.dyn.Expon.desc(size=4000, tau=5.),
                                         out=bp.dyn.COBA.desc(E=0.),
                                         post=self.N)
        self.I = bp.dyn.ProjAlignPostMg1(comm=bp.dnn.EventJitFPHomoLinear(800, 4000, prob=0.02, weight=6.7),
                                         syn=bp.dyn.Expon.desc(size=4000, tau=10.),
                                         out=bp.dyn.COBA.desc(E=-80.),
                                         post=self.N)

      def update(self, input):
        spk = self.delay.at('I')
        self.E(spk[:3200])
        self.I(spk[3200:])
        self.delay(self.N(input))
        return self.N.spike.value

    model = EINet()
    indices = bm.arange(1000)
    spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
    bp.visualize.raster_plot(indices, spks, show=True)

  Args:
    comm: The synaptic communication.
    syn: The synaptic dynamics.
    out: The synaptic output.
    post: The post-synaptic neuron group.
    name: str. The projection name.
    mode:  Mode. The computing mode.
  """

  def __init__(
      self,
      comm: DynamicalSystem,
      syn: ParamDescInit[JointType[DynamicalSystem, AlignPost]],
      out: ParamDescInit[JointType[DynamicalSystem, BindCondData]],
      post: Dynamic,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(syn, ParamDescInit[JointType[DynamicalSystem, AlignPost]])
    check.is_instance(out, ParamDescInit[JointType[DynamicalSystem, BindCondData]])
    check.is_instance(post, Dynamic)
    self.post = post
    self.comm = comm

    # synapse and output initialization
    self._post_repr = f'{syn._identifier} // {out._identifier}'
    if self._post_repr not in self.post.before_updates:
      syn_cls = syn()
      out_cls = out()
      self.post.cur_inputs[self.name] = out_cls
      self.post.before_updates[self._post_repr] = _AlignPost(syn_cls, out_cls)

  def update(self, x):
    current = self.comm(x)
    syn: _AlignPost = self.post.before_updates[self._post_repr].syn
    syn.add_current(current)  # synapse post current
    return current


class ProjAlignPostMg2(Projection):
  """Synaptic projection which defines the synaptic computation with the dimension of postsynaptic neuron group.

  **Code Examples**

  To define an E/I balanced network model.

  .. code-block:: python

      import brainpy as bp
      import brainpy.math as bm

      class EINet(bp.DynSysGroup):
        def __init__(self):
          super().__init__()
          ne, ni = 3200, 800
          self.E = bp.dyn.LifRef(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.I = bp.dyn.LifRef(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.E2E = bp.dyn.ProjAlignPostMg2(pre=self.E,
                                             delay=0.1,
                                             comm=bp.dnn.EventJitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                             syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                             out=bp.dyn.COBA.desc(E=0.),
                                             post=self.E)
          self.E2I = bp.dyn.ProjAlignPostMg2(pre=self.E,
                                             delay=0.1,
                                             comm=bp.dnn.EventJitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                             syn=bp.dyn.Expon.desc(size=ni, tau=5.),
                                             out=bp.dyn.COBA.desc(E=0.),
                                             post=self.I)
          self.I2E = bp.dyn.ProjAlignPostMg2(pre=self.I,
                                             delay=0.1,
                                             comm=bp.dnn.EventJitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                             syn=bp.dyn.Expon.desc(size=ne, tau=10.),
                                             out=bp.dyn.COBA.desc(E=-80.),
                                             post=self.E)
          self.I2I = bp.dyn.ProjAlignPostMg2(pre=self.I,
                                             delay=0.1,
                                             comm=bp.dnn.EventJitFPHomoLinear(ni, ni, prob=0.02, weight=6.7),
                                             syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                             out=bp.dyn.COBA.desc(E=-80.),
                                             post=self.I)

        def update(self, inp):
          self.E2E()
          self.E2I()
          self.I2E()
          self.I2I()
          self.E(inp)
          self.I(inp)
          return self.E.spike

      model = EINet()
      indices = bm.arange(1000)
      spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
      bp.visualize.raster_plot(indices, spks, show=True)

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
      pre: JointType[DynamicalSystem, AutoDelaySupp],
      delay: Union[None, int, float],
      comm: DynamicalSystem,
      syn: ParamDescInit[JointType[DynamicalSystem, AlignPost]],
      out: ParamDescInit[JointType[DynamicalSystem, BindCondData]],
      post: Dynamic,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(pre, JointType[DynamicalSystem, AutoDelaySupp])
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(syn, ParamDescInit[JointType[DynamicalSystem, AlignPost]])
    check.is_instance(out, ParamDescInit[JointType[DynamicalSystem, BindCondData]])
    check.is_instance(post, Dynamic)
    self.pre = pre
    self.post = post
    self.comm = comm

    # delay initialization
    if _pre_delay_repr not in self.pre.after_updates:
      # pre should support "ProjAutoDelay"
      delay_cls = _init_delay(pre.return_info())
      # add to "after_updates"
      self.pre.after_updates[_pre_delay_repr] = delay_cls
    delay_cls: Delay = pre.after_updates[_pre_delay_repr]
    delay_cls.register_entry(self.name, delay)

    # synapse and output initialization
    self._post_repr = f'{syn._identifier} // {out._identifier}'
    if self._post_repr not in self.post.before_updates:
      syn_cls = syn()
      out_cls = out()
      self.post.cur_inputs[self.name] = out_cls
      self.post.before_updates[self._post_repr] = _AlignPost(syn_cls, out_cls)

  def update(self):
    x = self.pre.after_updates[_pre_delay_repr].at(self.name)
    current = self.comm(x)
    self.post.before_updates[self._post_repr].syn.add_current(current)  # synapse post current
    return current


class ProjAlignPost1(Projection):
  """Synaptic projection which defines the synaptic computation with the dimension of postsynaptic neuron group.

  To simulate an E/I balanced network:

  .. code-block::

      class EINet(bp.DynSysGroup):
        def __init__(self):
          super().__init__()
          self.N = bp.dyn.LifRef(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
          self.E = bp.dyn.ProjAlignPost1(comm=bp.dnn.EventJitFPHomoLinear(3200, 4000, prob=0.02, weight=0.6),
                                         syn=bp.dyn.Expon(size=4000, tau=5.),
                                         out=bp.dyn.COBA(E=0.),
                                         post=self.N)
          self.I = bp.dyn.ProjAlignPost1(comm=bp.dnn.EventJitFPHomoLinear(800, 4000, prob=0.02, weight=6.7),
                                         syn=bp.dyn.Expon(size=4000, tau=10.),
                                         out=bp.dyn.COBA(E=-80.),
                                         post=self.N)

        def update(self, input):
          spk = self.delay.at('I')
          self.E(spk[:3200])
          self.I(spk[3200:])
          self.delay(self.N(input))
          return self.N.spike.value

      model = EINet()
      indices = bm.arange(1000)
      spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
      bp.visualize.raster_plot(indices, spks, show=True)


  Args:
    comm: The synaptic communication.
    syn: The synaptic dynamics.
    out: The synaptic output.
    post: The post-synaptic neuron group.
    name: str. The projection name.
    mode:  Mode. The computing mode.
  """

  def __init__(
      self,
      comm: DynamicalSystem,
      syn: JointType[DynamicalSystem, AlignPost],
      out: JointType[DynamicalSystem, BindCondData],
      post: Dynamic,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(syn, JointType[DynamicalSystem, AlignPost])
    check.is_instance(out, JointType[DynamicalSystem, BindCondData])
    check.is_instance(post, Dynamic)
    self.post = post
    self.comm = comm

    # synapse and output initialization
    self.post.cur_inputs[self.name] = out
    self.post.before_updates[self.name] = _AlignPost(syn, out)

  def update(self, x):
    current = self.comm(x)
    syn: _AlignPost = self.post.before_updates[self.name].syn
    syn.add_current(current)  # synapse post current
    return current


class ProjAlignPost2(Projection):
  """Synaptic projection which defines the synaptic computation with the dimension of postsynaptic neuron group.

  To simulate and define an E/I balanced network model:

  .. code-block:: python

      class EINet(bp.DynSysGroup):
        def __init__(self):
          super().__init__()
          ne, ni = 3200, 800
          self.E = bp.dyn.LifRef(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.I = bp.dyn.LifRef(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.E2E = bp.dyn.ProjAlignPost2(pre=self.E,
                                           delay=0.1,
                                           comm=bp.dnn.EventJitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                           syn=bp.dyn.Expon(size=ne, tau=5.),
                                           out=bp.dyn.COBA(E=0.),
                                           post=self.E)
          self.E2I = bp.dyn.ProjAlignPost2(pre=self.E,
                                           delay=0.1,
                                           comm=bp.dnn.EventJitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                           syn=bp.dyn.Expon(size=ni, tau=5.),
                                           out=bp.dyn.COBA(E=0.),
                                           post=self.I)
          self.I2E = bp.dyn.ProjAlignPost2(pre=self.I,
                                           delay=0.1,
                                           comm=bp.dnn.EventJitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                           syn=bp.dyn.Expon(size=ne, tau=10.),
                                           out=bp.dyn.COBA(E=-80.),
                                           post=self.E)
          self.I2I = bp.dyn.ProjAlignPost2(pre=self.I,
                                           delay=0.1,
                                           comm=bp.dnn.EventJitFPHomoLinear(ni, ni, prob=0.02, weight=6.7),
                                           syn=bp.dyn.Expon(size=ni, tau=10.),
                                           out=bp.dyn.COBA(E=-80.),
                                           post=self.I)

        def update(self, inp):
          self.E2E()
          self.E2I()
          self.I2E()
          self.I2I()
          self.E(inp)
          self.I(inp)
          return self.E.spike

      model = EINet()
      indices = bm.arange(1000)
      spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
      bp.visualize.raster_plot(indices, spks, show=True)


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
      pre: JointType[DynamicalSystem, AutoDelaySupp],
      delay: Union[None, int, float],
      comm: DynamicalSystem,
      syn: JointType[DynamicalSystem, AlignPost],
      out: JointType[DynamicalSystem, BindCondData],
      post: Dynamic,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(pre, JointType[DynamicalSystem, AutoDelaySupp])
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(syn, JointType[DynamicalSystem, AlignPost])
    check.is_instance(out, JointType[DynamicalSystem, BindCondData])
    check.is_instance(post, Dynamic)
    self.pre = pre
    self.post = post
    self.comm = comm

    # delay initialization
    if _pre_delay_repr not in self.pre.after_updates:
      # pre should support "ProjAutoDelay"
      delay_cls = _init_delay(pre.return_info())
      # add to "after_updates"
      self.pre.after_updates[_pre_delay_repr] = delay_cls
    delay_cls: Delay = pre.after_updates[_pre_delay_repr]
    delay_cls.register_entry(self.name, delay)

    # synapse and output initialization
    self.post.cur_inputs[self.name] = out
    self.post.before_updates[self.name] = _AlignPost(syn, out)

  def update(self):
    x = self.pre.after_updates[_pre_delay_repr].at(self.name)
    current = self.comm(x)
    self.post.before_updates[self.name].syn.add_current(current)  # synapse post current
    return current


class ProjAlignPreMg1(Projection):
  """Synaptic projection which defines the synaptic computation with the dimension of presynaptic neuron group.

  To simulate an E/I balanced network model:

  .. code-block:: python

      class EINet(bp.DynSysGroup):
        def __init__(self):
          super().__init__()
          ne, ni = 3200, 800
          self.E = bp.dyn.LifRef(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.I = bp.dyn.LifRef(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.E2E = bp.dyn.ProjAlignPreMg1(pre=self.E,
                                            syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                            delay=0.1,
                                            comm=bp.dnn.JitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                            out=bp.dyn.COBA(E=0.),
                                            post=self.E)
          self.E2I = bp.dyn.ProjAlignPreMg1(pre=self.E,
                                            syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                            delay=0.1,
                                            comm=bp.dnn.JitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                            out=bp.dyn.COBA(E=0.),
                                            post=self.I)
          self.I2E = bp.dyn.ProjAlignPreMg1(pre=self.I,
                                            syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                            delay=0.1,
                                            comm=bp.dnn.JitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                            out=bp.dyn.COBA(E=-80.),
                                            post=self.E)
          self.I2I = bp.dyn.ProjAlignPreMg1(pre=self.I,
                                            syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                            delay=0.1,
                                            comm=bp.dnn.JitFPHomoLinear(ni, ni, prob=0.02, weight=6.7),
                                            out=bp.dyn.COBA(E=-80.),
                                            post=self.I)

        def update(self, inp):
          self.E2E()
          self.E2I()
          self.I2E()
          self.I2I()
          self.E(inp)
          self.I(inp)
          return self.E.spike

      model = EINet()
      indices = bm.arange(1000)
      spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
      bp.visualize.raster_plot(indices, spks, show=True)


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
      pre: DynamicalSystem,
      syn: ParamDescInit[JointType[DynamicalSystem, AutoDelaySupp]],
      delay: Union[None, int, float],
      comm: Callable,
      out: JointType[DynamicalSystem, BindCondData],
      post: Dynamic,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(pre, DynamicalSystem)
    check.is_instance(syn, ParamDescInit[JointType[DynamicalSystem, AutoDelaySupp]])
    check.is_instance(comm, Callable)
    check.is_instance(out, JointType[DynamicalSystem, BindCondData])
    check.is_instance(post, Dynamic)
    self.pre = pre
    self.post = post
    self.comm = comm

    # synapse and delay initialization
    self._syn_id = syn.identifier
    if self._syn_id not in pre.after_updates:
      # "syn_cls" needs an instance of "ProjAutoDelay"
      syn_cls: AutoDelaySupp = syn()
      delay_cls = _init_delay(syn_cls.return_info())
      # add to "after_updates"
      pre.after_updates[self._syn_id] = _AlignPre(syn_cls, delay_cls)
    delay_cls: Delay = pre.after_updates[self._syn_id].delay
    delay_cls.register_entry(self.name, delay)

    # output initialization
    post.cur_inputs[self.name] = out

  def update(self, x=None):
    if x is None:
      x = self.pre.after_updates[self._syn_id].delay.at(self.name)
    current = self.comm(x)
    self.post.cur_inputs[self.name].bind_cond(current)
    return current


class ProjAlignPreMg2(Projection):
  """Synaptic projection which defines the synaptic computation with the dimension of presynaptic neuron group.

  To simulate an E/I balanced network model:

  .. code-block:: python

      class EINet(bp.DynSysGroup):
        def __init__(self):
          super().__init__()
          ne, ni = 3200, 800
          self.E = bp.dyn.LifRef(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.I = bp.dyn.LifRef(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.E2E = bp.dyn.ProjAlignPreMg2(pre=self.E,
                                            delay=0.1,
                                            syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                            comm=bp.dnn.JitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                            out=bp.dyn.COBA(E=0.),
                                            post=self.E)
          self.E2I = bp.dyn.ProjAlignPreMg2(pre=self.E,
                                            delay=0.1,
                                            syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                            comm=bp.dnn.JitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                            out=bp.dyn.COBA(E=0.),
                                            post=self.I)
          self.I2E = bp.dyn.ProjAlignPreMg2(pre=self.I,
                                            delay=0.1,
                                            syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                            comm=bp.dnn.JitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                            out=bp.dyn.COBA(E=-80.),
                                            post=self.E)
          self.I2I = bp.dyn.ProjAlignPreMg2(pre=self.I,
                                            delay=0.1,
                                            syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                            comm=bp.dnn.JitFPHomoLinear(ni, ni, prob=0.02, weight=6.7),
                                            out=bp.dyn.COBA(E=-80.),
                                            post=self.I)

        def update(self, inp):
          self.E2E()
          self.E2I()
          self.I2E()
          self.I2I()
          self.E(inp)
          self.I(inp)
          return self.E.spike

      model = EINet()
      indices = bm.arange(1000)
      spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
      bp.visualize.raster_plot(indices, spks, show=True)


  Args:
    pre: The pre-synaptic neuron group.
    delay: The synaptic delay.
    syn: The synaptic dynamics.
    comm: The synaptic communication.
    out: The synaptic output.
    post: The post-synaptic neuron group.
    name: str. The projection name.
    mode: Mode. The computing mode.
  """

  def __init__(
      self,
      pre: JointType[DynamicalSystem, AutoDelaySupp],
      delay: Union[None, int, float],
      syn: ParamDescInit[JointType[DynamicalSystem, AutoDelaySupp]],
      comm: Callable,
      out: JointType[DynamicalSystem, BindCondData],
      post: Dynamic,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(pre, JointType[DynamicalSystem, AutoDelaySupp])
    check.is_instance(syn, ParamDescInit[JointType[DynamicalSystem, AutoDelaySupp]])
    check.is_instance(comm, Callable)
    check.is_instance(out, JointType[DynamicalSystem, BindCondData])
    check.is_instance(post, Dynamic)
    self.pre = pre
    self.post = post
    self.comm = comm

    # synapse and delay initialization
    if _pre_delay_repr not in self.pre.after_updates:
      delay_ins = _init_delay(pre.return_info())
      self.pre.after_updates[_pre_delay_repr] = delay_ins

    # synapse
    self._syn_id = f'{str(delay)} / {syn.identifier}'
    if self._syn_id not in post.before_updates:
      # delay
      delay_ins: Delay = pre.after_updates[_pre_delay_repr]
      delay_access = DelayAccess(delay_ins, delay)
      # synapse
      syn_cls = syn()
      # add to "after_updates"
      post.before_updates[self._syn_id] = _AlignPreMg(delay_access, syn_cls)

    # output initialization
    post.cur_inputs[self.name] = out

  def update(self):
    x = _get_return(self.post.before_updates[self._syn_id].syn.return_info())
    current = self.comm(x)
    self.post.cur_inputs[self.name].bind_cond(current)
    return current
