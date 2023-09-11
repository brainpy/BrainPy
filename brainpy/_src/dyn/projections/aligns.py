from typing import Optional, Callable, Union

from brainpy import math as bm, check
from brainpy._src.delay import Delay, DelayAccess, delay_identifier, init_delay_by_return
from brainpy._src.dynsys import DynamicalSystem, Projection
from brainpy._src.mixin import (JointType, ParamDescInit, ReturnInfo,
                                SupportAutoDelay, BindCondData, AlignPost)

__all__ = [
  'VanillaProj',
  'ProjAlignPostMg1', 'ProjAlignPostMg2',
  'ProjAlignPost1', 'ProjAlignPost2',
  'ProjAlignPreMg1', 'ProjAlignPreMg2',
  'ProjAlignPre1', 'ProjAlignPre2',
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
          self.E = bp.dyn.VanillaProj(comm=bp.dnn.JitFPHomoLinear(3200, 4000, prob=0.02, weight=0.6),
                                      out=bp.dyn.COBA(E=0.),
                                      post=self.N)
          self.I = bp.dyn.VanillaProj(comm=bp.dnn.JitFPHomoLinear(800, 4000, prob=0.02, weight=6.7),
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
      post: DynamicalSystem,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(out, JointType[DynamicalSystem, BindCondData])
    check.is_instance(post, DynamicalSystem)
    self.comm = comm

    # output initialization
    post.add_inp_fun(self.name, out)

    # references
    self.refs = dict(post=post, out=out)  # invisible to ``self.nodes()``
    self.refs['comm'] = comm  # unify the access

  def update(self, x):
    current = self.comm(x)
    self.refs['out'].bind_cond(current)
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
    out_label: str. The prefix of the output function.
    name: str. The projection name.
    mode:  Mode. The computing mode.
  """

  def __init__(
      self,
      comm: DynamicalSystem,
      syn: ParamDescInit[JointType[DynamicalSystem, AlignPost]],
      out: ParamDescInit[JointType[DynamicalSystem, BindCondData]],
      post: DynamicalSystem,
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(syn, ParamDescInit[JointType[DynamicalSystem, AlignPost]])
    check.is_instance(out, ParamDescInit[JointType[DynamicalSystem, BindCondData]])
    check.is_instance(post, DynamicalSystem)
    self.comm = comm

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
    self.refs = dict(post=post)  # invisible to ``self.nodes()``
    self.refs['syn'] = post.get_bef_update(self._post_repr).syn
    self.refs['out'] = post.get_bef_update(self._post_repr).out
    self.refs['comm'] = comm  # unify the access

  def update(self, x):
    current = self.comm(x)
    self.refs['syn'].add_current(current)  # synapse post current
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
      pre: JointType[DynamicalSystem, SupportAutoDelay],
      delay: Union[None, int, float],
      comm: DynamicalSystem,
      syn: ParamDescInit[JointType[DynamicalSystem, AlignPost]],
      out: ParamDescInit[JointType[DynamicalSystem, BindCondData]],
      post: DynamicalSystem,
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(pre, JointType[DynamicalSystem, SupportAutoDelay])
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(syn, ParamDescInit[JointType[DynamicalSystem, AlignPost]])
    check.is_instance(out, ParamDescInit[JointType[DynamicalSystem, BindCondData]])
    check.is_instance(post, DynamicalSystem)
    self.comm = comm

    # delay initialization
    if not pre.has_aft_update(delay_identifier):
      # pre should support "ProjAutoDelay"
      delay_cls = init_delay_by_return(pre.return_info())
      # add to "after_updates"
      pre.add_aft_update(delay_identifier, delay_cls)
    delay_cls: Delay = pre.get_aft_update(delay_identifier)
    delay_cls.register_entry(self.name, delay)

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
    self.refs = dict(pre=pre, post=post)  # invisible to ``self.nodes()``
    self.refs['syn'] = post.get_bef_update(self._post_repr).syn  # invisible to ``self.node()``
    self.refs['out'] = post.get_bef_update(self._post_repr).out  # invisible to ``self.node()``
    # unify the access
    self.refs['comm'] = comm
    self.refs['delay'] = pre.get_aft_update(delay_identifier)

  def update(self):
    x = self.refs['pre'].get_aft_update(delay_identifier).at(self.name)
    current = self.comm(x)
    self.refs['syn'].add_current(current)  # synapse post current
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
      post: DynamicalSystem,
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(syn, JointType[DynamicalSystem, AlignPost])
    check.is_instance(out, JointType[DynamicalSystem, BindCondData])
    check.is_instance(post, DynamicalSystem)
    self.comm = comm

    # synapse and output initialization
    if out_label is None:
      out_name = self.name
    else:
      out_name = f'{out_label} // {self.name}'
    post.add_inp_fun(out_name, out)
    post.add_bef_update(self.name, _AlignPost(syn, out))

    # reference
    self.refs = dict()
    # invisible to ``self.nodes()``
    self.refs['post'] = post
    self.refs['syn'] = post.get_bef_update(self.name).syn
    self.refs['out'] = post.get_bef_update(self.name).out
    # unify the access
    self.refs['comm'] = comm

  def update(self, x):
    current = self.comm(x)
    self.refs['syn'].add_current(current)
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
      pre: JointType[DynamicalSystem, SupportAutoDelay],
      delay: Union[None, int, float],
      comm: DynamicalSystem,
      syn: JointType[DynamicalSystem, AlignPost],
      out: JointType[DynamicalSystem, BindCondData],
      post: DynamicalSystem,
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(pre, JointType[DynamicalSystem, SupportAutoDelay])
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(syn, JointType[DynamicalSystem, AlignPost])
    check.is_instance(out, JointType[DynamicalSystem, BindCondData])
    check.is_instance(post, DynamicalSystem)
    self.comm = comm
    self.syn = syn

    # delay initialization
    if not pre.has_aft_update(delay_identifier):
      # pre should support "ProjAutoDelay"
      delay_cls = init_delay_by_return(pre.return_info())
      # add to "after_updates"
      pre.add_aft_update(delay_identifier, delay_cls)
    delay_cls: Delay = pre.get_aft_update(delay_identifier)
    delay_cls.register_entry(self.name, delay)

    # synapse and output initialization
    if out_label is None:
      out_name = self.name
    else:
      out_name = f'{out_label} // {self.name}'
    post.add_inp_fun(out_name, out)

    # references
    self.refs = dict()
    # invisible to ``self.nodes()``
    self.refs['pre'] = pre
    self.refs['post'] = post
    self.refs['out'] = out
    # unify the access
    self.refs['delay'] = pre.get_aft_update(delay_identifier)
    self.refs['comm'] = comm
    self.refs['syn'] = syn

  def update(self):
    x = self.refs['pre'].get_aft_update(delay_identifier).at(self.name)
    g = self.syn(self.comm(x))
    self.refs['out'].bind_cond(g)  # synapse post current
    return g


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
      syn: ParamDescInit[JointType[DynamicalSystem, SupportAutoDelay]],
      delay: Union[None, int, float],
      comm: DynamicalSystem,
      out: JointType[DynamicalSystem, BindCondData],
      post: DynamicalSystem,
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(pre, DynamicalSystem)
    check.is_instance(syn, ParamDescInit[JointType[DynamicalSystem, SupportAutoDelay]])
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(out, JointType[DynamicalSystem, BindCondData])
    check.is_instance(post, DynamicalSystem)
    self.comm = comm

    # synapse and delay initialization
    self._syn_id = f'{syn.identifier} // Delay'
    if not pre.has_aft_update(self._syn_id):
      # "syn_cls" needs an instance of "ProjAutoDelay"
      syn_cls: SupportAutoDelay = syn()
      delay_cls = init_delay_by_return(syn_cls.return_info())
      # add to "after_updates"
      pre.add_aft_update(self._syn_id, _AlignPre(syn_cls, delay_cls))
    delay_cls: Delay = pre.get_aft_update(self._syn_id).delay
    delay_cls.register_entry(self.name, delay)

    # output initialization
    if out_label is None:
      out_name = self.name
    else:
      out_name = f'{out_label} // {self.name}'
    post.add_inp_fun(out_name, out)

    # references
    self.refs = dict()
    # invisible to ``self.nodes()``
    self.refs['pre'] = pre
    self.refs['post'] = post
    self.refs['out'] = out
    self.refs['delay'] = delay_cls
    self.refs['syn'] = pre.get_aft_update(self._syn_id).syn
    # unify the access
    self.refs['comm'] = comm

  def update(self, x=None):
    if x is None:
      x = self.refs['delay'].at(self.name)
    current = self.comm(x)
    self.refs['out'].bind_cond(current)
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
      pre: JointType[DynamicalSystem, SupportAutoDelay],
      delay: Union[None, int, float],
      syn: ParamDescInit[DynamicalSystem],
      comm: DynamicalSystem,
      out: JointType[DynamicalSystem, BindCondData],
      post: DynamicalSystem,
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(pre, JointType[DynamicalSystem, SupportAutoDelay])
    check.is_instance(syn, ParamDescInit[DynamicalSystem])
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(out, JointType[DynamicalSystem, BindCondData])
    check.is_instance(post, DynamicalSystem)
    self.comm = comm

    # delay initialization
    if not pre.has_aft_update(delay_identifier):
      delay_ins = init_delay_by_return(pre.return_info())
      pre.add_aft_update(delay_identifier, delay_ins)
    delay_cls = pre.get_aft_update(delay_identifier)

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
    self.refs = dict()
    # invisible to `self.nodes()`
    self.refs['pre'] = pre
    self.refs['post'] = post
    self.refs['syn'] = delay_cls.get_bef_update(self._syn_id).syn
    self.refs['out'] = out
    # unify the access
    self.refs['comm'] = comm

  def update(self):
    x = _get_return(self.refs['syn'].return_info())
    current = self.comm(x)
    self.refs['out'].bind_cond(current)
    return current


class ProjAlignPre1(Projection):
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
      syn: JointType[DynamicalSystem, SupportAutoDelay],
      delay: Union[None, int, float],
      comm: DynamicalSystem,
      out: JointType[DynamicalSystem, BindCondData],
      post: DynamicalSystem,
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(pre, DynamicalSystem)
    check.is_instance(syn, JointType[DynamicalSystem, SupportAutoDelay])
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(out, JointType[DynamicalSystem, BindCondData])
    check.is_instance(post, DynamicalSystem)
    self.comm = comm

    # synapse and delay initialization
    delay_cls = init_delay_by_return(syn.return_info())
    delay_cls.register_entry(self.name, delay)
    pre.add_aft_update(self.name, _AlignPre(syn, delay_cls))

    # output initialization
    if out_label is None:
      out_name = self.name
    else:
      out_name = f'{out_label} // {self.name}'
    post.add_inp_fun(out_name, out)

    # references
    self.refs = dict()
    # invisible to ``self.nodes()``
    self.refs['pre'] = pre
    self.refs['post'] = post
    self.refs['out'] = out
    self.refs['delay'] = delay_cls
    self.refs['syn'] = syn
    # unify the access
    self.refs['comm'] = comm

  def update(self, x=None):
    if x is None:
      x = self.refs['delay'].at(self.name)
    current = self.comm(x)
    self.refs['out'].bind_cond(current)
    return current


class ProjAlignPre2(Projection):
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
      pre: JointType[DynamicalSystem, SupportAutoDelay],
      delay: Union[None, int, float],
      syn: DynamicalSystem,
      comm: DynamicalSystem,
      out: JointType[DynamicalSystem, BindCondData],
      post: DynamicalSystem,
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(pre, JointType[DynamicalSystem, SupportAutoDelay])
    check.is_instance(syn, DynamicalSystem)
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(out, JointType[DynamicalSystem, BindCondData])
    check.is_instance(post, DynamicalSystem)
    self.comm = comm
    self.syn = syn

    # delay initialization
    if not pre.has_aft_update(delay_identifier):
      delay_ins = init_delay_by_return(pre.return_info())
      pre.add_aft_update(delay_identifier, delay_ins)
    delay_cls = pre.get_aft_update(delay_identifier)
    delay_cls.register_entry(self.name, delay)

    # output initialization
    if out_label is None:
      out_name = self.name
    else:
      out_name = f'{out_label} // {self.name}'
    post.add_inp_fun(out_name, out)

    # references
    self.refs = dict()
    # invisible to ``self.nodes()``
    self.refs['pre'] = pre
    self.refs['post'] = post
    self.refs['out'] = out
    self.refs['delay'] = pre.get_aft_update(delay_identifier)
    # unify the access
    self.refs['syn'] = syn
    self.refs['comm'] = comm

  def update(self):
    spk = self.refs['delay'].at(self.name)
    g = self.comm(self.syn(spk))
    self.refs['out'].bind_cond(g)
    return g