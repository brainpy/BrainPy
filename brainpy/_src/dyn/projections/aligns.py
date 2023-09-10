from typing import Optional, Callable, Union

from brainpy.types import ArrayType
from brainpy import math as bm, check
from brainpy._src.delay import Delay, DelayAccess, delay_identifier, init_delay_by_return
from brainpy._src.dynsys import DynamicalSystem, Projection
from brainpy._src.mixin import (JointType, ParamDescInit, ReturnInfo,
                                AutoDelaySupp, BindCondData, AlignPost, SupportPlasticity)
from brainpy._src.initialize import parameter
from brainpy._src.dyn.synapses.abstract_models import Expon

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
      pre: JointType[DynamicalSystem, AutoDelaySupp],
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
    check.is_instance(pre, JointType[DynamicalSystem, AutoDelaySupp])
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
      pre: JointType[DynamicalSystem, AutoDelaySupp],
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
    check.is_instance(pre, JointType[DynamicalSystem, AutoDelaySupp])
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
      syn: ParamDescInit[JointType[DynamicalSystem, AutoDelaySupp]],
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
    check.is_instance(syn, ParamDescInit[JointType[DynamicalSystem, AutoDelaySupp]])
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(out, JointType[DynamicalSystem, BindCondData])
    check.is_instance(post, DynamicalSystem)
    self.comm = comm

    # synapse and delay initialization
    self._syn_id = f'{syn.identifier} // Delay'
    if not pre.has_aft_update(self._syn_id):
      # "syn_cls" needs an instance of "ProjAutoDelay"
      syn_cls: AutoDelaySupp = syn()
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
      pre: JointType[DynamicalSystem, AutoDelaySupp],
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
    check.is_instance(pre, JointType[DynamicalSystem, AutoDelaySupp])
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
      syn: JointType[DynamicalSystem, AutoDelaySupp],
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
    check.is_instance(syn, JointType[DynamicalSystem, AutoDelaySupp])
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
      pre: JointType[DynamicalSystem, AutoDelaySupp],
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
    check.is_instance(pre, JointType[DynamicalSystem, AutoDelaySupp])
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
  >>> import brainpy as bp
  >>> import brainpy.math as bm
  >>> class STDPNet(bp.DynamicalSystem):
  >>>    def __init__(self, num_pre, num_post):
  >>>      super().__init__()
  >>>      self.pre = bp.dyn.LifRef(num_pre, name='neu1')
  >>>      self.post = bp.dyn.LifRef(num_post, name='neu2')
  >>>      self.syn = bp.dyn.STDP_Song2000(
  >>>        pre=self.pre,
  >>>        delay=1.,
  >>>        comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(1, pre=self.pre.num, post=self.post.num),
  >>>                                   weight=lambda s: bm.Variable(bm.random.rand(*s) * 0.1)),
  >>>        syn=bp.dyn.Expon.desc(self.post.varshape, tau=5.),
  >>>        out=bp.dyn.COBA.desc(E=0.),
  >>>        post=self.post,
  >>>        tau_s=16.8,
  >>>        tau_t=33.7,
  >>>        A1=0.96,
  >>>        A2=0.53,
  >>>      )
  >>>
  >>>    def update(self, I_pre, I_post):
  >>>      self.syn()
  >>>      self.pre(I_pre)
  >>>      self.post(I_post)
  >>>      conductance = self.syn.refs['syn'].g
  >>>      Apre = self.syn.refs['pre_trace'].g
  >>>      Apost = self.syn.refs['post_trace'].g
  >>>      current = self.post.sum_inputs(self.post.V)
  >>>      return self.pre.spike, self.post.spike, conductance, Apre, Apost, current, self.syn.comm.weight
  >>> duration = 300.
  >>> I_pre = bp.inputs.section_input([0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0],
  >>>                                 [5, 15, 15, 15, 15, 15, 100, 15, 15, 15, 15, 15, duration - 255])
  >>> I_post = bp.inputs.section_input([0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0],
  >>>                                  [10, 15, 15, 15, 15, 15, 90, 15, 15, 15, 15, 15, duration - 250])
  >>>
  >>> net = STDPNet(1, 1)
  >>> def run(i, I_pre, I_post):
  >>>   pre_spike, post_spike, g, Apre, Apost, current, W = net.step_run(i, I_pre, I_post)
  >>>   return pre_spike, post_spike, g, Apre, Apost, current, W
  >>>
  >>> indices = bm.arange(0, duration, bm.dt)
  >>> pre_spike, post_spike, g, Apre, Apost, current, W = bm.for_loop(run, [indices, I_pre, I_post], jit=True)

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
    self.comm.plasticity(delta_w)

    current = self.comm(x)
    if issubclass(self.syn.cls, AlignPost):
      self.refs['syn'].add_current(current)  # synapse post current
    else:
      self.refs['out'].bind_cond(current)
    return current
