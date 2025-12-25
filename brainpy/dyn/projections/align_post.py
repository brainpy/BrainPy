# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Optional, Callable, Union

from brainpy import math as bm, check
from brainpy.delay import (delay_identifier,
                           register_delay_by_return)
from brainpy.dynsys import DynamicalSystem, Projection
from brainpy.mixin import (JointType, ParamDescriber, SupportAutoDelay, BindCondData, AlignPost)

__all__ = [
    'HalfProjAlignPostMg', 'FullProjAlignPostMg',
    'HalfProjAlignPost', 'FullProjAlignPost',

]


def get_post_repr(out_label, syn, out):
    return f'{out_label} // {syn.identifier} // {out.identifier}'


def align_post_add_bef_update(out_label, syn_desc, out_desc, post, proj_name):
    # synapse and output initialization
    _post_repr = get_post_repr(out_label, syn_desc, out_desc)
    if not post.has_bef_update(_post_repr):
        syn_cls = syn_desc()
        out_cls = out_desc()

        # synapse and output initialization
        post.add_inp_fun(proj_name, out_cls, label=out_label)
        post.add_bef_update(_post_repr, _AlignPost(syn_cls, out_cls))
    syn = post.get_bef_update(_post_repr).syn
    out = post.get_bef_update(_post_repr).out
    return syn, out


class _AlignPost(DynamicalSystem):
    def __init__(self,
                 syn: Callable,
                 out: JointType[DynamicalSystem, BindCondData]):
        super().__init__()
        self.syn = syn
        self.out = out

    def update(self, *args, **kwargs):
        self.out.bind_cond(self.syn(*args, **kwargs))

    def reset_state(self, *args, **kwargs):
        pass


class HalfProjAlignPostMg(Projection):
    r"""Defining the half part of synaptic projection with the align-post reduction and the automatic synapse merging.

    The ``half-part`` means that the model only needs to provide half information needed for a projection,
    including ``comm`` -> ``syn`` -> ``out`` -> ``post``. Therefore, the model's ``update`` function needs
    the manual providing of the spiking input.

    The ``align-post`` means that the synaptic variables have the same dimension as the post-synaptic neuron group.

    The ``merging`` means that the same delay model is shared by all synapses, and the synapse model with same
    parameters (such like time constants) will also share the same synaptic variables.

    All align-post projection models prefer to use the event-driven computation mode. This means that the
    ``comm`` model should be the event-driven model.

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
          self.E = bp.dyn.HalfProjAlignPostMg(comm=bp.dnn.EventJitFPHomoLinear(3200, 4000, prob=0.02, weight=0.6),
                                           syn=bp.dyn.Expon.desc(size=4000, tau=5.),
                                           out=bp.dyn.COBA.desc(E=0.),
                                           post=self.N)
          self.I = bp.dyn.HalfProjAlignPostMg(comm=bp.dnn.EventJitFPHomoLinear(800, 4000, prob=0.02, weight=6.7),
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
        syn: ParamDescriber[JointType[DynamicalSystem, AlignPost]],
        out: ParamDescriber[JointType[DynamicalSystem, BindCondData]],
        post: DynamicalSystem,
        out_label: Optional[str] = None,
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
    ):
        super().__init__(name=name, mode=mode)

        # synaptic models
        check.is_instance(comm, DynamicalSystem)
        check.is_instance(syn, ParamDescriber[JointType[DynamicalSystem, AlignPost]])
        check.is_instance(out, ParamDescriber[JointType[DynamicalSystem, BindCondData]])
        check.is_instance(post, DynamicalSystem)
        self.comm = comm

        # synapse and output initialization
        syn, out = align_post_add_bef_update(out_label, syn_desc=syn, out_desc=out, post=post, proj_name=self.name)

        # references
        self.refs = dict(post=post)  # invisible to ``self.nodes()``
        self.refs['syn'] = syn
        self.refs['out'] = out
        self.refs['comm'] = comm  # unify the access

    def update(self, x):
        current = self.comm(x)
        self.refs['syn'].add_current(current)  # synapse post current
        return current

    syn = property(lambda self: self.refs['syn'])
    out = property(lambda self: self.refs['out'])
    post = property(lambda self: self.refs['post'])


class FullProjAlignPostMg(Projection):
    """Full-chain synaptic projection with the align-post reduction and the automatic synapse merging.

    The ``full-chain`` means that the model needs to provide all information needed for a projection,
    including ``pre`` -> ``delay`` -> ``comm`` -> ``syn`` -> ``out`` -> ``post``.

    The ``align-post`` means that the synaptic variables have the same dimension as the post-synaptic neuron group.

    The ``merging`` means that the same delay model is shared by all synapses, and the synapse model with same
    parameters (such like time constants) will also share the same synaptic variables.

    All align-post projection models prefer to use the event-driven computation mode. This means that the
    ``comm`` model should be the event-driven model.

    Moreover, it's worth noting that ``FullProjAlignPostMg`` has a different updating order with all align-pre
    projection models. The updating order of align-post projections is ``spikes`` -> ``comm`` -> ``syn`` -> ``out``.
    While, the updating order of all align-pre projection models is usually ``spikes`` -> ``syn`` -> ``comm`` -> ``out``.

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
            self.E2E = bp.dyn.FullProjAlignPostMg(pre=self.E,
                                               delay=0.1,
                                               comm=bp.dnn.EventJitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                               syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                               out=bp.dyn.COBA.desc(E=0.),
                                               post=self.E)
            self.E2I = bp.dyn.FullProjAlignPostMg(pre=self.E,
                                               delay=0.1,
                                               comm=bp.dnn.EventJitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                               syn=bp.dyn.Expon.desc(size=ni, tau=5.),
                                               out=bp.dyn.COBA.desc(E=0.),
                                               post=self.I)
            self.I2E = bp.dyn.FullProjAlignPostMg(pre=self.I,
                                               delay=0.1,
                                               comm=bp.dnn.EventJitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                               syn=bp.dyn.Expon.desc(size=ne, tau=10.),
                                               out=bp.dyn.COBA.desc(E=-80.),
                                               post=self.E)
            self.I2I = bp.dyn.FullProjAlignPostMg(pre=self.I,
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
        syn: ParamDescriber[JointType[DynamicalSystem, AlignPost]],
        out: ParamDescriber[JointType[DynamicalSystem, BindCondData]],
        post: DynamicalSystem,
        out_label: Optional[str] = None,
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
    ):
        super().__init__(name=name, mode=mode)

        # synaptic models
        check.is_instance(pre, JointType[DynamicalSystem, SupportAutoDelay])
        check.is_instance(comm, DynamicalSystem)
        check.is_instance(syn, ParamDescriber[JointType[DynamicalSystem, AlignPost]])
        check.is_instance(out, ParamDescriber[JointType[DynamicalSystem, BindCondData]])
        check.is_instance(post, DynamicalSystem)
        self.comm = comm

        # delay initialization
        delay_cls = register_delay_by_return(pre)
        delay_cls.register_entry(self.name, delay)

        # synapse and output initialization
        syn, out = align_post_add_bef_update(out_label, syn_desc=syn, out_desc=out, post=post, proj_name=self.name)

        # references
        self.refs = dict(pre=pre, post=post)  # invisible to ``self.nodes()``
        self.refs['syn'] = syn  # invisible to ``self.node()``
        self.refs['out'] = out  # invisible to ``self.node()``
        # unify the access
        self.refs['comm'] = comm
        self.refs['delay'] = pre.get_aft_update(delay_identifier)

    def update(self):
        x = self.refs['pre'].get_aft_update(delay_identifier).at(self.name)
        current = self.comm(x)
        self.refs['syn'].add_current(current)  # synapse post current
        return current

    syn = property(lambda self: self.refs['syn'])
    out = property(lambda self: self.refs['out'])
    delay = property(lambda self: self.refs['delay'])
    pre = property(lambda self: self.refs['pre'])
    post = property(lambda self: self.refs['post'])


class HalfProjAlignPost(Projection):
    """Defining the half-part of synaptic projection with the align-post reduction.

    The ``half-part`` means that the model only needs to provide half information needed for a projection,
    including ``comm`` -> ``syn`` -> ``out`` -> ``post``. Therefore, the model's ``update`` function needs
    the manual providing of the spiking input.

    The ``align-post`` means that the synaptic variables have the same dimension as the post-synaptic neuron group.

    All align-post projection models prefer to use the event-driven computation mode. This means that the
    ``comm`` model should be the event-driven model.

    To simulate an E/I balanced network:

    .. code-block::

        class EINet(bp.DynSysGroup):
          def __init__(self):
            super().__init__()
            self.N = bp.dyn.LifRef(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                   V_initializer=bp.init.Normal(-55., 2.))
            self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
            self.E = bp.dyn.HalfProjAlignPost(comm=bp.dnn.EventJitFPHomoLinear(3200, 4000, prob=0.02, weight=0.6),
                                           syn=bp.dyn.Expon(size=4000, tau=5.),
                                           out=bp.dyn.COBA(E=0.),
                                           post=self.N)
            self.I = bp.dyn.HalfProjAlignPost(comm=bp.dnn.EventJitFPHomoLinear(800, 4000, prob=0.02, weight=6.7),
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
        self.syn = syn
        self.out = out

        # synapse and output initialization
        post.add_inp_fun(self.name, out, label=out_label)

        # reference
        self.refs = dict()
        # invisible to ``self.nodes()``
        self.refs['post'] = post
        self.refs['syn'] = syn
        self.refs['out'] = out
        # unify the access
        self.refs['comm'] = comm

    def update(self, x):
        current = self.comm(x)
        g = self.syn(self.comm(x))
        self.refs['out'].bind_cond(g)  # synapse post current
        return current

    post = property(lambda self: self.refs['post'])


class FullProjAlignPost(Projection):
    """Full-chain synaptic projection with the align-post reduction.

    The ``full-chain`` means that the model needs to provide all information needed for a projection,
    including ``pre`` -> ``delay`` -> ``comm`` -> ``syn`` -> ``out`` -> ``post``.

    The ``align-post`` means that the synaptic variables have the same dimension as the post-synaptic neuron group.

    All align-post projection models prefer to use the event-driven computation mode. This means that the
    ``comm`` model should be the event-driven model.

    Moreover, it's worth noting that ``FullProjAlignPost`` has a different updating order with all align-pre
    projection models. The updating order of align-post projections is ``spikes`` -> ``comm`` -> ``syn`` -> ``out``.
    While, the updating order of all align-pre projection models is usually ``spikes`` -> ``syn`` -> ``comm`` -> ``out``.

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
            self.E2E = bp.dyn.FullProjAlignPost(pre=self.E,
                                             delay=0.1,
                                             comm=bp.dnn.EventJitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                             syn=bp.dyn.Expon(size=ne, tau=5.),
                                             out=bp.dyn.COBA(E=0.),
                                             post=self.E)
            self.E2I = bp.dyn.FullProjAlignPost(pre=self.E,
                                             delay=0.1,
                                             comm=bp.dnn.EventJitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                             syn=bp.dyn.Expon(size=ni, tau=5.),
                                             out=bp.dyn.COBA(E=0.),
                                             post=self.I)
            self.I2E = bp.dyn.FullProjAlignPost(pre=self.I,
                                             delay=0.1,
                                             comm=bp.dnn.EventJitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                             syn=bp.dyn.Expon(size=ne, tau=10.),
                                             out=bp.dyn.COBA(E=-80.),
                                             post=self.E)
            self.I2I = bp.dyn.FullProjAlignPost(pre=self.I,
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
        delay_cls = register_delay_by_return(pre)
        delay_cls.register_entry(self.name, delay)

        # synapse and output initialization
        post.add_inp_fun(self.name, out, label=out_label)

        # references
        self.refs = dict()
        # invisible to ``self.nodes()``
        self.refs['pre'] = pre
        self.refs['post'] = post
        self.refs['out'] = out
        # unify the access
        self.refs['delay'] = delay_cls
        self.refs['comm'] = comm
        self.refs['syn'] = syn

    def update(self):
        x = self.refs['delay'].at(self.name)
        g = self.syn(self.comm(x))
        self.refs['out'].bind_cond(g)  # synapse post current
        return g

    delay = property(lambda self: self.refs['delay'])
    pre = property(lambda self: self.refs['pre'])
    post = property(lambda self: self.refs['post'])
    out = property(lambda self: self.refs['out'])
