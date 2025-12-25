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
from typing import Optional, Union

from brainpy import math as bm, check
from brainpy.delay import (Delay, DelayAccess, init_delay_by_return, register_delay_by_return)
from brainpy.dynsys import DynamicalSystem, Projection
from brainpy.mixin import (JointType, ParamDescriber, SupportAutoDelay, BindCondData)
from .utils import _get_return

__all__ = [
    'FullProjAlignPreSDMg', 'FullProjAlignPreDSMg',
    'FullProjAlignPreSD', 'FullProjAlignPreDS',
]


def align_pre2_add_bef_update(syn_desc, delay, delay_cls, proj_name=None):
    _syn_id = f'Delay({str(delay)}) // {syn_desc.identifier}'
    if not delay_cls.has_bef_update(_syn_id):
        # delay
        delay_access = DelayAccess(delay_cls, delay, delay_entry=proj_name)
        # synapse
        syn_cls = syn_desc()
        # add to "after_updates"
        delay_cls.add_bef_update(_syn_id, _AlignPreMg(delay_access, syn_cls))
    syn = delay_cls.get_bef_update(_syn_id).syn
    return syn


class _AlignPreMg(DynamicalSystem):
    def __init__(self, access, syn):
        super().__init__()
        self.access = access
        self.syn = syn

    def update(self, *args, **kwargs):
        return self.syn(self.access())

    def reset_state(self, *args, **kwargs):
        pass


def align_pre1_add_bef_update(syn_desc, pre):
    _syn_id = f'{syn_desc.identifier} // Delay'
    if not pre.has_aft_update(_syn_id):
        # "syn_cls" needs an instance of "ProjAutoDelay"
        syn_cls: SupportAutoDelay = syn_desc()
        delay_cls = init_delay_by_return(syn_cls.return_info())
        # add to "after_updates"
        pre.add_aft_update(_syn_id, _AlignPre(syn_cls, delay_cls))
    delay_cls: Delay = pre.get_aft_update(_syn_id).delay
    syn = pre.get_aft_update(_syn_id).syn
    return delay_cls, syn


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

    def reset_state(self, *args, **kwargs):
        pass


class FullProjAlignPreSDMg(Projection):
    """Full-chain synaptic projection with the align-pre reduction and synapse+delay updating and merging.

    The ``full-chain`` means that the model needs to provide all information needed for a projection,
    including ``pre`` -> ``syn`` -> ``delay`` -> ``comm`` -> ``out`` -> ``post``.

    The ``align-pre`` means that the synaptic variables have the same dimension as the pre-synaptic neuron group.

    The ``synapse+delay updating`` means that the projection first computes the synapse states, then delivers the
    synapse states to the delay model, and finally computes the synaptic current.

    The ``merging`` means that the same delay model is shared by all synapses, and the synapse model with same
    parameters (such like time constants) will also share the same synaptic variables.

    Neither ``FullProjAlignPreSDMg`` nor ``FullProjAlignPreDSMg`` facilitates the event-driven computation.
    This is because the ``comm`` is computed after the synapse state, which is a floating-point number, rather
    than the spiking. To facilitate the event-driven computation, please use align post projections.

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
            self.E2E = bp.dyn.FullProjAlignPreSDMg(pre=self.E,
                                                  syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                  delay=0.1,
                                                  comm=bp.dnn.JitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                                  out=bp.dyn.COBA(E=0.),
                                                  post=self.E)
            self.E2I = bp.dyn.FullProjAlignPreSDMg(pre=self.E,
                                                  syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                  delay=0.1,
                                                  comm=bp.dnn.JitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                                  out=bp.dyn.COBA(E=0.),
                                                  post=self.I)
            self.I2E = bp.dyn.FullProjAlignPreSDMg(pre=self.I,
                                                  syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                                  delay=0.1,
                                                  comm=bp.dnn.JitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                                  out=bp.dyn.COBA(E=-80.),
                                                  post=self.E)
            self.I2I = bp.dyn.FullProjAlignPreSDMg(pre=self.I,
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
        syn: ParamDescriber[JointType[DynamicalSystem, SupportAutoDelay]],
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
        check.is_instance(syn, ParamDescriber[JointType[DynamicalSystem, SupportAutoDelay]])
        check.is_instance(comm, DynamicalSystem)
        check.is_instance(out, JointType[DynamicalSystem, BindCondData])
        check.is_instance(post, DynamicalSystem)
        self.comm = comm

        # synapse and delay initialization
        delay_cls, syn_cls = align_pre1_add_bef_update(syn, pre)
        delay_cls.register_entry(self.name, delay)

        # output initialization
        post.add_inp_fun(self.name, out, label=out_label)

        # references
        self.refs = dict()
        # invisible to ``self.nodes()``
        self.refs['pre'] = pre
        self.refs['post'] = post
        self.refs['out'] = out
        self.refs['delay'] = delay_cls
        self.refs['syn'] = syn_cls
        # unify the access
        self.refs['comm'] = comm

    def update(self, x=None):
        if x is None:
            x = self.refs['delay'].at(self.name)
        current = self.comm(x)
        self.refs['out'].bind_cond(current)
        return current

    pre = property(lambda self: self.refs['pre'])
    post = property(lambda self: self.refs['post'])
    syn = property(lambda self: self.refs['syn'])
    delay = property(lambda self: self.refs['delay'])
    out = property(lambda self: self.refs['out'])


class FullProjAlignPreDSMg(Projection):
    """Full-chain synaptic projection with the align-pre reduction and delay+synapse updating and merging.

    The ``full-chain`` means that the model needs to provide all information needed for a projection,
    including ``pre`` -> ``delay`` -> ``syn`` -> ``comm`` -> ``out`` -> ``post``.
    Note here, compared to ``FullProjAlignPreSDMg``, the ``delay`` and ``syn`` are exchanged.

    The ``align-pre`` means that the synaptic variables have the same dimension as the pre-synaptic neuron group.

    The ``delay+synapse updating`` means that the projection first delivers the pre neuron output (usually the
    spiking)  to the delay model, then computes the synapse states, and finally computes the synaptic current.

    The ``merging`` means that the same delay model is shared by all synapses, and the synapse model with same
    parameters (such like time constants) will also share the same synaptic variables.

    Neither ``FullProjAlignPreDSMg`` nor ``FullProjAlignPreSDMg`` facilitates the event-driven computation.
    This is because the ``comm`` is computed after the synapse state, which is a floating-point number, rather
    than the spiking. To facilitate the event-driven computation, please use align post projections.


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
            self.E2E = bp.dyn.FullProjAlignPreDSMg(pre=self.E,
                                                  delay=0.1,
                                                  syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                  comm=bp.dnn.JitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                                  out=bp.dyn.COBA(E=0.),
                                                  post=self.E)
            self.E2I = bp.dyn.FullProjAlignPreDSMg(pre=self.E,
                                                  delay=0.1,
                                                  syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                  comm=bp.dnn.JitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                                  out=bp.dyn.COBA(E=0.),
                                                  post=self.I)
            self.I2E = bp.dyn.FullProjAlignPreDSMg(pre=self.I,
                                                  delay=0.1,
                                                  syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                                  comm=bp.dnn.JitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                                  out=bp.dyn.COBA(E=-80.),
                                                  post=self.E)
            self.I2I = bp.dyn.FullProjAlignPreDSMg(pre=self.I,
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
        syn: ParamDescriber[DynamicalSystem],
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
        check.is_instance(syn, ParamDescriber[DynamicalSystem])
        check.is_instance(comm, DynamicalSystem)
        check.is_instance(out, JointType[DynamicalSystem, BindCondData])
        check.is_instance(post, DynamicalSystem)
        self.comm = comm

        # delay initialization
        delay_cls = register_delay_by_return(pre)

        # synapse initialization
        syn_cls = align_pre2_add_bef_update(syn, delay, delay_cls, self.name)

        # output initialization
        post.add_inp_fun(self.name, out, label=out_label)

        # references
        self.refs = dict()
        # invisible to `self.nodes()`
        self.refs['pre'] = pre
        self.refs['post'] = post
        self.refs['syn'] = syn_cls
        self.refs['out'] = out
        # unify the access
        self.refs['comm'] = comm

    def update(self):
        x = _get_return(self.refs['syn'].return_info())
        current = self.comm(x)
        self.refs['out'].bind_cond(current)
        return current

    pre = property(lambda self: self.refs['pre'])
    post = property(lambda self: self.refs['post'])
    syn = property(lambda self: self.refs['syn'])
    out = property(lambda self: self.refs['out'])


class FullProjAlignPreSD(Projection):
    """Full-chain synaptic projection with the align-pre reduction and synapse+delay updating.

    The ``full-chain`` means that the model needs to provide all information needed for a projection,
    including ``pre`` -> ``syn`` -> ``delay`` -> ``comm`` -> ``out`` -> ``post``.

    The ``align-pre`` means that the synaptic variables have the same dimension as the pre-synaptic neuron group.

    The ``synapse+delay updating`` means that the projection first computes the synapse states, then delivers the
    synapse states to the delay model, and finally computes the synaptic current.

    Neither ``FullProjAlignPreSD`` nor ``FullProjAlignPreDS`` facilitates the event-driven computation.
    This is because the ``comm`` is computed after the synapse state, which is a floating-point number, rather
    than the spiking. To facilitate the event-driven computation, please use align post projections.


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
            self.E2E = bp.dyn.FullProjAlignPreSD(pre=self.E,
                                                syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                delay=0.1,
                                                comm=bp.dnn.JitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                                out=bp.dyn.COBA(E=0.),
                                                post=self.E)
            self.E2I = bp.dyn.FullProjAlignPreSD(pre=self.E,
                                                syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                delay=0.1,
                                                comm=bp.dnn.JitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                                out=bp.dyn.COBA(E=0.),
                                                post=self.I)
            self.I2E = bp.dyn.FullProjAlignPreSD(pre=self.I,
                                                syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                                delay=0.1,
                                                comm=bp.dnn.JitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                                out=bp.dyn.COBA(E=-80.),
                                                post=self.E)
            self.I2I = bp.dyn.FullProjAlignPreSD(pre=self.I,
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
        post.add_inp_fun(self.name, out, label=out_label)

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

    pre = property(lambda self: self.refs['pre'])
    post = property(lambda self: self.refs['post'])
    syn = property(lambda self: self.refs['syn'])
    delay = property(lambda self: self.refs['delay'])
    out = property(lambda self: self.refs['out'])


class FullProjAlignPreDS(Projection):
    """Full-chain synaptic projection with the align-pre reduction and delay+synapse updating.

    The ``full-chain`` means that the model needs to provide all information needed for a projection,
    including ``pre`` -> ``syn`` -> ``delay`` -> ``comm`` -> ``out`` -> ``post``.
    Note here, compared to ``FullProjAlignPreSD``, the ``delay`` and ``syn`` are exchanged.

    The ``align-pre`` means that the synaptic variables have the same dimension as the pre-synaptic neuron group.

    The ``delay+synapse updating`` means that the projection first delivers the pre neuron output (usually the
    spiking)  to the delay model, then computes the synapse states, and finally computes the synaptic current.

    Neither ``FullProjAlignPreDS`` nor ``FullProjAlignPreSD`` facilitates the event-driven computation.
    This is because the ``comm`` is computed after the synapse state, which is a floating-point number, rather
    than the spiking. To facilitate the event-driven computation, please use align post projections.


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
            self.E2E = bp.dyn.FullProjAlignPreDS(pre=self.E,
                                                delay=0.1,
                                                syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                comm=bp.dnn.JitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                                out=bp.dyn.COBA(E=0.),
                                                post=self.E)
            self.E2I = bp.dyn.FullProjAlignPreDS(pre=self.E,
                                                delay=0.1,
                                                syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                comm=bp.dnn.JitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                                out=bp.dyn.COBA(E=0.),
                                                post=self.I)
            self.I2E = bp.dyn.FullProjAlignPreDS(pre=self.I,
                                                delay=0.1,
                                                syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                                comm=bp.dnn.JitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                                out=bp.dyn.COBA(E=-80.),
                                                post=self.E)
            self.I2I = bp.dyn.FullProjAlignPreDS(pre=self.I,
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
        delay_cls = register_delay_by_return(pre)
        delay_cls.register_entry(self.name, delay)

        # output initialization
        post.add_inp_fun(self.name, out, label=out_label)

        # references
        self.refs = dict()
        # invisible to ``self.nodes()``
        self.refs['pre'] = pre
        self.refs['post'] = post
        self.refs['out'] = out
        self.refs['delay'] = delay_cls
        # unify the access
        self.refs['syn'] = syn
        self.refs['comm'] = comm

    def update(self):
        spk = self.refs['delay'].at(self.name)
        g = self.comm(self.syn(spk))
        self.refs['out'].bind_cond(g)
        return g

    pre = property(lambda self: self.refs['pre'])
    post = property(lambda self: self.refs['post'])
    delay = property(lambda self: self.refs['delay'])
    out = property(lambda self: self.refs['out'])
