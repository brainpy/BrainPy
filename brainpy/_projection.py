# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from typing import Callable, Union
from typing import Optional

import brainevent

import brainstate
from brainstate._state import State
from brainstate.mixin import JointTypes, ParamDescriber
from brainstate.nn._dynamics import maybe_init_prefetch
from ._synouts import SynOut
from .mixin import BindCondData, AlignPost

if brainstate.__version__ < '0.2.0':
    from brainstate.util.others import get_unique_name
else:
    from brainstate.util import get_unique_name

__all__ = [
    'Projection',
    'AlignPostProj',
    'DeltaProj',
    'CurrentProj',
    'align_pre_projection',
    'align_post_projection',
]


class Projection(brainstate.nn.Module):
    """
    Base class for synaptic projection modules in neural network modeling.

    This class defines the interface for modules that handle projections between
    neural populations. Projections process input signals and transform them
    before they reach the target neurons, implementing the connectivity patterns
    in neural networks.

    In the BrainState execution order, Projection modules are updated before
    Dynamics modules, following the natural information flow in neural systems:
    1. Projections process inputs (synaptic transmission)
    2. Dynamics update neuron states (neural integration)

    The Projection class does not implement the update logic directly but delegates
    to its child nodes. If no child nodes exist, it raises a ValueError.

    Parameters
    ----------
    *args
        Arguments passed to the parent Module class.
    **kwargs
        Keyword arguments passed to the parent Module class.

    Raises
    ------
    ValueError
        If the update() method is called but no child nodes are defined.

    Notes
    -----
    Derived classes should implement specific projection behaviors, such as
    dense connectivity, sparse connectivity, or specific weight update rules.
    """
    __module__ = 'brainstate.nn'

    def update(self, *args, **kwargs):
        sub_nodes = tuple(self.nodes(allowed_hierarchy=(1, 1)).values())
        if len(sub_nodes):
            for node in sub_nodes:
                node(*args, **kwargs)
        else:
            raise ValueError('Do not implement the update() function.')


def _check_modules(*modules):
    # checking modules
    for module in modules:
        if not callable(module) and not isinstance(module, State):
            raise TypeError(
                f'The module should be a callable function or a brainstate.State, but got {module}.'
            )
    return tuple(modules)


def call_module(module, *args, **kwargs):
    if callable(module):
        return module(*args, **kwargs)
    elif isinstance(module, State):
        return module.value
    else:
        raise TypeError(
            f'The module should be a callable function or a brainstate.State, but got {module}.'
        )


def is_instance(x, cls) -> bool:
    return isinstance(x, cls)


def get_post_repr(label, syn, out):
    if label is None:
        return f'{syn.identifier} // {out.identifier}'
    else:
        return f'{label}{syn.identifier} // {out.identifier}'


def align_post_add_bef_update(
    syn_desc: ParamDescriber[AlignPost],
    out_desc: ParamDescriber[BindCondData],
    post: brainstate.nn.Dynamics,
    proj_name: str,
    label: str,
):
    # synapse and output initialization
    _post_repr = get_post_repr(label, syn_desc, out_desc)
    if not post._has_before_update(_post_repr):
        syn_cls = syn_desc()
        out_cls = out_desc()

        # synapse and output initialization
        post.add_current_input(proj_name, out_cls, label=label)
        post._add_before_update(_post_repr, _AlignPost(syn_cls, out_cls))
    syn = post._get_before_update(_post_repr).syn
    out = post._get_before_update(_post_repr).out
    return syn, out


class _AlignPost(brainstate.nn.Module):
    def __init__(
        self,
        syn: brainstate.nn.Dynamics,
        out: BindCondData
    ):
        super().__init__()
        self.syn = syn
        self.out = out

    def update(self, *args, **kwargs):
        self.out.bind_cond(self.syn(*args, **kwargs))


class AlignPostProj(Projection):
    """
    Align-post projection of the neural network.


    Examples
    --------

    Here is an example of using the `AlignPostProj` to create a synaptic projection.
    Note that this projection needs the manual input of pre-synaptic spikes.

    >>> import brainstate
    >>> import brainunit as u
    >>> n_exc = 3200
    >>> n_inh = 800
    >>> num = n_exc + n_inh
    >>> pop = brainstate.nn.LIFRef(
    ...        num,
    ...        V_rest=-49. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
    ...        tau=20. * u.ms, tau_ref=5. * u.ms,
    ...        V_initializer=brainstate.nn.Normal(-55., 2., unit=u.mV)
    ... )
    >>> pop.init_state()
    >>> E = brainstate.nn.AlignPostProj(
    ...        comm=brainstate.nn.FixedNumConn(n_exc, num, prob=80 / num, weight=1.62 * u.mS),
    ...        syn=brainstate.nn.Expon.desc(num, tau=5. * u.ms),
    ...        out=brainstate.nn.CUBA.desc(scale=u.volt),
    ...        post=pop
    ... )
    >>> exe_current = E(pop.get_spike())

    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        *modules,
        comm: Callable,
        syn: Union[ParamDescriber[AlignPost], AlignPost],
        out: Union[ParamDescriber[SynOut], SynOut],
        post: brainstate.nn.Dynamics,
        label: Optional[str] = None,
    ):
        super().__init__(name=get_unique_name(self.__class__.__name__))

        # checking modules
        self.modules = _check_modules(*modules)

        # checking communication model
        if not callable(comm):
            raise TypeError(
                f'The communication should be an instance of callable function, but got {comm}.'
            )

        # checking synapse and output models
        if is_instance(syn, ParamDescriber[AlignPost]):
            if not is_instance(out, ParamDescriber[SynOut]):
                if is_instance(out, ParamDescriber):
                    raise TypeError(
                        f'The output should be an instance of describer {ParamDescriber[SynOut]} when '
                        f'the synapse is an instance of {AlignPost}, but got {out}.'
                    )
                raise TypeError(
                    f'The output should be an instance of describer {ParamDescriber[SynOut]} when '
                    f'the synapse is a describer, but we got {out}.'
                )
            merging = True
        else:
            if is_instance(syn, ParamDescriber):
                raise TypeError(
                    f'The synapse should be an instance of describer {ParamDescriber[AlignPost]}, but got {syn}.'
                )
            if not is_instance(out, SynOut):
                raise TypeError(
                    f'The output should be an instance of {SynOut} when the synapse is '
                    f'not a describer, but we got {out}.'
                )
            merging = False
        self.merging = merging

        # checking post model
        if not is_instance(post, brainstate.nn.Dynamics):
            raise TypeError(
                f'The post should be an instance of {brainstate.nn.Dynamics}, but got {post}.'
            )

        if merging:
            # synapse and output initialization
            syn, out = align_post_add_bef_update(syn_desc=syn,
                                                 out_desc=out,
                                                 post=post,
                                                 proj_name=self.name,
                                                 label=label)
        else:
            post.add_current_input(self.name, out)

        # references
        self.comm = comm
        self.syn: JointTypes[brainstate.nn.Dynamics, AlignPost] = syn
        self.out: BindCondData = out
        self.post: brainstate.nn.Dynamics = post

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        for module in self.modules:
            maybe_init_prefetch(module, *args, **kwargs)

    def update(self, *args):
        # call all modules
        for module in self.modules:
            x = call_module(module, *args)
            args = (x,)
        # communication module
        x = self.comm(*args)
        # add synapse input
        self.syn.add_delta_input(self.name, x)
        if not self.merging:
            # synapse and output interaction
            conductance = self.syn()
            self.out.bind_cond(conductance)


class DeltaProj(Projection):
    """
    Delta-based projection of the neural network.

    This projection directly applies delta inputs to post-synaptic neurons without intervening
    synaptic dynamics. It processes inputs through optional prefetch modules, applies a communication model,
    and adds the result directly as a delta input to the post-synaptic population.

    Parameters
    ----------
    *prefetch
        Optional prefetch modules to process input before communication.
    comm : callable
        Communication model that determines how signals are transmitted.
    post : Dynamics
        Post-synaptic neural population to receive the delta inputs.
    label : Optional[str], default=None
        Optional label for the projection to identify it in the post-synaptic population.

    Examples
    --------
    >>> import brainstate
    >>> import brainunit as u
    >>> n_neurons = 100
    >>> pop = brainstate.nn.LIF(n_neurons, V_rest=-70*u.mV, V_threshold=-50*u.mV)
    >>> pop.init_state()
    >>> delta_input = brainstate.nn.DeltaProj(
    ...     comm=lambda x: x * 10.0*u.mV,
    ...     post=pop
    ... )
    >>> delta_input(1.0)  # Apply voltage increment directly
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        *prefetch,
        comm: Callable,
        post: brainstate.nn.Dynamics,
        label=None,
    ):
        super().__init__(name=get_unique_name(self.__class__.__name__))

        self.label = label

        # checking modules
        self.prefetches = _check_modules(*prefetch)

        # checking communication model
        if not callable(comm):
            raise TypeError(
                f'The communication should be an instance of callable function, but got {comm}.'
            )
        self.comm = comm

        # post model
        if not isinstance(post, brainstate.nn.Dynamics):
            raise TypeError(
                f'The post should be an instance of {brainstate.nn.Dynamics}, but got {post}.'
            )
        self.post = post

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        for prefetch in self.prefetches:
            maybe_init_prefetch(prefetch, *args, **kwargs)

    def update(self, *x):
        for module in self.prefetches:
            x = (call_module(module, *x),)
        assert len(x) == 1, f'The output of the modules should be a single value, but got {x}.'
        x = self.comm(x[0])
        self.post.add_delta_input(self.name, x, label=self.label)


class CurrentProj(Projection):
    """
    Current-based projection of the neural network.

    This projection directly modulates post-synaptic currents without separate synaptic dynamics.
    It processes inputs through optional prefetch modules, applies a communication model,
    and binds the result to the output model which is then added as a current input to the post-synaptic population.

    Parameters
    ----------
    *prefetch
        Optional prefetch modules to process input before communication.
        The last element must be an instance of Prefetch or PrefetchDelayAt if any are provided.
    comm : callable
        Communication model that determines how signals are transmitted.
    out : SynOut
        Output model that converts communication results to post-synaptic currents.
    post : Dynamics
        Post-synaptic neural population to receive the currents.

    Examples
    --------
    >>> import brainstate
    >>> import brainunit as u
    >>> n_neurons = 100
    >>> pop = brainstate.nn.LIF(n_neurons, V_rest=-70*u.mV, V_threshold=-50*u.mV)
    >>> pop.init_state()
    >>> current_input = brainstate.nn.CurrentProj(
    ...     comm=lambda x: x * 0.5,
    ...     out=brainstate.nn.CUBA(scale=1.0*u.nA),
    ...     post=pop
    ... )
    >>> current_input(0.2)  # Apply external current
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        *prefetch,
        comm: Callable,
        out: SynOut,
        post: brainstate.nn.Dynamics,
    ):
        super().__init__(name=get_unique_name(self.__class__.__name__))

        # check prefetch
        self.prefetch = prefetch
        if len(self.prefetch) > 0 and not isinstance(
            prefetch[-1], (brainstate.nn.Prefetch, brainstate.nn.PrefetchDelayAt)
        ):
            raise TypeError(
                f'The last element of prefetch should be an instance '
                f'of {brainstate.nn.Prefetch} or {brainstate.nn.PrefetchDelayAt}, '
                f'but got {prefetch[-1]}.'
            )

        # check out
        if not isinstance(out, SynOut):
            raise TypeError(f'The out should be a SynOut, but got {out}.')
        self.out = out

        # check post
        if not isinstance(post, brainstate.nn.Dynamics):
            raise TypeError(f'The post should be a Dynamics, but got {post}.')
        self.post = post
        post.add_current_input(self.name, out)

        # output initialization
        self.comm = comm

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        for prefetch in self.prefetch:
            maybe_init_prefetch(prefetch, *args, **kwargs)

    def update(self, *x):
        for prefetch in self.prefetch:
            x = (call_module(prefetch, *x),)
        x = self.comm(*x)
        self.out.bind_cond(x)


class align_pre_projection(Projection):
    """
    Represents a pre-synaptic alignment projection mechanism.

    This class inherits from the `Projection` base class and is designed to
    manage the pre-synaptic alignment process in neural network simulations.
    It takes into account pre-synaptic dynamics, synaptic properties, delays,
    communication functions, synaptic outputs, post-synaptic dynamics, and
    short-term plasticity.

    """

    def __init__(
        self,
        *spike_generator,
        syn: brainstate.nn.Dynamics,
        comm: Callable,
        out: SynOut,
        post: brainstate.nn.Dynamics,
        stp: brainstate.nn.Dynamics = None,
    ):
        super().__init__()

        self.spike_generator = _check_modules(*spike_generator)
        self.projection = CurrentProj(comm=comm, out=out, post=post)
        self.syn = syn
        self.stp = stp

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        for module in self.spike_generator:
            maybe_init_prefetch(module, *args, **kwargs)

    def update(self, *x):
        for fun in self.spike_generator:
            x = fun(*x)
            if isinstance(x, (tuple, list)):
                x = tuple(x)
            else:
                x = (x,)
        assert len(x) == 1, "Spike generator must return a single value or a tuple/list of values"
        x = brainevent.BinaryArray(x[0])  # Ensure input is a BinaryFloat for spike generation
        if self.stp is not None:
            x = brainevent.MaskedFloat(self.stp(x))  # Ensure STP output is a MaskedFloat
        x = self.syn(x)  # Apply pre-synaptic alignment
        return self.projection(x)


class align_post_projection(Projection):
    """
    Represents a post-synaptic alignment projection mechanism.

    This class inherits from the `Projection` base class and is designed to
    manage the post-synaptic alignment process in neural network simulations.
    It takes into account spike generators, communication functions, synaptic
    properties, synaptic outputs, post-synaptic dynamics, and short-term plasticity.

    """

    def __init__(
        self,
        *spike_generator,
        comm: Callable,
        syn: Union[AlignPost, ParamDescriber[AlignPost]],
        out: Union[SynOut, ParamDescriber[SynOut]],
        post: brainstate.nn.Dynamics,
        stp: brainstate.nn.Dynamics = None,
    ):
        super().__init__()

        self.spike_generator = _check_modules(*spike_generator)
        self.projection = AlignPostProj(comm=comm, syn=syn, out=out, post=post)
        self.stp = stp

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        for module in self.spike_generator:
            maybe_init_prefetch(module, *args, **kwargs)

    def update(self, *x):
        for fun in self.spike_generator:
            x = fun(*x)
            if isinstance(x, (tuple, list)):
                x = tuple(x)
            else:
                x = (x,)
        assert len(x) == 1, "Spike generator must return a single value or a tuple/list of values"
        x = brainevent.BinaryArray(x[0])  # Ensure input is a BinaryFloat for spike generation
        if self.stp is not None:
            x = brainevent.MaskedFloat(self.stp(x))  # Ensure STP output is a MaskedFloat
        return self.projection(x)
