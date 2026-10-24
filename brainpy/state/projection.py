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

from collections.abc import Callable
from typing import Union

import brainevent
import brainstate
import brainunit as u
from brainstate import State
from brainstate.mixin import JointTypes, ParamDescriber
from brainstate.nn import init_maybe_prefetch
from brainstate.typing import ArrayLike
from brainstate.util import get_unique_name

from brainpy.state._base import Dynamics
from brainpy.state._mixin import BindCondData, AlignPost
from brainpy.state.delay import InputDelay
from brainpy.state.synouts import SynOut

__all__ = [
    'Projection',
    'AlignPostProj',
    'DeltaProj',
    'CurrentProj',
    'align_pre_projection',
    'align_post_projection',
]


class Projection(brainstate.nn.Module):
    r"""
    Base class for synaptic projection modules.

    A projection connects pre-synaptic and post-synaptic neural populations,
    handling synaptic transmission, weight application, and input delivery.
    In the BrainState execution order, projections are updated *before*
    dynamics modules, following the natural information flow: projections
    process inputs first, then neurons integrate.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :class:`brainstate.nn.Module`.
    **kwargs
        Keyword arguments forwarded to :class:`brainstate.nn.Module`.

    Raises
    ------
    ValueError
        If :meth:`update` is called but no child nodes are defined.

    See Also
    --------
    AlignPostProj : Post-synaptic alignment projection.
    DeltaProj : Delta-input projection (direct voltage changes).
    CurrentProj : Current-based projection.
    align_pre_projection : Pre-synaptic alignment convenience wrapper.
    align_post_projection : Post-synaptic alignment convenience wrapper.

    Notes
    -----
    Subclasses typically compose a communication module (connection weights),
    a synapse model, and a synaptic output module. The base class delegates
    its :meth:`update` call to child nodes registered via the module tree.
    """
    __module__ = 'brainpy.state'

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
    post: Dynamics,
    proj_name: str,
    label: str | None,
):
    # synapse and output initialization
    _post_repr = get_post_repr(label, syn_desc, out_desc)
    if not post.has_before_update(_post_repr):
        syn_cls = syn_desc()
        out_cls = out_desc()

        # synapse and output initialization
        post.add_current_input(proj_name, out_cls, label=label)
        post.add_before_update(_post_repr, _AlignPost(syn_cls, out_cls))
    syn = post.get_before_update(_post_repr).syn
    out = post.get_before_update(_post_repr).out
    return syn, out


class _AlignPost(brainstate.nn.Module):
    def __init__(
        self,
        syn: Dynamics,
        out: BindCondData
    ):
        super().__init__()
        self.syn = syn
        self.out = out

    def update(self, *args, **kwargs):
        self.out.bind_cond(self.syn(*args, **kwargs))


class AlignPostProj(Projection):
    r"""
    Post-synaptic alignment projection.

    In this projection pattern, the synapse dynamics and synaptic output
    are aligned with (owned by) the post-synaptic neuron. Multiple
    projections targeting the same post-synaptic population with the same
    synapse/output descriptor share a single synapse and output instance,
    enabling efficient event-driven updates.

    The update pipeline is:

    1. Optional pre-processing modules transform the input.
    2. The communication module (``comm``) maps pre-synaptic signals
       to post-synaptic space.
    3. The result is added as a delta input to the shared synapse.
    4. The synapse and output are updated by the post-synaptic neuron's
       ``before_update`` hook (if using descriptor merging).

    Parameters
    ----------
    *modules
        Optional pre-processing modules applied sequentially to the input
        before the communication step.
    comm : Callable
        Communication module (e.g., ``brainevent.nn.FixedProb``) that
        maps pre-synaptic activity to post-synaptic space.
    syn : ParamDescriber[AlignPost] or AlignPost
        Synapse model or its descriptor. When a descriptor is provided,
        the synapse is created lazily and shared across projections
        targeting the same post-synaptic neuron.
    out : ParamDescriber[SynOut] or SynOut
        Synaptic output module or its descriptor.
    post : Dynamics
        Post-synaptic neural population.
    label : str, optional
        Label for identifying this projection's contribution in the
        post-synaptic neuron's input dictionary.
    delay : ArrayLike or Quantity, optional
        Conduction delay applied to the pre-synaptic input. ``None`` (default)
        keeps the original, zero-overhead code path. A scalar time
        (e.g. ``1.5 * u.ms``) applies one global/homogeneous delay; a
        ``(N_pre,)`` array applies an axonal, per-pre-neuron delay. Sub-``dt``
        delays are linearly interpolated. The buffer is sized from
        ``ceil(max(delay) / dt)`` at :meth:`init_state`, so ``dt`` must not
        change afterwards.

    Raises
    ------
    TypeError
        If ``comm`` is not callable, if ``syn``/``out`` types are
        inconsistent, or if ``post`` is not a :class:`Dynamics` instance.

    See Also
    --------
    DeltaProj : Direct delta-input projection.
    CurrentProj : Current-based projection.
    align_post_projection : Convenience wrapper with spike generation.

    Notes
    -----
    - When both ``syn`` and ``out`` are descriptors (``ParamDescriber``),
      the projection attempts to merge with existing synapse/output
      instances on the post-synaptic neuron, avoiding duplicate state.
    - When ``syn`` is an already-instantiated ``AlignPost`` object, no
      merging occurs and ``out`` must also be an instantiated ``SynOut``.

    References
    ----------
    .. [1] Brette, R., et al. (2007). Simulation of networks of spiking
           neurons: a review of tools and strategies. Journal of
           Computational Neuroscience, 23(3), 349-398.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>> n_pre, n_post = 800, 200
        >>> post_pop = brainpy.state.LIF(n_post, tau=20.*u.ms)
        >>> post_pop.init_state()
        >>> proj = brainpy.state.AlignPostProj(
        ...     comm=brainstate.nn.Linear(n_pre, n_post),
        ...     syn=brainpy.state.Expon.desc(n_post, tau=5.*u.ms),
        ...     out=brainpy.state.CUBA.desc(scale=u.volt),
        ...     post=post_pop,
        ... )
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        *modules,
        comm: brainstate.nn.Module,
        # Union (not `|`): ParamDescriber[...] is a runtime instance, and
        # `instance | type` raises TypeError when this annotation is evaluated.
        syn: Union[ParamDescriber[AlignPost], AlignPost],
        out: Union[ParamDescriber[SynOut], SynOut],
        post: Dynamics,
        label: str | None = None,
        delay: ArrayLike | u.Quantity | None = None,
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
        if not is_instance(post, Dynamics):
            raise TypeError(
                f'The post should be an instance of {Dynamics}, but got {post}.'
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
        self.syn: JointTypes[Dynamics, AlignPost] = syn
        self.out: BindCondData = out
        self.post: Dynamics = post

        # optional axonal/global delay over the pre-synaptic input to ``comm``
        self.delay = delay
        self.delay_seam = InputDelay(comm.in_size, delay) if delay is not None else None

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        for module in self.modules:
            init_maybe_prefetch(module, *args, **kwargs)

    def update(self, *args):
        # call all modules
        for module in self.modules:
            x = call_module(module, *args)
            args = (x,)
        # axonal/global delay on the pre-synaptic input
        if self.delay_seam is not None:
            args = (self.delay_seam.update(args[0]),)
        # communication module
        x = self.comm(*args)
        # add synapse input
        self.syn.add_delta_input(self.name, x)
        if not self.merging:
            # synapse and output interaction
            conductance = self.syn()
            self.out.bind_cond(conductance)


class DeltaProj(Projection):
    r"""
    Delta-input projection.

    Applies pre-synaptic signals directly as delta (voltage) inputs to the
    post-synaptic population, bypassing synapse dynamics entirely. Useful
    for instantaneous coupling or simplified network models.

    The update pipeline is:

    1. Optional pre-fetch modules transform the input.
    2. The communication module (``comm``) maps the signal to
       post-synaptic space.
    3. The result is added as a delta input to the post-synaptic neuron's
       membrane potential.

    Parameters
    ----------
    *prefetch
        Optional modules applied sequentially to the input before
        the communication step.
    comm : Callable
        Communication module mapping pre-synaptic activity to
        post-synaptic delta inputs.
    post : Dynamics
        Post-synaptic neural population.
    label : str, optional
        Label for identifying this projection's delta input.

    Raises
    ------
    TypeError
        If ``comm`` is not callable or ``post`` is not a
        :class:`Dynamics` instance.

    See Also
    --------
    AlignPostProj : Projection with full synapse dynamics.
    CurrentProj : Current-based projection.

    Notes
    -----
    - Delta projections add directly to the membrane potential rather
      than to the current, making them suitable for modeling gap junctions
      or abstract coupling.
    - The ``label`` parameter allows multiple delta projections to coexist
      on the same post-synaptic population with distinct labels.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>> n_neurons = 100
        >>> pop = brainpy.state.LIF(n_neurons, tau=10.*u.ms)
        >>> pop.init_state()
        >>> delta_proj = brainpy.state.DeltaProj(
        ...     comm=brainstate.nn.Linear(n_neurons, n_neurons),
        ...     post=pop,
        ... )
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        *prefetch,
        comm: Callable,
        post: Dynamics,
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
        if not isinstance(post, Dynamics):
            raise TypeError(
                f'The post should be an instance of {Dynamics}, but got {post}.'
            )
        self.post = post

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        for prefetch in self.prefetches:
            init_maybe_prefetch(prefetch, *args, **kwargs)

    def update(self, *x):
        for module in self.prefetches:
            x = (call_module(module, *x),)
        assert len(x) == 1, f'The output of the modules should be a single value, but got {x}.'
        x = self.comm(x[0])
        self.post.add_delta_input(self.name, x, label=self.label)


class CurrentProj(Projection):
    r"""
    Current-based projection.

    Delivers current input to post-synaptic neurons by passing the
    communication output through a synaptic output module (e.g.,
    :class:`COBA` or :class:`CUBA`) and registering it as a current
    input on the post-synaptic population.

    The update pipeline is:

    1. Optional pre-fetch modules transform the input.
    2. The communication module (``comm``) maps pre-synaptic activity.
    3. The synaptic output module (``out``) converts conductance to
       current and binds it to the post-synaptic neuron.

    Parameters
    ----------
    *prefetch
        Optional pre-fetch modules. If provided, the last element must
        be a :class:`brainstate.nn.Prefetch` or
        :class:`brainstate.nn.PrefetchDelayAt` instance.
    comm : Callable
        Communication module mapping pre-synaptic activity.
    out : SynOut
        Synaptic output module that converts the communication result
        to post-synaptic current.
    post : Dynamics
        Post-synaptic neural population.
    delay : ArrayLike or Quantity, optional
        Conduction delay applied to the pre-synaptic input. ``None`` (default)
        keeps the original, zero-overhead code path. A scalar time
        (e.g. ``1.5 * u.ms``) applies one global/homogeneous delay; a
        ``(N_pre,)`` array applies an axonal, per-pre-neuron delay. Sub-``dt``
        delays are linearly interpolated. The buffer is sized from
        ``ceil(max(delay) / dt)`` at :meth:`init_state`, so ``dt`` must not
        change afterwards.

    Raises
    ------
    TypeError
        If ``comm`` is not callable, ``out`` is not a :class:`SynOut`,
        ``post`` is not a :class:`Dynamics`, or the last prefetch module
        has an incorrect type.

    See Also
    --------
    AlignPostProj : Projection with aligned synapse dynamics.
    DeltaProj : Direct delta-input projection.
    align_pre_projection : Convenience wrapper with pre-synaptic alignment.

    Notes
    -----
    - The output module is immediately registered as a current input on
      the post-synaptic population at construction time.
    - Unlike :class:`AlignPostProj`, this projection does not use synapse
      dynamics -- the communication result is directly converted to current.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>> n_neurons = 100
        >>> pop = brainpy.state.LIF(n_neurons, tau=10.*u.ms)
        >>> pop.init_state()
        >>> proj = brainpy.state.CurrentProj(
        ...     comm=brainstate.nn.Linear(n_neurons, n_neurons),
        ...     out=brainpy.state.CUBA(scale=u.volt),
        ...     post=pop,
        ... )
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        *prefetch,
        comm: brainstate.nn.Module,
        out: SynOut,
        post: Dynamics,
        delay: ArrayLike | u.Quantity | None = None,
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
        if not isinstance(post, Dynamics):
            raise TypeError(f'The post should be a Dynamics, but got {post}.')
        self.post = post
        post.add_current_input(self.name, out)

        # output initialization
        self.comm = comm

        # optional axonal/global delay over the pre-synaptic input to ``comm``
        self.delay = delay
        self.delay_seam = InputDelay(comm.in_size, delay) if delay is not None else None

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        for prefetch in self.prefetch:
            init_maybe_prefetch(prefetch, *args, **kwargs)

    def update(self, *x):
        for prefetch in self.prefetch:
            x = (call_module(prefetch, *x),)
        # axonal/global delay on the pre-synaptic input
        if self.delay_seam is not None:
            x = (self.delay_seam.update(x[0]),)
        x = self.comm(*x)
        self.out.bind_cond(x)


class align_pre_projection(Projection):
    r"""
    Pre-synaptic alignment projection with spike generation.

    A convenience wrapper that combines spike generation, optional
    short-term plasticity (STP), pre-synaptic synapse dynamics, and
    a :class:`CurrentProj` into a single module. The synapse operates
    in pre-synaptic space, processing spikes before the communication
    step transmits them to post-synaptic neurons.

    The update pipeline is:

    1. Spike generator modules produce binary spike signals.
    2. If STP is provided, spikes are modulated by short-term
       plasticity dynamics.
    3. The pre-synaptic synapse model filters the (modulated) spikes.
    4. The :class:`CurrentProj` maps the filtered signal to post-synaptic
       current.

    Parameters
    ----------
    *spike_generator
        One or more modules that produce spike signals from the input.
    syn : Dynamics
        Pre-synaptic synapse dynamics (e.g., :class:`Expon`).
    comm : Callable
        Communication module mapping pre-synaptic to post-synaptic space.
    out : SynOut
        Synaptic output module.
    post : Dynamics
        Post-synaptic neural population.
    stp : Dynamics, optional
        Short-term plasticity module applied after spike generation.

    See Also
    --------
    align_post_projection : Post-synaptic alignment variant.
    CurrentProj : Underlying current projection used internally.
    AlignPostProj : Alternative projection with post-synaptic alignment.

    Notes
    -----
    - Pre-synaptic alignment means the synapse state lives in
      pre-synaptic space. This is natural for models where synaptic
      filtering (e.g., exponential decay) should happen before the
      communication step.
    - Spike signals are wrapped in ``brainevent.BinaryArray`` for
      efficient event-driven processing.
    - When STP is used, its output is wrapped in
      ``brainevent.MaskedFloat`` to preserve sparsity.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>> pre = brainpy.state.LIF(800, tau=20.*u.ms)
        >>> post = brainpy.state.LIF(200, tau=20.*u.ms)
        >>> pre.init_state()
        >>> post.init_state()
        >>> proj = brainpy.state.align_pre_projection(
        ...     pre,
        ...     syn=brainpy.state.Expon(800, tau=5.*u.ms),
        ...     comm=brainstate.nn.Linear(800, 200),
        ...     out=brainpy.state.CUBA(scale=u.volt),
        ...     post=post,
        ... )
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        *spike_generator,
        syn: Dynamics,
        comm: Callable,
        out: SynOut,
        post: Dynamics,
        stp: Dynamics | None = None,
    ):
        super().__init__()

        self.spike_generator = _check_modules(*spike_generator)
        self.projection = CurrentProj(comm=comm, out=out, post=post)
        self.syn = syn
        self.stp = stp

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        for module in self.spike_generator:
            init_maybe_prefetch(module, *args, **kwargs)

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
    r"""
    Post-synaptic alignment projection with spike generation.

    A convenience wrapper that combines spike generation, optional
    short-term plasticity (STP), and an :class:`AlignPostProj` into a
    single module. The synapse operates in post-synaptic space, sharing
    state across projections that target the same post-synaptic neuron.

    The update pipeline is:

    1. Spike generator modules produce binary spike signals.
    2. If STP is provided, spikes are modulated by short-term
       plasticity dynamics.
    3. The :class:`AlignPostProj` handles communication and
       post-aligned synapse/output updates.

    Parameters
    ----------
    *spike_generator
        One or more modules that produce spike signals from the input.
    comm : Callable
        Communication module mapping pre-synaptic to post-synaptic space.
    syn : AlignPost or ParamDescriber[AlignPost]
        Post-synaptic synapse model or its descriptor.
    out : SynOut or ParamDescriber[SynOut]
        Synaptic output module or its descriptor.
    post : Dynamics
        Post-synaptic neural population.
    stp : Dynamics, optional
        Short-term plasticity module applied after spike generation.

    See Also
    --------
    align_pre_projection : Pre-synaptic alignment variant.
    AlignPostProj : Underlying post-aligned projection used internally.

    Notes
    -----
    - Post-synaptic alignment enables synapse state sharing: if multiple
      projections target the same post-synaptic population with the same
      synapse/output descriptor, they share a single synapse instance.
    - Spike signals are wrapped in ``brainevent.BinaryArray`` for
      efficient event-driven processing.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>> pre = brainpy.state.LIF(800, tau=20.*u.ms)
        >>> post = brainpy.state.LIF(200, tau=20.*u.ms)
        >>> pre.init_state()
        >>> post.init_state()
        >>> proj = brainpy.state.align_post_projection(
        ...     pre,
        ...     comm=brainstate.nn.Linear(800, 200),
        ...     syn=brainpy.state.Expon.desc(200, tau=5.*u.ms),
        ...     out=brainpy.state.CUBA.desc(scale=u.volt),
        ...     post=post,
        ... )
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        *spike_generator,
        comm: Callable,
        # Union (not `|`): ParamDescriber[...] is a runtime instance, and
        # `instance | type` raises TypeError when this annotation is evaluated.
        syn: Union[AlignPost, ParamDescriber[AlignPost]],
        out: Union[SynOut, ParamDescriber[SynOut]],
        post: Dynamics,
        stp: Dynamics | None = None,
    ):
        super().__init__()

        self.spike_generator = _check_modules(*spike_generator)
        self.projection = AlignPostProj(comm=comm, syn=syn, out=out, post=post)
        self.stp = stp

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        for module in self.spike_generator:
            init_maybe_prefetch(module, *args, **kwargs)

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
