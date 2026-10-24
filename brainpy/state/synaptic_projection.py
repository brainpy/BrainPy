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
# -*- coding: utf-8 -*-


from collections.abc import Callable

import jax.numpy as jnp
import brainstate
import braintools
import brainunit as u
from brainstate.typing import ArrayLike
from brainstate.util import get_unique_name

from brainpy.state._misc import set_module_as
from brainpy.state.projection import Projection

__all__ = [
    'SymmetryGapJunction',
    'AsymmetryGapJunction',
]


class align_pre_ltp(Projection):
    __module__ = 'brainpy.state'


class align_post_ltp(Projection):
    __module__ = 'brainpy.state'


class SymmetryGapJunction(Projection):
    r"""
    Implements a symmetric electrical coupling (gap junction) between neuron populations.

    This class represents electrical synapses where the conductance is identical in both
    directions. Gap junctions allow bidirectional flow of electrical current directly between
    neurons, with the current magnitude proportional to the voltage difference between
    connected neurons.

    Parameters
    ----------
    couples : Union[Tuple[Dynamics, Dynamics], Dynamics]
        Either a single Dynamics object (when pre and post populations are the same)
        or a tuple of two Dynamics objects (pre, post) representing the coupled neuron populations.
    states : Union[str, Tuple[str, str]]
        Either a single string (when pre and post states are the same)
        or a tuple of two strings (pre_state, post_state) representing the state variables
        to use for calculating voltage differences (typically membrane potentials).
    conn : Callable
        Connection function that returns pre_ids and post_ids arrays defining connections.
    weight : Union[Callable, ArrayLike]
        Conductance weights for the gap junctions. The same weight applies in both directions
        of the connection.
    param_type : type, optional
        The parameter state type to use for weights, defaults to ParamState.

    Notes
    -----
    The symmetric gap junction applies identical conductance in both directions between
    connected neurons, ensuring balanced electrical coupling in the network.

    See Also
    --------
    AsymmetryGapJunction : For gap junctions with different conductances in each direction.

    References
    ----------
    .. [1] Bennett, M. V., & Zukin, R. S. (2004). Electrical coupling and
           neuronal synchronization in the mammalian brain. Neuron, 41(4),
           495-511.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>> # Create two neuron populations
        >>> neurons = brainpy.state.HH(100)
        >>> neurons.init_state(batch_size=1)
        >>> # Create symmetric gap junctions (self-coupling)
        >>> gj = brainpy.state.SymmetryGapJunction(
        ...     couples=neurons,
        ...     states='V',
        ...     conn=lambda pre_size, post_size: ([0,1,2], [1,2,0]),
        ...     weight=0.1 * u.msiemens,
        ... )
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        couples: tuple[brainstate.nn.Dynamics, brainstate.nn.Dynamics] | brainstate.nn.Dynamics,
        states: str | tuple[str, str],
        conn: Callable,
        weight: Callable | ArrayLike,
        param_type: type = brainstate.ParamState
    ):
        super().__init__()

        if isinstance(states, str):
            pre_state = states
            post_state = states
        else:
            pre_state, post_state = states
        if isinstance(couples, brainstate.nn.Dynamics):
            pre = couples
            post = couples
        else:
            pre, post = couples
        assert isinstance(pre_state, str), "pre_state must be a string representing the pre-synaptic state"
        assert isinstance(post_state, str), "post_state must be a string representing the post-synaptic state"
        assert isinstance(pre, brainstate.nn.Dynamics), "pre must be a Dynamics object"
        assert isinstance(post, brainstate.nn.Dynamics), "post must be a Dynamics object"
        self.pre_state = pre_state
        self.post_state = post_state
        self.pre = pre
        self.post = post
        self.pre_ids, self.post_ids = conn(pre.out_size, post.out_size)
        self.weight = param_type(braintools.init.param(weight, (len(self.pre_ids),)))
        # Stable per-instance input keys so repeated updates target the same slot
        # instead of leaking a fresh, count-derived slot on each call.
        self._post_key = get_unique_name('gap_junction_post')
        self._pre_key = get_unique_name('gap_junction_pre')

    def update(self, *args, **kwargs):
        if not hasattr(self.pre, self.pre_state):
            raise ValueError(f"pre_state {self.pre_state} not found in pre-synaptic neuron group")
        if not hasattr(self.post, self.post_state):
            raise ValueError(f"post_state {self.post_state} not found in post-synaptic neuron group")
        pre = getattr(self.pre, self.pre_state).value
        post = getattr(self.post, self.post_state).value

        return symmetry_gap_junction_projection(
            pre=self.pre,
            pre_value=pre,
            post=self.post,
            post_value=post,
            pre_ids=self.pre_ids,
            post_ids=self.post_ids,
            weight=self.weight.value,
            post_key=self._post_key,
            pre_key=self._pre_key,
        )


@set_module_as('brainpy.state')
def symmetry_gap_junction_projection(
    pre: brainstate.nn.Dynamics,
    pre_value: ArrayLike,
    post: brainstate.nn.Dynamics,
    post_value: ArrayLike,
    pre_ids: ArrayLike,
    post_ids: ArrayLike,
    weight: ArrayLike,
    post_key: str = 'gap_junction_post',
    pre_key: str = 'gap_junction_pre',
):
    r"""
    Calculate symmetrical electrical coupling through gap junctions between neurons.

    This function implements bidirectional gap junction coupling where the same weight is
    applied in both directions. It computes the electrical current flowing between
    connected neurons based on their potential differences and updates both pre-synaptic
    and post-synaptic neuron groups with appropriate input currents.

    Parameters
    ----------
    pre : Dynamics
        The pre-synaptic neuron group dynamics object.
    pre_value : ArrayLike
        State values (typically membrane potentials) of the pre-synaptic neurons.
    post : Dynamics
        The post-synaptic neuron group dynamics object.
    post_value : ArrayLike
        State values (typically membrane potentials) of the post-synaptic neurons.
    pre_ids : ArrayLike
        Indices of pre-synaptic neurons that form gap junctions.
    post_ids : ArrayLike
        Indices of post-synaptic neurons that form gap junctions,
        where each pre_ids[i] is connected to post_ids[i].
    weight : ArrayLike
        Conductance weights for the gap junctions. Can be a scalar (same weight for all connections)
        or an array with length matching pre_ids.

    Returns
    -------
    ArrayLike
        The input currents that were added to the pre-synaptic neuron group.

    Notes
    -----
    The electrical coupling is implemented as I = g(V_pre - V_post), where:
    - I is the current flowing from pre to post neuron
    - g is the gap junction conductance (weight)
    - V_pre and V_post are the membrane potentials of connected neurons

    Equal but opposite currents are applied to both connected neurons, ensuring
    conservation of current in the network.

    Raises
    ------
    AssertionError
        If weight dimensionality is incorrect or pre_ids and post_ids have different lengths.
    """
    assert u.math.ndim(weight) == 0 or weight.shape[0] == len(pre_ids), \
        "weight must be a scalar or have the same length as pre_ids"
    assert len(pre_ids) == len(post_ids), "pre_ids and post_ids must have the same length"
    # Calculate the voltage difference between connected pre-synaptic and post-synaptic neurons
    # and multiply by the connection weights
    # Use `...` indexing to support both batched (batch, neurons) and unbatched (neurons,) values
    diff = (pre_value[..., pre_ids] - post_value[..., post_ids]) * weight

    # Decompose into mantissa and unit for scatter-add
    # (diff has current units, e.g. μA, while value arrays have voltage units)
    diff_mantissa = diff.mantissa if hasattr(diff, 'mantissa') else diff
    diff_unit = diff.unit if hasattr(diff, 'unit') else None
    post_shape = post_value.mantissa.shape if hasattr(post_value, 'mantissa') else post_value.shape
    pre_shape = pre_value.mantissa.shape if hasattr(pre_value, 'mantissa') else pre_value.shape

    # add to post-synaptic neuron group
    post_inputs = jnp.zeros(post_shape)
    post_inputs = post_inputs.at[..., post_ids].add(diff_mantissa)
    if diff_unit is not None:
        post_inputs = post_inputs * diff_unit
    post.add_current_input(post_key, post_inputs)

    # add to pre-synaptic neuron group
    pre_inputs = jnp.zeros(pre_shape)
    pre_inputs = pre_inputs.at[..., pre_ids].add(diff_mantissa)
    if diff_unit is not None:
        pre_inputs = pre_inputs * diff_unit
    pre.add_current_input(pre_key, -pre_inputs)
    return pre_inputs


class AsymmetryGapJunction(Projection):
    r"""
    Implements an asymmetric electrical coupling (gap junction) between neuron populations.

    This class represents electrical synapses where the conductance in one direction can differ
    from the conductance in the opposite direction. Unlike chemical synapses, gap junctions
    allow bidirectional flow of electrical current directly between neurons, with the current
    magnitude proportional to the voltage difference between connected neurons.

    Parameters
    ----------
    pre : Dynamics
        The pre-synaptic neuron group dynamics object.
    pre_state : str
        Name of the state variable in pre-synaptic neurons (typically membrane potential).
    post : Dynamics
        The post-synaptic neuron group dynamics object.
    post_state : str
        Name of the state variable in post-synaptic neurons (typically membrane potential).
    conn : Callable
        Connection function that returns pre_ids and post_ids arrays defining connections.
    weight : Union[Callable, ArrayLike]
        Conductance weights for the gap junctions. Must have shape [..., 2] where the last
        dimension contains [pre_weight, post_weight] for each connection, defining
        potentially different conductances in each direction.
    param_type : type, optional
        The parameter state type to use for weights, defaults to ParamState.

    See Also
    --------
    SymmetryGapJunction : For gap junctions with identical conductance in both directions.

    Notes
    -----
    The asymmetric gap junction allows for different conductances in each direction between
    the same pair of neurons. This can model rectifying electrical synapses that preferentially
    allow current to flow in one direction.

    References
    ----------
    .. [1] Bennett, M. V., & Zukin, R. S. (2004). Electrical coupling and
           neuronal synchronization in the mammalian brain. Neuron, 41(4),
           495-511.
    .. [2] Phelan, P. (2005). Innexins: members of an evolutionarily
           conserved family of gap-junction proteins. Biochimica et
           Biophysica Acta (BBA)-Biomembranes, 1711(2), 225-245.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as brainpy
        >>> import brainunit as u
        >>> import numpy as np
        >>>
        >>> # Create two neuron populations
        >>> n_neurons = 100
        >>> pre_pop = brainpy.LIF(n_neurons, V_rest=-70*u.mV, V_threshold=-50*u.mV)
        >>> post_pop = brainpy.LIF(n_neurons, V_rest=-70*u.mV, V_threshold=-50*u.mV)
        >>> pre_pop.init_state()
        >>> post_pop.init_state()
        >>>
        >>> # Create asymmetric gap junction with different weights in each direction
        >>> weights = np.ones((n_neurons, 2)) * u.nS
        >>> weights[:, 0] *= 2.0  # Double weight in pre->post direction
        >>>
        >>> gap_junction = brainpy.AsymmetryGapJunction(
        ...     pre=pre_pop,
        ...     pre_state='V',
        ...     post=post_pop,
        ...     post_state='V',
        ...     conn=one_to_one,
        ...     weight=weights
        ... )
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        pre: brainstate.nn.Dynamics,
        pre_state: str,
        post: brainstate.nn.Dynamics,
        post_state: str,
        conn: Callable,
        weight: Callable | ArrayLike,
        param_type: type = brainstate.ParamState
    ):
        super().__init__()

        assert isinstance(pre_state, str), "pre_state must be a string representing the pre-synaptic state"
        assert isinstance(post_state, str), "post_state must be a string representing the post-synaptic state"
        self.pre_state = pre_state
        self.post_state = post_state
        self.pre = pre
        self.post = post
        self.pre_ids, self.post_ids = conn(pre.out_size, post.out_size)
        self.weight = param_type(braintools.init.param(weight, (len(self.pre_ids), 2)))
        # Stable per-instance input keys so repeated updates target the same slot
        # instead of leaking a fresh, count-derived slot on each call.
        self._post_key = get_unique_name('gap_junction_post')
        self._pre_key = get_unique_name('gap_junction_pre')

    def update(self, *args, **kwargs):
        if not hasattr(self.pre, self.pre_state):
            raise ValueError(f"pre_state {self.pre_state} not found in pre-synaptic neuron group")
        if not hasattr(self.post, self.post_state):
            raise ValueError(f"post_state {self.post_state} not found in post-synaptic neuron group")
        pre = getattr(self.pre, self.pre_state).value
        post = getattr(self.post, self.post_state).value

        return asymmetry_gap_junction_projection(
            pre=self.pre,
            pre_value=pre,
            post=self.post,
            post_value=post,
            pre_ids=self.pre_ids,
            post_ids=self.post_ids,
            weight=self.weight.value,
            post_key=self._post_key,
            pre_key=self._pre_key,
        )


@set_module_as('brainpy.state')
def asymmetry_gap_junction_projection(
    pre: brainstate.nn.Dynamics,
    pre_value: ArrayLike,
    post: brainstate.nn.Dynamics,
    post_value: ArrayLike,
    pre_ids: ArrayLike,
    post_ids: ArrayLike,
    weight: ArrayLike,
    post_key: str = 'gap_junction_post',
    pre_key: str = 'gap_junction_pre',
):
    r"""
    Calculate asymmetrical electrical coupling through gap junctions between neurons.

    This function implements bidirectional gap junction coupling where different weights
    can be applied in each direction. It computes the electrical current flowing between
    connected neurons based on their potential differences and updates both pre-synaptic
    and post-synaptic neuron groups with appropriate input currents.

    Parameters
    ----------
    pre : Dynamics
        The pre-synaptic neuron group dynamics object.
    pre_value : ArrayLike
        State values (typically membrane potentials) of the pre-synaptic neurons.
    post : Dynamics
        The post-synaptic neuron group dynamics object.
    post_value : ArrayLike
        State values (typically membrane potentials) of the post-synaptic neurons.
    pre_ids : ArrayLike
        Indices of pre-synaptic neurons that form gap junctions.
    post_ids : ArrayLike
        Indices of post-synaptic neurons that form gap junctions,
        where each pre_ids[i] is connected to post_ids[i].
    weight : ArrayLike
        Conductance weights for the gap junctions. Must have shape [..., 2], where
        the last dimension contains [pre_weight, post_weight] for each connection.
        Can be a 1D array [pre_weight, post_weight] (same weights for all connections)
        or a 2D array with shape [len(pre_ids), 2] for connection-specific weights.

    Returns
    -------
    ArrayLike
        The input currents that were added to the pre-synaptic neuron group.

    Notes
    -----
    The electrical coupling is implemented with direction-specific conductances:
    - I_pre2post = g_pre * (V_pre - V_post) flowing from pre to post neuron
    - I_post2pre = g_post * (V_pre - V_post) flowing from post to pre neuron
    where g_pre and g_post can be different, allowing for asymmetrical coupling.

    Raises
    ------
    AssertionError
        If weight dimensionality is incorrect or pre_ids and post_ids have different lengths.
    ValueError
        If weight shape is incompatible with asymmetrical gap junction requirements.
    """
    assert weight.shape[-1] == 2, 'weight must be a 2-element array for asymmetry gap junctions'
    assert len(pre_ids) == len(post_ids), "pre_ids and post_ids must have the same length"
    if u.math.ndim(weight) == 1:
        # If weight is a 1D array, it should have two elements for pre and post weights
        assert weight.shape[0] == 2, "weight must be a 2-element array for asymmetry gap junctions"
        pre_weight = weight[0]
        post_weight = weight[1]
    elif u.math.ndim(weight) == 2:
        # If weight is a 2D array, it should have two rows for pre and post weights
        pre_weight = weight[:, 0]
        post_weight = weight[:, 1]
        assert pre_weight.shape[0] == len(pre_ids), "pre_weight must have the same length as pre_ids"
        assert post_weight.shape[0] == len(post_ids), "post_weight must have the same length as post_ids"
    else:
        raise ValueError("weight must be a 1D or 2D array for asymmetry gap junctions")

    # Calculate the voltage difference between connected pre-synaptic and post-synaptic neurons
    # and multiply by the connection weights
    # Use `...` indexing to support both batched (batch, neurons) and unbatched (neurons,) values
    diff = pre_value[..., pre_ids] - post_value[..., post_ids]
    pre2post_current = diff * pre_weight
    post2pre_current = diff * post_weight

    # Decompose into mantissa and unit for scatter-add
    pre2post_mantissa = pre2post_current.mantissa if hasattr(pre2post_current, 'mantissa') else pre2post_current
    post2pre_mantissa = post2pre_current.mantissa if hasattr(post2pre_current, 'mantissa') else post2pre_current
    current_unit = pre2post_current.unit if hasattr(pre2post_current, 'unit') else None
    post_shape = post_value.mantissa.shape if hasattr(post_value, 'mantissa') else post_value.shape
    pre_shape = pre_value.mantissa.shape if hasattr(pre_value, 'mantissa') else pre_value.shape

    # add to post-synaptic neuron group
    post_inputs = jnp.zeros(post_shape)
    post_inputs = post_inputs.at[..., post_ids].add(pre2post_mantissa)
    if current_unit is not None:
        post_inputs = post_inputs * current_unit
    post.add_current_input(post_key, post_inputs)

    # add to pre-synaptic neuron group
    pre_inputs = jnp.zeros(pre_shape)
    pre_inputs = pre_inputs.at[..., pre_ids].add(post2pre_mantissa)
    if current_unit is not None:
        pre_inputs = pre_inputs * current_unit
    pre.add_current_input(pre_key, -pre_inputs)
    return pre_inputs
