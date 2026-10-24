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


from typing import Any

import brainstate

__all__ = [
    'AlignPost',
    'BindCondData',
]


class AlignPost(brainstate.mixin.Mixin):
    r"""
    Mixin for aligning post-synaptic inputs.

    This mixin provides an interface for components that need to receive and
    process post-synaptic inputs, such as synaptic connections or neural
    populations. The ``align_post_input_add`` method should be implemented
    to handle the accumulation of external currents or inputs.

    Notes
    -----
    Classes that inherit from this mixin must implement the
    ``align_post_input_add`` method.

    Examples
    --------
    Implementing a synapse with post-synaptic alignment:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> class Synapse(brainstate.mixin.AlignPost):
        ...     def __init__(self, weight):
        ...         self.weight = weight
        ...         self.post_current = brainstate.State(0.0)
        ...
        ...     def align_post_input_add(self, current):
        ...         # Accumulate the weighted current into post-synaptic target
        ...         self.post_current.value += current * self.weight
        >>>
        >>> # Usage
        >>> synapse = Synapse(weight=0.5)
        >>> synapse.align_post_input_add(10.0)
        >>> print(synapse.post_current.value)  # Output: 5.0

    Using with neural populations:

    .. code-block:: python

        >>> class NeuronGroup(brainstate.mixin.AlignPost):
        ...     def __init__(self, size):
        ...         self.size = size
        ...         self.input_current = brainstate.State(jnp.zeros(size))
        ...
        ...     def align_post_input_add(self, current):
        ...         # Add external current to neurons
        ...         self.input_current.value = self.input_current.value + current
        >>>
        >>> neurons = NeuronGroup(100)
        >>> external_input = jnp.ones(100) * 0.5
        >>> neurons.align_post_input_add(external_input)
    """

    def align_post_input_add(self, *args, **kwargs):
        r"""
        Add external inputs to the post-synaptic component.

        Parameters
        ----------
        *args
            Positional arguments for the input.
        **kwargs
            Keyword arguments for the input.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError


class BindCondData(brainstate.mixin.Mixin):
    r"""
    Mixin for binding temporary conductance data.

    This mixin provides an interface for temporarily storing conductance data,
    which is useful in synaptic models where conductance values need to be
    passed between computation steps without being part of the permanent state.

    Attributes
    ----------
    _conductance : Any, optional
        Temporarily bound conductance data.

    Examples
    --------
    Using conductance binding in a synapse:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> class ConductanceBasedSynapse(brainstate.mixin.BindCondData):
        ...     def __init__(self):
        ...         self._conductance = None
        ...
        ...     def compute(self, pre_spike):
        ...         if pre_spike:
        ...             # Bind conductance data temporarily
        ...             self.bind_cond(0.5)
        ...
        ...         # Use conductance if available
        ...         if self._conductance is not None:
        ...             current = self._conductance * (0.0 - (-70.0))
        ...             # Clear after use
        ...             self.unbind_cond()
        ...             return current
        ...         return 0.0
        >>>
        >>> synapse = ConductanceBasedSynapse()
        >>> current = synapse.compute(pre_spike=True)

    Managing conductance in a network:

    .. code-block:: python

        >>> class SynapticConnection(brainstate.mixin.BindCondData):
        ...     def __init__(self, g_max):
        ...         self.g_max = g_max
        ...         self._conductance = None
        ...
        ...     def prepare_conductance(self, activation):
        ...         # Bind conductance based on activation
        ...         g = self.g_max * activation
        ...         self.bind_cond(g)
        ...
        ...     def apply_conductance(self, voltage):
        ...         if self._conductance is not None:
        ...             current = self._conductance * voltage
        ...             self.unbind_cond()
        ...             return current
        ...         return 0.0
    """
    # Attribute to store temporary conductance data
    # Holds opaque conductance arrays/quantities, or None when unbound.
    _conductance: Any

    def bind_cond(self, conductance):
        r"""
        Bind conductance data temporarily.

        Parameters
        ----------
        conductance : Any
            The conductance data to bind.
        """
        self._conductance = conductance

    def unbind_cond(self):
        r"""
        Unbind (clear) the conductance data.
        """
        self._conductance = None
