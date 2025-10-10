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

from typing import Callable, Optional

import braintools

import brainstate

__all__ = [
    'Neuron', 'Synapse',
]


class Neuron(brainstate.nn.Dynamics):
    r"""
    Base class for all spiking neuron models.

    This abstract class serves as the foundation for implementing various spiking neuron
    models in the BrainPy framework. It extends the ``brainstate.nn.Dynamics`` class and
    provides common functionality for spike generation, membrane potential dynamics, and
    surrogate gradient handling required for training spiking neural networks.

    All neuron models (e.g., IF, LIF, LIFRef, ALIF) should inherit from this class and
    implement the required abstract methods, particularly ``get_spike()`` which defines
    the spike generation mechanism.

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron layer. Can be an integer for 1D input or a tuple
        for multi-dimensional input (e.g., ``100`` or ``(28, 28)``).
    spk_fun : Callable, optional
        Surrogate gradient function for the non-differentiable spike generation operation.
        Default is ``braintools.surrogate.InvSquareGrad()``. Common alternatives include:

        - ``braintools.surrogate.ReluGrad()``
        - ``braintools.surrogate.SigmoidGrad()``
        - ``braintools.surrogate.GaussianGrad()``
        - ``braintools.surrogate.ATan()``
    spk_reset : str, optional
        Reset mechanism applied after spike generation. Default is ``'soft'``.

        - ``'soft'``: Subtract threshold from membrane potential (``V = V - V_th``).
          This allows for more biological realism and better gradient flow.
        - ``'hard'``: Apply strict reset using ``jax.lax.stop_gradient`` to set
          voltage to reset value (``V = V_reset``).
    name : str, optional
        Name identifier for the neuron layer. If ``None``, an automatic name will be
        generated. Useful for debugging and visualization.

    Attributes
    ----------
    spk_reset : str
        The reset mechanism used by the neuron.
    spk_fun : Callable
        The surrogate gradient function used for spike generation.


    Notes
    -----
    **Surrogate Gradients**

    The spike generation operation is inherently non-differentiable (a threshold function),
    which poses challenges for gradient-based learning. Surrogate gradients provide a
    differentiable approximation during the backward pass while maintaining the discrete
    spike behavior during the forward pass. This is crucial for training SNNs with
    backpropagation through time (BPTT).

    **Reset Mechanisms**

    - **Soft Reset**: More biologically plausible as it preserves information about
      how far above threshold the membrane potential was. This can encode information
      in the residual voltage and often leads to better gradient flow.

    - **Hard Reset**: Provides a clean reset to a fixed value, which can be easier to
      analyze mathematically but may lead to vanishing gradients in deep networks.

    **State Management**

    Neuron models typically maintain state variables (e.g., membrane potential ``V``,
    adaptation current ``a``) as ``brainstate.HiddenState`` objects. These states are:

    - Initialized via ``init_state(batch_size=None, **kwargs)``
    - Reset via ``reset_state(batch_size=None, **kwargs)``
    - Updated via ``update(x)`` which returns spikes for the current timestep

    Examples
    --------
    **Creating a Custom Neuron Model**

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> import braintools
        >>> import brainpy
        >>>
        >>> class SimpleNeuron(brainpy.Neuron):
        ...     def __init__(self, in_size, V_th=1.0*u.mV, **kwargs):
        ...         super().__init__(in_size, **kwargs)
        ...         self.V_th = V_th
        ...
        ...     def init_state(self, batch_size=None, **kwargs):
        ...         self.V = brainstate.HiddenState(
        ...             braintools.init.param(
        ...                 braintools.init.Constant(0.*u.mV),
        ...                 self.varshape,
        ...                 batch_size
        ...             )
        ...         )
        ...
        ...     def reset_state(self, batch_size=None, **kwargs):
        ...         self.V.value = braintools.init.param(
        ...             braintools.init.Constant(0.*u.mV),
        ...             self.varshape,
        ...             batch_size
        ...         )
        ...
        ...     def get_spike(self, V=None):
        ...         V = self.V.value if V is None else V
        ...         return self.spk_fun((V - self.V_th) / self.V_th)
        ...
        ...     def update(self, x):
        ...         self.V.value += x
        ...         return self.get_spike()

    **Using Built-in Neuron Models**

    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>>
        >>> # Create a LIF neuron layer
        >>> neuron = brainpy.LIF(
        ...     in_size=100,
        ...     tau=10*u.ms,
        ...     V_th=1.0*u.mV,
        ...     spk_fun=braintools.surrogate.ReluGrad(),
        ...     spk_reset='soft'
        ... )
        >>>
        >>> # Initialize state for batch processing
        >>> neuron.init_state(batch_size=32)
        >>>
        >>> # Process input and get spikes
        >>> input_current = 2.0 * u.mA
        >>> spikes = neuron.update(input_current)
        >>> print(spikes.shape)
        (32, 100)

    **Building a Multi-Layer Spiking Network**

    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>>
        >>> # Create a network with multiple neuron types
        >>> class SpikingNet(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.layer1 = brainpy.LIF(784, tau=5*u.ms)
        ...         self.fc1 = brainstate.nn.Linear(784, 256)
        ...         self.layer2 = brainpy.ALIF(256, tau=10*u.ms, tau_a=200*u.ms)
        ...         self.fc2 = brainstate.nn.Linear(256, 10)
        ...         self.layer3 = brainpy.LIF(10, tau=8*u.ms)
        ...
        ...     def __call__(self, x):
        ...         spikes1 = self.layer1.update(x)
        ...         x1 = self.fc1(spikes1)
        ...         spikes2 = self.layer2.update(x1)
        ...         x2 = self.fc2(spikes2)
        ...         spikes3 = self.layer3.update(x2)
        ...         return spikes3

    References
    ----------
    .. [1] Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in
           spiking neural networks: Bringing the power of gradient-based optimization to
           spiking neural networks. IEEE Signal Processing Magazine, 36(6), 51-63.
    .. [2] Zenke, F., & Ganguli, S. (2018). SuperSpike: Supervised learning in multilayer
           spiking neural networks. Neural computation, 30(6), 1514-1541.
    .. [3] Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014). Neuronal dynamics:
           From single neurons to networks and models of cognition. Cambridge University Press.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        spk_fun: Callable = braintools.surrogate.InvSquareGrad(),
        spk_reset: str = 'soft',
        name: Optional[str] = None,
    ):
        super().__init__(in_size, name=name)
        self.spk_reset = spk_reset
        self.spk_fun = spk_fun

    def get_spike(self, *args, **kwargs):
        """
        Generate spikes based on neuron state variables.

        This abstract method must be implemented by subclasses to define the
        spike generation mechanism. The method should use the surrogate gradient
        function ``self.spk_fun`` to enable gradient-based learning.

        Parameters
        ----------
        *args
            Positional arguments (typically state variables like membrane potential)
        **kwargs
            Keyword arguments

        Returns
        -------
        ArrayLike
            Binary spike tensor where 1 indicates a spike and 0 indicates no spike.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError


class Synapse(brainstate.nn.Dynamics):
    r"""
    Base class for synapse dynamics.

    This class serves as the foundation for all synapse models in the BrainPy framework,
    providing a common interface for implementing various types of synaptic connectivity
    and transmission mechanisms. Synapses model the transmission of signals (typically
    spikes) between neurons, including temporal dynamics, plasticity, and neurotransmitter
    effects.

    All specific synapse implementations (like Expon, Alpha, DualExpon, AMPA, GABAa, etc.)
    should inherit from this class and implement the required methods for state management
    and dynamics update.

    Parameters
    ----------
    in_size : Size
        Size of the presynaptic input. Can be an integer for 1D input or a tuple
        for multi-dimensional input (e.g., ``100`` or ``(10, 10)``).
    name : str, optional
        Name identifier for the synapse layer. If ``None``, an automatic name will be
        generated. Useful for debugging and model inspection.

    Attributes
    ----------
    varshape : tuple
        Shape of the synaptic state variables, derived from ``in_size``.

    Returns
    -------
    ArrayLike
        Synaptic output (e.g., conductance, current, or gating variable)

    See Also
    --------
    Expon : Simple first-order exponential decay synapse model
    DualExpon : Dual exponential synapse model with separate rise and decay
    Alpha : Alpha function synapse model
    AMPA : AMPA receptor-mediated excitatory synapse
    GABAa : GABAa receptor-mediated inhibitory synapse

    Notes
    -----
    **Synaptic Dynamics**

    Synapses implement temporal filtering of presynaptic signals. The dynamics are
    typically described by differential equations that govern how synaptic conductance
    or current evolves over time in response to presynaptic spikes.

    **State Variables**

    Synapse models typically maintain state variables (e.g., conductance ``g``,
    gating variables) as ``brainstate.HiddenState`` or ``brainstate.ShortTermState``
    objects depending on whether they need to be preserved across simulation episodes.

    **Integration with Neurons**

    Synapses are commonly used in conjunction with projection layers or connectivity
    matrices to model synaptic transmission between neuron populations:

    - In feedforward networks: Linear layer → Synapse → Neuron
    - In recurrent networks: Neuron → Linear layer → Synapse → Neuron

    **Alignment Patterns**

    Some synapse models inherit from :class:`AlignPost` to enable
    event-driven computation where synaptic variables are aligned with postsynaptic
    neurons. This is particularly efficient for sparse connectivity patterns.

    Examples
    --------
    **Creating a Custom Synapse Model**

    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>> import braintools
        >>>
        >>> class SimpleSynapse(brainpy.Synapse):
        ...     def __init__(self, in_size, tau=5.0*u.ms, **kwargs):
        ...         super().__init__(in_size, **kwargs)
        ...         self.tau = braintools.init.param(tau, self.varshape)
        ...         self.g_init = braintools.init.Constant(0.*u.mS)
        ...
        ...     def init_state(self, batch_size=None, **kwargs):
        ...         self.g = brainstate.HiddenState(braintools.init.param(self.g_init, self.varshape, batch_size))
        ...
        ...     def reset_state(self, batch_size=None, **kwargs):
        ...         self.g.value = braintools.init.param(self.g_init, self.varshape, batch_size)
        ...
        ...     def update(self, x=None):
        ...         # Simple exponential decay: dg/dt = -g/tau + x
        ...         dg = lambda g: -g / self.tau
        ...         self.g.value = brainstate.nn.exp_euler_step(dg, self.g.value)
        ...         if x is not None:
        ...             self.g.value += x
        ...         return self.g.value

    **Using Built-in Synapse Models**

    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>> import jax
        >>>
        >>> # Create an exponential synapse
        >>> synapse = brainpy.Expon(in_size=100, tau=8.0*u.ms)
        >>>
        >>> # Initialize state
        >>> synapse.init_state(batch_size=32)
        >>>
        >>> # Update with presynaptic spikes
        >>> spikes = jax.random.bernoulli(
        ...     jax.random.PRNGKey(0),
        ...     p=0.1,
        ...     shape=(32, 100)
        ... )
        >>> conductance = synapse.update(spikes * 1.0*u.mS)
        >>> print(conductance.shape)
        (32, 100)

    **Building a Feedforward Spiking Network**

    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>>
        >>> class SynapticNetwork(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         # Input layer
        ...         self.input_neurons = brainpy.LIF(784, tau=5*u.ms)
        ...         # First hidden layer with synaptic filtering
        ...         self.fc1 = brainstate.nn.Linear(784, 256)
        ...         self.syn1 = brainpy.Expon(256, tau=8*u.ms)
        ...         self.hidden1 = brainpy.LIF(256, tau=10*u.ms)
        ...         # Second hidden layer with AMPA synapse
        ...         self.fc2 = brainstate.nn.Linear(256, 128)
        ...         self.syn2 = brainpy.AMPA(128)
        ...         self.hidden2 = brainpy.LIF(128, tau=10*u.ms)
        ...         # Output layer
        ...         self.fc3 = brainstate.nn.Linear(128, 10)
        ...         self.output_neurons = brainpy.LIF(10, tau=8*u.ms)
        ...
        ...     def __call__(self, x):
        ...         # Input layer
        ...         spikes0 = self.input_neurons.update(x)
        ...         # First hidden layer
        ...         current1 = self.fc1(spikes0)
        ...         g1 = self.syn1.update(current1)
        ...         spikes1 = self.hidden1.update(g1)
        ...         # Second hidden layer
        ...         current2 = self.fc2(spikes1)
        ...         g2 = self.syn2.update(current2)
        ...         spikes2 = self.hidden2.update(g2)
        ...         # Output layer
        ...         current3 = self.fc3(spikes2)
        ...         output_spikes = self.output_neurons.update(current3)
        ...         return output_spikes

    **Recurrent Network with Inhibition**

    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>>
        >>> class EINetwork(brainstate.nn.Module):
        ...     def __init__(self, n_exc=800, n_inh=200):
        ...         super().__init__()
        ...         # Excitatory population
        ...         self.exc_neurons = brainpy.LIF(n_exc, tau=10*u.ms)
        ...         self.exc_syn = brainpy.AMPA(n_exc)
        ...         # Inhibitory population
        ...         self.inh_neurons = brainpy.LIF(n_inh, tau=8*u.ms)
        ...         self.inh_syn = brainpy.GABAa(n_inh)
        ...         # Connectivity
        ...         self.exc_to_exc = brainstate.nn.Linear(n_exc, n_exc)
        ...         self.exc_to_inh = brainstate.nn.Linear(n_exc, n_inh)
        ...         self.inh_to_exc = brainstate.nn.Linear(n_inh, n_exc)
        ...         self.inh_to_inh = brainstate.nn.Linear(n_inh, n_inh)
        ...
        ...     def __call__(self, ext_input):
        ...         # Excitatory neurons receive external input and recurrent excitation/inhibition
        ...         exc_current = (ext_input +
        ...                       self.exc_to_exc(self.exc_syn.g.value) -
        ...                       self.inh_to_exc(self.inh_syn.g.value))
        ...         exc_spikes = self.exc_neurons.update(exc_current)
        ...         self.exc_syn.update(exc_spikes)
        ...         # Inhibitory neurons receive excitatory input and recurrent inhibition
        ...         inh_current = (self.exc_to_inh(self.exc_syn.g.value) -
        ...                       self.inh_to_inh(self.inh_syn.g.value))
        ...         inh_spikes = self.inh_neurons.update(inh_current)
        ...         self.inh_syn.update(inh_spikes)
        ...         return exc_spikes, inh_spikes

    References
    ----------
    .. [1] Destexhe, A., Mainen, Z. F., & Sejnowski, T. J. (1994). Synthesis of models for
           excitable membranes, synaptic transmission and neuromodulation using a common
           kinetic formalism. Journal of computational neuroscience, 1(3), 195-230.
    .. [2] Dayan, P., & Abbott, L. F. (2001). Theoretical neuroscience: Computational and
           mathematical modeling of neural systems. MIT Press.
    .. [3] Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014). Neuronal dynamics:
           From single neurons to networks and models of cognition. Cambridge University Press.
    """
    __module__ = 'brainpy'
