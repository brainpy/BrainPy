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

from typing import Optional

import brainstate

__all__ = [
    'AlignPost',
    'BindCondData',
    'Mode',
    'JointMode',
    'Batching',
    'Training',
]


class AlignPost(brainstate.mixin.Mixin):
    """
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
        """
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
    """
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
    _conductance: Optional

    def bind_cond(self, conductance):
        """
        Bind conductance data temporarily.

        Parameters
        ----------
        conductance : Any
            The conductance data to bind.
        """
        self._conductance = conductance

    def unbind_cond(self):
        """
        Unbind (clear) the conductance data.
        """
        self._conductance = None


class Mode(brainstate.nn.Mixin):
    """
    Base class for computation behavior modes.

    Modes are used to represent different computational contexts or behaviors,
    such as training vs evaluation, batched vs single-sample processing, etc.
    They provide a flexible way to configure how models and components behave
    in different scenarios.

    Examples
    --------
    Creating a custom mode:

    .. code-block:: python

        >>> import brainstate
        >>>
        >>> class InferenceMode(brainstate.mixin.Mode):
        ...     def __init__(self, use_cache=True):
        ...         self.use_cache = use_cache
        >>>
        >>> # Create mode instances
        >>> inference = InferenceMode(use_cache=True)
        >>> print(inference)  # Output: InferenceMode

    Checking mode types:

    .. code-block:: python

        >>> class FastMode(brainstate.mixin.Mode):
        ...     pass
        >>>
        >>> class SlowMode(brainstate.mixin.Mode):
        ...     pass
        >>>
        >>> fast = FastMode()
        >>> slow = SlowMode()
        >>>
        >>> # Check exact mode type
        >>> assert fast.is_a(FastMode)
        >>> assert not fast.is_a(SlowMode)
        >>>
        >>> # Check if mode is an instance of a type
        >>> assert fast.has(brainstate.mixin.Mode)

    Using modes in a model:

    .. code-block:: python

        >>> class Model:
        ...     def __init__(self):
        ...         self.mode = brainstate.mixin.Training()
        ...
        ...     def forward(self, x):
        ...         if self.mode.has(brainstate.mixin.Training):
        ...             # Training-specific logic
        ...             return self.train_forward(x)
        ...         else:
        ...             # Inference logic
        ...             return self.eval_forward(x)
        ...
        ...     def train_forward(self, x):
        ...         return x + 0.1  # Add noise during training
        ...
        ...     def eval_forward(self, x):
        ...         return x  # No noise during evaluation
    """

    def __repr__(self):
        """
        String representation of the mode.

        Returns
        -------
        str
            The class name of the mode.
        """
        return self.__class__.__name__

    def __eq__(self, other: 'Mode'):
        """
        Check equality of modes based on their type.

        Parameters
        ----------
        other : Mode
            Another mode to compare with.

        Returns
        -------
        bool
            True if both modes are of the same class.
        """
        assert isinstance(other, Mode)
        return other.__class__ == self.__class__

    def is_a(self, mode: type):
        """
        Check whether the mode is exactly the desired mode type.

        This performs an exact type match, not checking for subclasses.

        Parameters
        ----------
        mode : type
            The mode type to check against.

        Returns
        -------
        bool
            True if this mode is exactly of the specified type.

        Examples
        --------
        .. code-block:: python

            >>> import brainstate
            >>>
            >>> training_mode = brainstate.mixin.Training()
            >>> assert training_mode.is_a(brainstate.mixin.Training)
            >>> assert not training_mode.is_a(brainstate.mixin.Batching)
        """
        assert isinstance(mode, type), 'Must be a type.'
        return self.__class__ == mode

    def has(self, mode: type):
        """
        Check whether the mode includes the desired mode type.

        This checks if the current mode is an instance of the specified type,
        including checking for subclasses.

        Parameters
        ----------
        mode : type
            The mode type to check for.

        Returns
        -------
        bool
            True if this mode is an instance of the specified type.

        Examples
        --------
        .. code-block:: python

            >>> import brainstate
            >>>
            >>> # Create a custom mode that extends Training
            >>> class AdvancedTraining(brainstate.mixin.Training):
            ...     pass
            >>>
            >>> advanced = AdvancedTraining()
            >>> assert advanced.has(brainstate.mixin.Training)  # True (subclass)
            >>> assert advanced.has(brainstate.mixin.Mode)      # True (base class)
        """
        assert isinstance(mode, type), 'Must be a type.'
        return isinstance(self, mode)


class JointMode(Mode):
    """
    A mode that combines multiple modes simultaneously.

    JointMode allows expressing that a computation is in multiple modes at once,
    such as being both in training mode and batching mode. This is useful for
    complex scenarios where multiple behavioral aspects need to be active.

    Parameters
    ----------
    *modes : Mode
        The modes to combine.

    Attributes
    ----------
    modes : tuple of Mode
        The individual modes that are combined.
    types : set of type
        The types of the combined modes.

    Raises
    ------
    TypeError
        If any of the provided arguments is not a Mode instance.

    Examples
    --------
    Combining training and batching modes:

    .. code-block:: python

        >>> import brainstate
        >>>
        >>> # Create individual modes
        >>> training = brainstate.mixin.Training()
        >>> batching = brainstate.mixin.Batching(batch_size=32)
        >>>
        >>> # Combine them
        >>> joint = brainstate.mixin.JointMode(training, batching)
        >>> print(joint)  # JointMode(Training, Batching(in_size=32, axis=0))
        >>>
        >>> # Check if specific modes are present
        >>> assert joint.has(brainstate.mixin.Training)
        >>> assert joint.has(brainstate.mixin.Batching)
        >>>
        >>> # Access attributes from combined modes
        >>> print(joint.batch_size)  # 32 (from Batching mode)

    Using in model configuration:

    .. code-block:: python

        >>> class NeuralNetwork:
        ...     def __init__(self):
        ...         self.mode = None
        ...
        ...     def set_train_mode(self, batch_size=1):
        ...         # Set both training and batching modes
        ...         training = brainstate.mixin.Training()
        ...         batching = brainstate.mixin.Batching(batch_size=batch_size)
        ...         self.mode = brainstate.mixin.JointMode(training, batching)
        ...
        ...     def forward(self, x):
        ...         if self.mode.has(brainstate.mixin.Training):
        ...             x = self.apply_dropout(x)
        ...
        ...         if self.mode.has(brainstate.mixin.Batching):
        ...             # Process in batches
        ...             batch_size = self.mode.batch_size
        ...             return self.batch_process(x, batch_size)
        ...
        ...         return self.process(x)
        >>>
        >>> model = NeuralNetwork()
        >>> model.set_train_mode(batch_size=64)
    """

    def __init__(self, *modes: Mode):
        # Validate that all arguments are Mode instances
        for m_ in modes:
            if not isinstance(m_, Mode):
                raise TypeError(f'The supported type must be a tuple/list of Mode. But we got {m_}')

        # Store the modes as a tuple
        self.modes = tuple(modes)

        # Store the types of the modes for quick lookup
        self.types = set([m.__class__ for m in modes])

    def __repr__(self):
        """
        String representation showing all combined modes.

        Returns
        -------
        str
            A string showing the joint mode and its components.
        """
        return f'{self.__class__.__name__}({", ".join([repr(m) for m in self.modes])})'

    def has(self, mode: type):
        """
        Check whether any of the combined modes includes the desired type.

        Parameters
        ----------
        mode : type
            The mode type to check for.

        Returns
        -------
        bool
            True if any of the combined modes is or inherits from the specified type.

        Examples
        --------
        .. code-block:: python

            >>> import brainstate
            >>>
            >>> training = brainstate.mixin.Training()
            >>> batching = brainstate.mixin.Batching(batch_size=16)
            >>> joint = brainstate.mixin.JointMode(training, batching)
            >>>
            >>> assert joint.has(brainstate.mixin.Training)
            >>> assert joint.has(brainstate.mixin.Batching)
            >>> assert joint.has(brainstate.mixin.Mode)  # Base class
        """
        assert isinstance(mode, type), 'Must be a type.'
        # Check if any of the combined mode types is a subclass of the target mode
        return any([issubclass(cls, mode) for cls in self.types])

    def is_a(self, cls: type):
        """
        Check whether the joint mode is exactly the desired combined type.

        This is a complex check that verifies the joint mode matches a specific
        combination of types.

        Parameters
        ----------
        cls : type
            The combined type to check against.

        Returns
        -------
        bool
            True if the joint mode exactly matches the specified type combination.
        """
        # Use JointTypes to create the expected type from our mode types
        return brainstate.mixin.JointTypes(*tuple(self.types)) == cls

    def __getattr__(self, item):
        """
        Get attributes from the combined modes.

        This method searches through all combined modes to find the requested
        attribute, allowing transparent access to properties of any of the
        combined modes.

        Parameters
        ----------
        item : str
            The attribute name to search for.

        Returns
        -------
        Any
            The attribute value from the first mode that has it.

        Raises
        ------
        AttributeError
            If the attribute is not found in any of the combined modes.

        Examples
        --------
        .. code-block:: python

            >>> import brainstate
            >>>
            >>> batching = brainstate.mixin.Batching(batch_size=32, batch_axis=1)
            >>> training = brainstate.mixin.Training()
            >>> joint = brainstate.mixin.JointMode(batching, training)
            >>>
            >>> # Access batching attributes directly
            >>> print(joint.batch_size)  # 32
            >>> print(joint.batch_axis)  # 1
        """
        # Don't interfere with accessing modes and types attributes
        if item in ['modes', 'types']:
            return super().__getattribute__(item)

        # Search for the attribute in each combined mode
        for m in self.modes:
            if hasattr(m, item):
                return getattr(m, item)

        # If not found, fall back to default behavior (will raise AttributeError)
        return super().__getattribute__(item)


class Batching(Mode):
    """
    Mode indicating batched computation.

    This mode specifies that computations should be performed on batches of data,
    including information about the batch size and which axis represents the batch
    dimension.

    Parameters
    ----------
    batch_size : int, default 1
        The size of each batch.
    batch_axis : int, default 0
        The axis along which batching occurs.

    Attributes
    ----------
    batch_size : int
        The number of samples in each batch.
    batch_axis : int
        The axis index representing the batch dimension.

    Examples
    --------
    Basic batching configuration:

    .. code-block:: python

        >>> import brainstate
        >>>
        >>> # Create a batching mode
        >>> batching = brainstate.mixin.Batching(batch_size=32, batch_axis=0)
        >>> print(batching)  # Batching(in_size=32, axis=0)
        >>>
        >>> # Access batch parameters
        >>> print(f"Processing {batching.batch_size} samples at once")
        >>> print(f"Batch dimension is axis {batching.batch_axis}")

    Using in a model:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> class BatchedModel:
        ...     def __init__(self):
        ...         self.mode = None
        ...
        ...     def set_batch_mode(self, batch_size, batch_axis=0):
        ...         self.mode = brainstate.mixin.Batching(batch_size, batch_axis)
        ...
        ...     def process(self, x):
        ...         if self.mode is not None and self.mode.has(brainstate.mixin.Batching):
        ...             # Process in batches
        ...             batch_size = self.mode.batch_size
        ...             axis = self.mode.batch_axis
        ...             return jnp.mean(x, axis=axis, keepdims=True)
        ...         return x
        >>>
        >>> model = BatchedModel()
        >>> model.set_batch_mode(batch_size=64)
        >>>
        >>> # Process batched data
        >>> data = jnp.random.randn(64, 100)  # 64 samples, 100 features
        >>> result = model.process(data)

    Combining with other modes:

    .. code-block:: python

        >>> # Combine batching with training mode
        >>> training = brainstate.mixin.Training()
        >>> batching = brainstate.mixin.Batching(batch_size=128)
        >>> combined = brainstate.mixin.JointMode(training, batching)
        >>>
        >>> # Use in a training loop
        >>> def train_step(model, data, mode):
        ...     if mode.has(brainstate.mixin.Batching):
        ...         # Split data into batches
        ...         batch_size = mode.batch_size
        ...         # ... batched processing ...
        ...     if mode.has(brainstate.mixin.Training):
        ...         # Apply training-specific operations
        ...         # ... training logic ...
        ...     pass
    """

    def __init__(self, batch_size: int = 1, batch_axis: int = 0):
        self.batch_size = batch_size
        self.batch_axis = batch_axis

    def __repr__(self):
        """
        String representation showing batch configuration.

        Returns
        -------
        str
            A string showing the batch size and axis.
        """
        return f'{self.__class__.__name__}(in_size={self.batch_size}, axis={self.batch_axis})'


class Training(Mode):
    """
    Mode indicating training computation.

    This mode specifies that the model is in training mode, which typically
    enables behaviors like dropout, batch normalization in training mode,
    gradient computation, etc.

    Examples
    --------
    Basic training mode:

    .. code-block:: python

        >>> import brainstate
        >>>
        >>> # Create training mode
        >>> training = brainstate.mixin.Training()
        >>> print(training)  # Training
        >>>
        >>> # Check mode
        >>> assert training.is_a(brainstate.mixin.Training)
        >>> assert training.has(brainstate.mixin.Mode)

    Using in a model with dropout:

    .. code-block:: python

        >>> import brainstate
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> class ModelWithDropout:
        ...     def __init__(self, dropout_rate=0.5):
        ...         self.dropout_rate = dropout_rate
        ...         self.mode = None
        ...
        ...     def set_training(self, is_training=True):
        ...         if is_training:
        ...             self.mode = brainstate.mixin.Training()
        ...         else:
        ...             self.mode = brainstate.mixin.Mode()  # Evaluation mode
        ...
        ...     def forward(self, x, rng_key):
        ...         # Apply dropout only during training
        ...         if self.mode is not None and self.mode.has(brainstate.mixin.Training):
        ...             keep_prob = 1.0 - self.dropout_rate
        ...             mask = jax.random.bernoulli(rng_key, keep_prob, x.shape)
        ...             x = jnp.where(mask, x / keep_prob, 0)
        ...         return x
        >>>
        >>> model = ModelWithDropout()
        >>>
        >>> # Training mode
        >>> model.set_training(True)
        >>> key = jax.random.PRNGKey(0)
        >>> x_train = jnp.ones((10, 20))
        >>> out_train = model.forward(x_train, key)  # Dropout applied
        >>>
        >>> # Evaluation mode
        >>> model.set_training(False)
        >>> out_eval = model.forward(x_train, key)  # No dropout

    Combining with batching:

    .. code-block:: python

        >>> # Create combined training and batching mode
        >>> training = brainstate.mixin.Training()
        >>> batching = brainstate.mixin.Batching(batch_size=32)
        >>> mode = brainstate.mixin.JointMode(training, batching)
        >>>
        >>> # Use in training configuration
        >>> class Trainer:
        ...     def __init__(self, model, mode):
        ...         self.model = model
        ...         self.mode = mode
        ...
        ...     def train_epoch(self, data):
        ...         if self.mode.has(brainstate.mixin.Training):
        ...             # Enable training-specific behaviors
        ...             self.model.set_training(True)
        ...
        ...         if self.mode.has(brainstate.mixin.Batching):
        ...             # Process in batches
        ...             batch_size = self.mode.batch_size
        ...             # ... batched training loop ...
        ...         pass
    """
    pass
