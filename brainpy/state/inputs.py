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

from collections.abc import Sequence, Callable

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size, DTypeLike

from brainpy.state._misc import set_module_as

__all__ = [
    'SpikeTime',
    'PoissonSpike',
    'PoissonEncoder',
    'PoissonInput',
    'poisson_input',
    'SectionInput',
    'ConstantInput',
    'StepInput',
    'RampInput',
    'SinusoidalInput',
    'OUProcessInput',
    'WienerProcessInput',
]


def _now():
    r"""Resolve the current simulation time as a ``brainunit`` quantity.

    Mirrors :meth:`SpikeTime.update`: the step index ``i`` in
    ``brainstate.environ`` is the primary clock (``t = i * dt``); ``t`` is used
    as a fallback. This keeps every generator synchronized with the network's
    clock rather than carrying an independent counter that could drift.
    """
    dt = brainstate.environ.get_dt()
    i = brainstate.environ.get('i', default=None)
    if i is not None:
        return i * dt
    return brainstate.environ.get('t')


class SpikeTime(brainstate.nn.Dynamics):
    r"""The input neuron group characterized by spikes emitting at given times.

    Internally builds a ``brainevent.CSR`` matrix of shape
    ``[n_max_time_step, n_neuron]`` so that ``update()`` reduces to a
    single CSR row slice.

    >>> # Get 2 neurons, firing spikes at 10 ms and 20 ms.
    >>> SpikeTime(2, times=[10, 20])
    >>> # or
    >>> # Get 2 neurons, the neuron 0 fires spikes at 10 ms and 20 ms.
    >>> SpikeTime(2, times=[10, 20], indices=[0, 0])
    >>> # or
    >>> # Get 2 neurons, neuron 0 fires at 10 ms and 30 ms, neuron 1 fires at 20 ms.
    >>> SpikeTime(2, times=[10, 20, 30], indices=[0, 1, 0])
    >>> # or
    >>> # Get 2 neurons; at 10 ms, neuron 0 fires; at 20 ms, neuron 0 and 1 fire;
    >>> # at 30 ms, neuron 1 fires.
    >>> SpikeTime(2, times=[10, 20, 20, 30], indices=[0, 0, 1, 1])
    >>> # or
    >>> # Get 2 neurons with a uniform float weight for all spikes.
    >>> SpikeTime(2, times=[10, 20], indices=[0, 1], weights=2.5)
    >>> # or
    >>> # Get 2 neurons with per-event weights.
    >>> SpikeTime(2, times=[10, 20], indices=[0, 1], weights=[0.5, 0.8])

    Parameters
    ----------
    in_size : int, tuple, list
        The neuron group geometry.
    indices : list, tuple, ArrayType
        The neuron indices at each time point to emit spikes.
    times : list, tuple, ArrayType
        The time points which generate the spikes.
    weights : bool, float, Sequence, or ArrayLike, default=True
        Spike weights. ``True`` produces boolean output (backward-compatible).
        A float scalar applies the same weight to every spike. A sequence of
        the same length as ``indices``/``times`` assigns per-event weights.
        The output dtype is inferred from this parameter.
    time_as_step : callable, str, default='round'
        Rounding method for converting spike times to integer step indices.
        Options: ``'floor'``, ``'round'``, ``'ceil'``.
    name : str, optional
        The name of the dynamic system.

    See Also
    --------
    PoissonSpike : Stochastic Poisson spike generator.
    PoissonEncoder : Poisson encoder with dynamic rates.

    Notes
    -----
    - Internally, a ``brainevent.CSR`` sparse matrix of shape
      ``(n_max_time_step, n_neuron)`` is pre-built at construction, so
      :meth:`update` reduces to a single CSR row slice.
    - Only 1-D neuron groups are supported.
    - Spike times are converted to integer step indices using the rounding
      method specified by ``time_as_step``.
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        indices: Sequence | ArrayLike,
        times: Sequence | ArrayLike,
        weights: float | Sequence | ArrayLike = 1.0,
        time_as_step: str | Callable = 'round',
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        # parameters
        if len(indices) != len(times):
            raise ValueError(f'The length of "indices" and "times" must be the same. '
                             f'However, we got {len(indices)} != {len(times)}.')
        if not callable(time_as_step):
            if time_as_step not in ('floor', 'round', 'ceil'):
                raise ValueError(f'"time_as_step" must be one of "floor", "round", "ceil". '
                                 f'Got {time_as_step!r}.')
        self.num_times = len(times)
        self.time_as_step = time_as_step

        # weights
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()
        weights = u.math.asarray(weights, dtype=dftype)
        self._scalar_weight = (weights.ndim == 0)
        self._spk_dtype = weights.dtype
        if not self._scalar_weight and len(weights) != self.num_times:
            raise ValueError(
                f'Length of "weights" must match "indices"/"times". '
                f'Got {len(weights)} != {self.num_times}.'
            )

        # data about times and indices
        times = u.math.asarray(times, dtype=dftype)
        indices = u.math.asarray(indices, dtype=ditype)
        if self._scalar_weight:
            self.weights = weights
            self.times, self.indices = u.lax.sort((times, indices), dimension=0)
        else:
            sorted_arrays = u.lax.sort((times, indices, weights), dimension=0)
            self.times = sorted_arrays[0]
            self.indices = sorted_arrays[1]
            self.weights = sorted_arrays[2]

        # Build CSR matrix [n_max_time_step, n_neuron]
        import brainevent

        assert len(self.varshape) == 1, 'SpikeTime only supports 1D neuron groups for now.'
        n_cols = self.varshape[-1]

        dt = brainstate.environ.get_dt()
        ratio = self.times / dt
        if isinstance(ratio, u.Quantity):
            ratio = ratio.to_decimal()

        steps = jnp.asarray(self.rounding(ratio), dtype=ditype)
        indices = jnp.asarray(self.indices, dtype=ditype)

        if self.num_times == 0:
            return

        # Build CSR indptr
        n_rows = int(steps.max()) + 1
        counts = np.bincount(np.asarray(steps), minlength=n_rows)
        indptr_np = np.zeros(n_rows + 1, dtype=ditype)
        np.cumsum(counts, out=indptr_np[1:])

        self._csr = brainevent.CSR(
            self.weights,
            jax.numpy.asarray(indices, dtype=ditype),
            jax.numpy.asarray(indptr_np, dtype=ditype),
            shape=(n_rows, n_cols)
        )

    @property
    def rounding(self):
        if callable(self.time_as_step):
            return self.time_as_step
        elif self.time_as_step == 'floor':
            return u.math.floor
        elif self.time_as_step == 'round':
            return u.math.round
        else:
            return u.math.ceil

    def _to_int(self, ratio):
        if isinstance(ratio, u.Quantity):
            ratio = ratio.to_decimal()
        return u.math.asarray(self.rounding(ratio), dtype=brainstate.environ.ditype())

    def update(self, index=None, time=None):
        if self.num_times == 0:
            return jax.numpy.zeros(self.varshape, dtype=self._spk_dtype)

        if index is not None:
            i = jax.numpy.asarray(index, dtype=brainstate.environ.ditype())
        elif time is not None:
            i = self._to_int(time / brainstate.environ.get_dt())
        else:
            # Both None — read from brainstate.environ
            i = brainstate.environ.get('i', default=None)
            if i is None:
                t = brainstate.environ.get('t')
                dt = brainstate.environ.get_dt()
                i = self._to_int(t / dt)
            else:
                i = jax.numpy.asarray(i, dtype=brainstate.environ.ditype())

        # Slice CSR row → dense spike vector of shape (n_cols,)
        # Reshape to 1D for csr_slice_rows, then squeeze back.
        # Steps outside [0, n_rows) carry no events: clamp the gather index to a
        # valid row, then mask the result back to zeros. This keeps behaviour
        # identical in eager mode (no IndexError) and under JIT (no silent clamp).
        ditype = brainstate.environ.ditype()
        i = jnp.asarray(i, dtype=ditype)
        n_rows = self._csr.shape[0]
        row = self._csr[jnp.clip(i, 0, n_rows - 1).reshape(1)][0]
        return jnp.where(
            (i >= 0) & (i < n_rows),
            row,
            jnp.zeros(self.varshape, dtype=self._spk_dtype),
        )


def _per_bin_spike_prob(freq):
    r"""Exact per-bin spike probability of a Poisson process, ``1 - exp(-freq*dt)``.

    Bounded in ``[0, 1)`` for any rate, unlike the linear ``freq*dt`` approximation
    which exceeds 1 — and yields invalid binomial parameters / NaN — once
    ``freq*dt > 1``.
    """
    rate_dt = freq * brainstate.environ.get_dt()
    rate_dt = rate_dt.to_decimal() if isinstance(rate_dt, u.Quantity) else rate_dt
    return -u.math.expm1(-rate_dt)


class PoissonSpike(brainstate.nn.Dynamics):
    r"""Poisson spike generator with fixed firing rates.

    Generates independent Poisson spike trains for each neuron at every
    time step. The probability of a spike in each time bin is:

    .. math::

        P(\text{spike}) = \text{freq} \cdot dt

    where ``freq`` is the firing frequency and ``dt`` is the simulation
    time step.

    Parameters
    ----------
    in_size : Size
        Number of neurons (spike channels) to generate.
    freqs : ArrayLike or Callable
        Firing frequency for each neuron. Can be a scalar (same rate for
        all neurons), an array of shape ``in_size``, or a callable
        initializer.
    spk_type : DTypeLike, default=bool
        Data type of the output spike array. Use ``bool`` for binary
        spikes or a float type for weighted spikes.
    name : str, optional
        Name of the module.

    See Also
    --------
    PoissonEncoder : Poisson encoder that accepts dynamic firing rates.
    PoissonInput : Efficient Poisson input applied directly to a state variable.
    SpikeTime : Deterministic spike generator at specified times.

    Notes
    -----
    - Unlike :class:`PoissonEncoder`, the firing rates are fixed at
      construction time and cannot be changed during simulation.
    - Each call to :meth:`update` generates a fresh independent sample;
      there is no memory of previous spikes (renewal process).
    - For large populations, consider using :class:`PoissonInput` which
      avoids materializing the full spike array.

    References
    ----------
    .. [1] Dayan, P., & Abbott, L. F. (2001). Theoretical Neuroscience:
           Computational and Mathematical Modeling of Neural Systems.
           MIT Press.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>> # Create 100 Poisson neurons firing at 50 Hz
        >>> poisson = brainpy.state.PoissonSpike(100, freqs=50.*u.Hz)
        >>> with brainstate.environ.context(dt=0.1*u.ms):
        ...     spikes = poisson.update()
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        freqs: ArrayLike | Callable,
        spk_type: DTypeLike = bool,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.spk_type = spk_type

        # parameters
        self.freqs = braintools.init.param(freqs, self.varshape, allow_none=False)

    def update(self):
        spikes = brainstate.random.rand(*self.varshape) <= _per_bin_spike_prob(self.freqs)
        spikes = u.math.asarray(spikes, dtype=self.spk_type)
        return spikes


class PoissonEncoder(brainstate.nn.Dynamics):
    r"""Poisson spike encoder for converting firing rates to spike trains.

    This class implements a Poisson process to generate spikes based on provided
    firing rates. Unlike the PoissonSpike class, this encoder accepts firing rates
    as input during the update step rather than having them fixed at initialization.

    The spike generation follows a Poisson process where the probability of a spike
    in each time step is proportional to the firing rate and the simulation time step:

    $$
    P(\text{spike}) = \text{rate} \cdot \text{dt}
    $$

    For each neuron and time step, the encoder draws a random number from a uniform
    distribution [0,1] and generates a spike if the number is less than or equal to
    the spiking probability.

    Parameters
    ----------
    in_size : Size
        Size of the input to the encoder, defining the shape of the output spike train.
    spk_type : DTypeLike, default=bool
        Data type for the generated spikes. Typically boolean for binary spikes.
    name : str, optional
        Name of the encoder brainstate.nn.Module.

    See Also
    --------
    PoissonSpike : Poisson generator with fixed firing rates.
    PoissonInput : Efficient Poisson input applied directly to a state.

    Notes
    -----
    - This encoder is particularly useful for rate-to-spike conversion in neuromorphic
      computing applications and sensory encoding tasks.
    - The statistical properties of the generated spike trains follow a Poisson process,
      where the inter-spike intervals are exponentially distributed.
    - For small time steps (dt), the number of spikes in a time window T approximately
      follows a Poisson distribution with parameter λ = rate * T.
    - Unlike PoissonSpike which has fixed rates, this encoder allows dynamic rate changes
      with every update call, making it suitable for encoding time-varying signals.
    - The independence of spike generation between time steps results in renewal process
      statistics without memory of previous spiking history.

    References
    ----------
    .. [1] Dayan, P., & Abbott, L. F. (2001). Theoretical Neuroscience:
           Computational and Mathematical Modeling of Neural Systems.
           MIT Press.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>> import numpy as np
        >>>
        >>> # Create a Poisson encoder for 10 neurons
        >>> encoder = brainpy.state.PoissonEncoder(10)
        >>>
        >>> # Generate spikes with varying firing rates
        >>> rates = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) * u.Hz
        >>> spikes = encoder.update(rates)
        >>>
        >>> # Use in a more complex processing pipeline
        >>> # First, generate rate-coded output from an analog signal
        >>> analog_values = np.random.rand(10) * 100  # values between 0 and 100
        >>> firing_rates = analog_values * u.Hz  # convert to firing rates
        >>> spike_train = encoder.update(firing_rates)
        >>>
        >>> # Feed the spikes into a spiking neural network
        >>> neuron_layer = brainpy.state.LIF(10)
        >>> neuron_layer.init_state(batch_size=1)
        >>> output_spikes = neuron_layer.update(spike_train)
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        spk_type: DTypeLike = bool,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)
        self.spk_type = spk_type

    def update(self, freqs: ArrayLike):
        spikes = brainstate.random.rand(*self.varshape) <= _per_bin_spike_prob(freqs)
        spikes = u.math.asarray(spikes, dtype=self.spk_type)
        return spikes


class PoissonInput(brainstate.nn.Module):
    r"""Poisson Input to the given state variable.

    This class provides a way to add independent Poisson-distributed spiking input
    to a target state variable. For large numbers of inputs, this implementation is
    computationally more efficient than creating separate Poisson spike generators.

    The synaptic events are generated randomly during simulation runtime and are not
    preloaded or stored in memory, which improves memory efficiency for large-scale
    simulations. All inputs target the same variable with the same frequency and
    synaptic weight.

    The Poisson process generates spikes with probability based on the frequency and
    simulation time step:

    $$
    P(\text{spike}) = \text{freq} \cdot \text{dt}
    $$

    For computational efficiency, two different methods are used for spike generation:

    1. For large numbers of inputs, a normal approximation:
       $$
       \text{inputs} \sim \mathcal{N}(\mu, \sigma^2)
       $$
       where $\mu = \text{num\_input} \cdot p$ and $\sigma^2 = \text{num\_input} \cdot p \cdot (1-p)$

    2. For smaller numbers, a direct binomial sampling:
       $$
       \text{inputs} \sim \text{Binomial}(\text{num\_input}, p)
       $$

    where $p = \text{freq} \cdot \text{dt}$ in both cases.

    Parameters
    ----------
    target : brainstate.nn.Prefetch
        The variable that is targeted by this input. Should be an instance of
        :py:class:`brainstate.State` that's brainstate.nn.Prefetched via the target mechanism.
    indices : Union[np.ndarray, jax.Array]
        Indices of the target to receive input. If None, input is applied to the entire target.
    num_input : int
        The number of independent Poisson input sources.
    freq : Union[int, float]
        The firing frequency of each input source in Hz.
    weight :  ndarray, float, or brainunit.Quantity
        The synaptic weight of each input spike.
    name : Optional[str], optional
        The name of this brainstate.nn.Module.

    See Also
    --------
    PoissonSpike : Poisson spike generator as a neuron group.
    PoissonEncoder : Poisson encoder for rate-to-spike conversion.
    poisson_input : Functional version of Poisson input generation.

    Notes
    -----
    - The Poisson inputs are statistically independent between update steps and across
      target neurons.
    - This implementation is particularly efficient for large numbers of inputs or targets.
    - For very sparse connectivity patterns, consider using individual PoissonSpike neurons
      with specific connectivity patterns instead.
    - The update method internally calls the poisson_input function which handles the
      spike generation and target state updates.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>> import numpy as np
        >>>
        >>> # Create a neuron group with membrane potential
        >>> neuron = brainpy.state.LIF(100)
        >>> neuron.init_state(batch_size=1)
        >>>
        >>> # Add Poisson input to all neurons
        >>> poisson_in = brainpy.state.PoissonInput(
        ...     target=neuron.V,
        ...     indices=None,
        ...     num_input=200,
        ...     freq=50 * u.Hz,
        ...     weight=0.1 * u.mV
        ... )
        >>>
        >>> # Add Poisson input only to specific neurons
        >>> indices = np.array([0, 10, 20, 30])
        >>> specific_input = brainpy.state.PoissonInput(
        ...     target=neuron.V,
        ...     indices=indices,
        ...     num_input=50,
        ...     freq=100 * u.Hz,
        ...     weight=0.2 * u.mV
        ... )
        >>>
        >>> # Run simulation with the inputs
        >>> for t in range(100):
        ...     poisson_in.update()
        ...     specific_input.update()
        ...     neuron.update()
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        target: brainstate.nn.Prefetch,
        indices: np.ndarray | jax.Array,
        num_input: int,
        freq: u.Quantity[u.Hz],
        weight: jax.typing.ArrayLike | u.Quantity,
        name: str | None = None,
    ):
        super().__init__(name=name)

        self.target = target
        self.indices = indices
        self.num_input = num_input
        self.freq = freq
        self.weight = weight

    def update(self):
        target_state = getattr(self.target.module, self.target.item)

        # generate Poisson input
        poisson_input(
            self.freq,
            self.num_input,
            self.weight,
            target_state,
            self.indices,
        )


@set_module_as('brainpy.state')
def poisson_input(
    freq: u.Quantity[u.Hz],
    num_input: int,
    weight: jax.typing.ArrayLike | u.Quantity,
    target: brainstate.State,
    indices: np.ndarray | jax.Array | None = None,
    refractory: jax.Array | None = None,
):
    r"""Generates Poisson-distributed input spikes to a target state variable.

    This function simulates Poisson input to a given state, updating the target
    variable with generated spikes based on the specified frequency, number of inputs,
    and synaptic weight. The input can be applied to specific indices of the target
    or to the entire target if indices are not provided.

    The function uses two different methods to generate the Poisson-distributed input:
    1. For large numbers of inputs (a > 5 and b > 5), a normal approximation is used
    2. For smaller numbers, a direct binomial sampling approach is used

    Mathematical model for Poisson input:
    $$
    P(\text{spike}) = \text{freq} \cdot \text{dt}
    $$

    For the normal approximation (when a > 5 and b > 5):
    $$
    \text{inputs} \sim \mathcal{N}(a, b \cdot p)
    $$
    where:
    $$
    a = \text{num\_input} \cdot p
    $$
    $$
    b = \text{num\_input} \cdot (1 - p)
    $$
    $$
    p = \text{freq} \cdot \text{dt}
    $$

    For direct binomial sampling (when a ≤ 5 or b ≤ 5):
    $$
    \text{inputs} \sim \text{Binomial}(\text{num\_input}, p)
    $$

    Parameters
    ----------
    freq : u.Quantity[u.Hz]
        The frequency of the Poisson input in Hertz.
    num_input : int
        The number of input channels or neurons generating the Poisson spikes.
    weight : u.Quantity
        The synaptic weight applied to each spike.
    target : State
        The target state variable to which the Poisson input is applied.
    indices : Optional[Union[np.ndarray, jax.Array]], optional
        Specific indices of the target to apply the input. If None, the input is applied
        to the entire target.
    refractory : Optional[Union[jax.Array]], optional
        A boolean array indicating which parts of the target are in a refractory state
        and should not be updated. Should be the same length as the target.

    See Also
    --------
    PoissonInput : Object-oriented wrapper around this function.
    PoissonSpike : Poisson spike generator as a neuron group.

    Notes
    -----
    - The function automatically switches between normal approximation and binomial
      sampling based on the input parameters to optimize computation efficiency.
    - For large numbers of inputs, the normal approximation provides significant
      performance improvements.
    - The weight parameter is applied uniformly to all generated spikes.
    - When refractory is provided, the corresponding target elements are not updated.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>> import numpy as np
        >>>
        >>> # Create a membrane potential state
        >>> V = brainstate.HiddenState(np.zeros(100) * u.mV)
        >>>
        >>> # Add Poisson input to all neurons at 50 Hz
        >>> brainpy.state.poisson_input(
        ...     freq=50 * u.Hz,
        ...     num_input=200,
        ...     weight=0.1 * u.mV,
        ...     target=V
        ... )
        >>>
        >>> # Apply Poisson input only to a subset of neurons
        >>> indices = np.array([0, 10, 20, 30])
        >>> brainpy.state.poisson_input(
        ...     freq=100 * u.Hz,
        ...     num_input=50,
        ...     weight=0.2 * u.mV,
        ...     target=V,
        ...     indices=indices
        ... )
        >>>
        >>> # Apply input with refractory mask
        >>> refractory = np.zeros(100, dtype=bool)
        >>> refractory[40:60] = True  # neurons 40-59 are in refractory period
        >>> brainpy.state.poisson_input(
        ...     freq=75 * u.Hz,
        ...     num_input=100,
        ...     weight=0.15 * u.mV,
        ...     target=V,
        ...     refractory=refractory
        ... )
    """
    freq = brainstate.maybe_state(freq)
    weight = brainstate.maybe_state(weight)

    assert isinstance(target, brainstate.State), 'The target must be a State.'
    p = _per_bin_spike_prob(freq)
    a = num_input * p
    b = num_input * (1 - p)
    tar_val = target.value
    cond = u.math.logical_and(a > 5, b > 5)
    std = u.math.sqrt(b * p)  # std = sqrt(n * p * (1-p))

    if indices is None:
        # generate Poisson input
        branch1 = jax.tree.map(
            lambda tar: u.math.maximum(
                u.math.round(brainstate.random.normal(a, std, tar.shape, dtype=tar.dtype)),
                0.,
            ),
            tar_val,
            is_leaf=u.math.is_quantity
        )
        branch2 = jax.tree.map(
            lambda tar: brainstate.random.binomial(
                num_input,
                p,
                tar.shape,
                check_valid=False,
                dtype=tar.dtype
            ),
            tar_val,
            is_leaf=u.math.is_quantity,
        )

        inp = jax.tree.map(
            lambda b1, b2: u.math.where(cond, b1, b2),
            branch1,
            branch2,
            is_leaf=u.math.is_quantity,
        )

        # update target variable
        data = jax.tree.map(
            lambda tar, x: tar + x * weight,
            target.value,
            inp,
            is_leaf=u.math.is_quantity
        )

    else:
        # generate Poisson input
        branch1 = jax.tree.map(
            lambda tar: u.math.maximum(
                u.math.round(brainstate.random.normal(a, std, tar[indices].shape, dtype=tar.dtype)),
                0.,
            ),
            tar_val,
            is_leaf=u.math.is_quantity
        )
        branch2 = jax.tree.map(
            lambda tar: brainstate.random.binomial(
                num_input,
                p,
                tar[indices].shape,
                check_valid=False,
                dtype=tar.dtype
            ),
            tar_val,
            is_leaf=u.math.is_quantity
        )

        inp = jax.tree.map(
            lambda b1, b2: u.math.where(cond, b1, b2),
            branch1,
            branch2,
            is_leaf=u.math.is_quantity,
        )

        # update target variable
        data = jax.tree.map(
            lambda x, tar: tar.at[indices].add(x * weight),
            inp,
            tar_val,
            is_leaf=u.math.is_quantity
        )

    if refractory is not None:
        target.value = jax.tree.map(
            lambda x, tar: u.math.where(refractory, tar, x),
            data,
            tar_val,
            is_leaf=u.math.is_quantity
        )
    else:
        target.value = data


# ---------------------------------------------------------------------------
# Analog current generators
#
# Per-step ``Module`` form (not the offline arrays produced by
# ``braintools.input``): each :meth:`update` returns the instantaneous current
# at the network's current time, so the signal can be wired into a population
# via ``add_current_input`` and stays synchronized with the simulation clock.
# Scalar parameters are broadcast to ``in_size``; the stochastic generators draw
# independent samples per element.
# ---------------------------------------------------------------------------


class SectionInput(brainstate.nn.Dynamics):
    r"""Piecewise-constant current built from sections of given value and duration.

    Each section ``k`` holds current ``values[k]`` for ``durations[k]``; the
    output is zero once the total duration has elapsed.

    Parameters
    ----------
    in_size : Size
        Output geometry. Scalar section values are broadcast to this shape.
    values : sequence of Quantity
        Per-section current amplitudes (each a scalar current quantity).
    durations : sequence of Quantity
        Per-section durations; same length as ``values``.
    name : str, optional
        Module name.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy, brainunit as u
        >>> stim = brainpy.state.SectionInput(1, [0, 10, 0] * u.pA, [50, 100, 50] * u.ms)
    """
    __module__ = 'brainpy.state'

    def __init__(self, in_size, values, durations, name=None):
        super().__init__(in_size=in_size, name=name)
        if len(values) != len(durations):
            raise ValueError(f'"values" and "durations" must be the same length, '
                             f'got {len(values)} != {len(durations)}.')
        values = u.math.asarray(values)
        self._mag, self._unit = u.split_mantissa_unit(values)
        self._time_unit = u.get_unit(u.math.asarray(durations))
        dur = u.math.asarray(durations).to_decimal(self._time_unit)
        self._cum = jnp.asarray(np.cumsum(np.asarray(dur)))

    def update(self):
        t_d = _now().to_decimal(self._time_unit)
        sec = jnp.searchsorted(self._cum, t_d, side='right')
        in_win = sec < self._cum.shape[0]
        mag = jnp.where(in_win, self._mag[jnp.clip(sec, 0, self._cum.shape[0] - 1)], 0.0)
        return u.maybe_decimal(jnp.broadcast_to(mag, self.varshape) * self._unit)


class ConstantInput(brainstate.nn.Dynamics):
    r"""Constant current for a fixed duration, zero afterwards.

    Parameters
    ----------
    in_size : Size
        Output geometry; ``value`` is broadcast to this shape.
    value : Quantity
        Current amplitude.
    duration : Quantity
        Time window ``[0, duration)`` over which the current is applied.
    name : str, optional
        Module name.
    """
    __module__ = 'brainpy.state'

    def __init__(self, in_size, value, duration, name=None):
        super().__init__(in_size=in_size, name=name)
        self._mag, self._unit = u.split_mantissa_unit(value)
        self._duration = duration

    def update(self):
        t = _now()
        in_win = t < self._duration
        mag = jnp.where(in_win, self._mag, 0.0)
        return u.maybe_decimal(jnp.broadcast_to(mag, self.varshape) * self._unit)


class StepInput(brainstate.nn.Dynamics):
    r"""Staircase current: amplitude ``amplitudes[k]`` from ``step_times[k]`` onward.

    Before the first step time the output is zero; after the last step time the
    final amplitude is held.

    Parameters
    ----------
    in_size : Size
        Output geometry; scalar amplitudes are broadcast to this shape.
    amplitudes : sequence of Quantity
        Current amplitude for each step.
    step_times : sequence of Quantity
        Time at which each amplitude begins. Sorted automatically.
    name : str, optional
        Module name.
    """
    __module__ = 'brainpy.state'

    def __init__(self, in_size, amplitudes, step_times, name=None):
        super().__init__(in_size=in_size, name=name)
        if len(amplitudes) != len(step_times):
            raise ValueError(f'"amplitudes" and "step_times" must be the same length, '
                             f'got {len(amplitudes)} != {len(step_times)}.')
        amplitudes = u.math.asarray(amplitudes)
        self._mag, self._unit = u.split_mantissa_unit(amplitudes)
        self._time_unit = u.get_unit(u.math.asarray(step_times))
        times = np.asarray(u.math.asarray(step_times).to_decimal(self._time_unit))
        order = np.argsort(times)
        self._times = jnp.asarray(times[order])
        self._mag = self._mag[order]

    def update(self):
        t_d = _now().to_decimal(self._time_unit)
        idx = jnp.searchsorted(self._times, t_d, side='right') - 1
        in_win = idx >= 0
        mag = jnp.where(in_win, self._mag[jnp.clip(idx, 0, self._times.shape[0] - 1)], 0.0)
        return u.maybe_decimal(jnp.broadcast_to(mag, self.varshape) * self._unit)


class RampInput(brainstate.nn.Dynamics):
    r"""Linear ramp from ``c_start`` to ``c_end`` over ``[t_start, t_start + duration)``.

    The output is zero outside that window.

    Parameters
    ----------
    in_size : Size
        Output geometry; the ramp is broadcast to this shape.
    c_start, c_end : Quantity
        Start and end current amplitudes (same units).
    duration : Quantity
        Length of the ramp window.
    t_start : Quantity, optional
        When the ramp begins (default ``0``).
    name : str, optional
        Module name.
    """
    __module__ = 'brainpy.state'

    def __init__(self, in_size, c_start, c_end, duration, t_start=None, name=None):
        super().__init__(in_size=in_size, name=name)
        self._c_start, self._unit = u.split_mantissa_unit(c_start)
        self._c_end = u.Quantity(c_end).to_decimal(self._unit)
        self._time_unit = u.get_unit(duration)
        self._duration = u.Quantity(duration).to_decimal(self._time_unit)
        self._t_start = 0.0 if t_start is None else u.Quantity(t_start).to_decimal(self._time_unit)

    def update(self):
        t_d = _now().to_decimal(self._time_unit)
        frac = (t_d - self._t_start) / self._duration
        in_win = (t_d >= self._t_start) & (t_d < self._t_start + self._duration)
        mag = jnp.where(in_win, self._c_start + (self._c_end - self._c_start) * frac, 0.0)
        return u.maybe_decimal(jnp.broadcast_to(mag, self.varshape) * self._unit)


class SinusoidalInput(brainstate.nn.Dynamics):
    r"""Sinusoidal current ``bias + amplitude * sin(2*pi*frequency*t)``.

    The phase starts at zero; the output is zero outside ``[0, duration)``.

    Parameters
    ----------
    in_size : Size
        Output geometry; the waveform is broadcast to this shape.
    amplitude : Quantity
        Peak current amplitude.
    frequency : Quantity
        Oscillation frequency (Hz).
    duration : Quantity
        Time window over which the drive is active.
    bias : Quantity, optional
        Constant current offset added within the window (default none).
    name : str, optional
        Module name.
    """
    __module__ = 'brainpy.state'

    def __init__(self, in_size, amplitude, frequency, duration, bias=None, name=None):
        super().__init__(in_size=in_size, name=name)
        self._amp, self._unit = u.split_mantissa_unit(amplitude)
        self._bias = 0.0 if bias is None else u.Quantity(bias).to_decimal(self._unit)
        self._freq_hz = u.Quantity(frequency).to_decimal(u.Hz)
        self._duration_s = u.Quantity(duration).to_decimal(u.second)

    def update(self):
        t_s = _now().to_decimal(u.second)
        in_win = (t_s >= 0.) & (t_s < self._duration_s)
        sig = self._amp * jnp.sin(2 * jnp.pi * self._freq_hz * t_s) + self._bias
        mag = jnp.where(in_win, sig, 0.0)
        return u.maybe_decimal(jnp.broadcast_to(mag, self.varshape) * self._unit)


class WienerProcessInput(brainstate.nn.Dynamics):
    r"""White-noise (Wiener increment) current ``sigma * sqrt(dt) * N(0, 1)`` per step.

    Independent samples are drawn per element each step (a memoryless process);
    the output is zero outside ``[0, duration)``. The increment scales with
    ``sqrt(dt)`` so its variance is ``sigma**2 * dt`` regardless of step size.

    Parameters
    ----------
    in_size : Size
        Number of independent noise channels.
    sigma : Quantity
        Noise standard-deviation scale (current units).
    duration : Quantity
        Time window over which the noise is active.
    name : str, optional
        Module name.
    """
    __module__ = 'brainpy.state'

    def __init__(self, in_size, sigma, duration, name=None):
        super().__init__(in_size=in_size, name=name)
        self._sigma, self._unit = u.split_mantissa_unit(sigma)
        self._duration = duration

    def update(self):
        dt_mag, time_unit = u.split_mantissa_unit(brainstate.environ.get_dt())
        in_win = _now() < self._duration
        noise = brainstate.random.randn(*self.varshape) * self._sigma * jnp.sqrt(dt_mag)
        mag = jnp.where(in_win, noise, 0.0)
        return u.maybe_decimal(mag * self._unit)


class OUProcessInput(brainstate.nn.Dynamics):
    r"""Ornstein-Uhlenbeck (coloured-noise) current.

    Integrates :math:`dx = (mean - x)\,dt/\tau + \sigma\,\sqrt{dt}\,N(0, 1)`,
    initialised at ``mean``; the output is zero outside ``[0, duration)``. The
    discretization matches :func:`braintools.input.ou_process`, so the
    steady-state variance is ``sigma**2 * tau / 2``.

    Parameters
    ----------
    in_size : Size
        Number of independent OU channels.
    mean : Quantity
        Asymptotic mean (drift) current.
    sigma : Quantity
        Noise scale (current units).
    tau : Quantity
        Relaxation time constant.
    duration : Quantity
        Time window over which the process is active.
    name : str, optional
        Module name.
    """
    __module__ = 'brainpy.state'

    def __init__(self, in_size, mean, sigma, tau, duration, name=None):
        super().__init__(in_size=in_size, name=name)
        self._mean, self._unit = u.split_mantissa_unit(mean)
        self._sigma = u.Quantity(sigma).to_decimal(self._unit)
        self._tau = tau
        self._duration = duration

    def init_state(self, *args, **kwargs):
        self.x = brainstate.HiddenState(jnp.broadcast_to(jnp.asarray(self._mean, dtype=float),
                                                         self.varshape))

    def update(self):
        dt = brainstate.environ.get_dt()
        dt_mag, time_unit = u.split_mantissa_unit(dt)
        tau_d = u.Quantity(self._tau).to_decimal(time_unit)
        in_win = _now() < self._duration
        x = self.x.value
        noise = brainstate.random.randn(*self.varshape)
        x_new = x + (self._mean - x) * (dt_mag / tau_d) + self._sigma * jnp.sqrt(dt_mag) * noise
        x_upd = jnp.where(in_win, x_new, x)
        self.x.value = x_upd
        mag = jnp.where(in_win, x_upd, 0.0)
        return u.maybe_decimal(mag * self._unit)
