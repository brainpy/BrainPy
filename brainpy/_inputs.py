# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

from typing import Union, Optional, Sequence, Callable

import braintools
import brainunit as u
import jax
import numpy as np

import brainstate
from brainstate.typing import ArrayLike, Size, DTypeLike
from brainpy._misc import set_module_as


__all__ = [
    'SpikeTime',
    'PoissonSpike',
    'PoissonEncoder',
    'PoissonInput',
    'poisson_input',
]


class SpikeTime(brainstate.nn.Dynamics):
    """The input neuron group characterized by spikes emitting at given times.

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

    Parameters
    ----------
    in_size : int, tuple, list
        The neuron group geometry.
    indices : list, tuple, ArrayType
        The neuron indices at each time point to emit spikes.
    times : list, tuple, ArrayType
        The time points which generate the spikes.
    name : str, optional
        The name of the dynamic system.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        indices: Union[Sequence, ArrayLike],
        times: Union[Sequence, ArrayLike],
        spk_type: DTypeLike = bool,
        name: Optional[str] = None,
        need_sort: bool = True,
    ):
        super().__init__(in_size=in_size, name=name)

        # parameters
        if len(indices) != len(times):
            raise ValueError(f'The length of "indices" and "times" must be the same. '
                             f'However, we got {len(indices)} != {len(times)}.')
        self.num_times = len(times)
        self.spk_type = spk_type

        # data about times and indices
        self.times = u.math.asarray(times)
        self.indices = u.math.asarray(indices, dtype=brainstate.environ.ditype())
        if need_sort:
            sort_idx = u.math.argsort(self.times)
            self.indices = self.indices[sort_idx]
            self.times = self.times[sort_idx]

    def init_state(self, *args, **kwargs):
        self.i = brainstate.ShortTermState(-1)

    def reset_state(self, batch_size=None, **kwargs):
        self.i.value = -1

    def update(self):
        t = brainstate.environ.get('t')

        def _cond_fun(spikes):
            i = self.i.value
            return u.math.logical_and(i < self.num_times, t >= self.times[i])

        def _body_fun(spikes):
            i = self.i.value
            spikes = spikes.at[..., self.indices[i]].set(True)
            self.i.value += 1
            return spikes

        spike = u.math.zeros(self.varshape, dtype=self.spk_type)
        spike = brainstate.transform.while_loop(_cond_fun, _body_fun, spike)
        return spike


class PoissonSpike(brainstate.nn.Dynamics):
    """
    Poisson Neuron Group.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        freqs: Union[ArrayLike, Callable],
        spk_type: DTypeLike = bool,
        name: Optional[str] = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.spk_type = spk_type

        # parameters
        self.freqs = braintools.init.param(freqs, self.varshape, allow_none=False)

    def update(self):
        spikes = brainstate.random.rand(*self.varshape) <= (self.freqs * brainstate.environ.get_dt())
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

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>> import numpy as np
    >>>
    >>> # Create a Poisson encoder for 10 neurons
    >>> encoder = brainpy.PoissonEncoder(10)
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
    >>> neuron_layer = brainpy.LIF(10)
    >>> neuron_layer.init_state(batch_size=1)
    >>> output_spikes = neuron_layer.update(spike_train)

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
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        spk_type: DTypeLike = bool,
        name: Optional[str] = None,
    ):
        super().__init__(in_size=in_size, name=name)
        self.spk_type = spk_type

    def update(self, freqs: ArrayLike):
        spikes = brainstate.random.rand(*self.varshape) <= (freqs * brainstate.environ.get_dt())
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

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>> import numpy as np
    >>>
    >>> # Create a neuron group with membrane potential
    >>> neuron = brainpy.LIF(100)
    >>> neuron.init_state(batch_size=1)
    >>>
    >>> # Add Poisson input to all neurons
    >>> poisson_in = brainpy.PoissonInput(
    ...     target=neuron.V,
    ...     indices=None,
    ...     num_input=200,
    ...     freq=50 * u.Hz,
    ...     weight=0.1 * u.mV
    ... )
    >>>
    >>> # Add Poisson input only to specific neurons
    >>> indices = np.array([0, 10, 20, 30])
    >>> specific_input = brainpy.PoissonInput(
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

    Notes
    -----
    - The Poisson inputs are statistically independent between update steps and across
      target neurons.
    - This implementation is particularly efficient for large numbers of inputs or targets.
    - For very sparse connectivity patterns, consider using individual PoissonSpike neurons
      with specific connectivity patterns instead.
    - The update method internally calls the poisson_input function which handles the
      spike generation and target state updates.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        target: brainstate.nn.Prefetch,
        indices: Union[np.ndarray, jax.Array],
        num_input: int,
        freq: u.Quantity[u.Hz],
        weight: Union[jax.typing.ArrayLike, u.Quantity],
        name: Optional[str] = None,
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


@set_module_as('brainpy')
def poisson_input(
    freq: u.Quantity[u.Hz],
    num_input: int,
    weight: Union[jax.typing.ArrayLike, u.Quantity],
    target: brainstate.State,
    indices: Optional[Union[np.ndarray, jax.Array]] = None,
    refractory: Optional[Union[jax.Array]] = None,
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

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>> import numpy as np
    >>>
    >>> # Create a membrane potential state
    >>> V = brainstate.HiddenState(np.zeros(100) * u.mV)
    >>>
    >>> # Add Poisson input to all neurons at 50 Hz
    >>> brainpy.poisson_input(
    ...     freq=50 * u.Hz,
    ...     num_input=200,
    ...     weight=0.1 * u.mV,
    ...     target=V
    ... )
    >>>
    >>> # Apply Poisson input only to a subset of neurons
    >>> indices = np.array([0, 10, 20, 30])
    >>> brainpy.poisson_input(
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
    >>> brainpy.poisson_input(
    ...     freq=75 * u.Hz,
    ...     num_input=100,
    ...     weight=0.15 * u.mV,
    ...     target=V,
    ...     refractory=refractory
    ... )

    Notes
    -----
    - The function automatically switches between normal approximation and binomial
      sampling based on the input parameters to optimize computation efficiency.
    - For large numbers of inputs, the normal approximation provides significant
      performance improvements.
    - The weight parameter is applied uniformly to all generated spikes.
    - When refractory is provided, the corresponding target elements are not updated.
    """
    freq = brainstate.maybe_state(freq)
    weight = brainstate.maybe_state(weight)

    assert isinstance(target, brainstate.State), 'The target must be a State.'
    p = freq * brainstate.environ.get_dt()
    p = p.to_decimal() if isinstance(p, u.Quantity) else p
    a = num_input * p
    b = num_input * (1 - p)
    tar_val = target.value
    cond = u.math.logical_and(a > 5, b > 5)

    if indices is None:
        # generate Poisson input
        branch1 = jax.tree.map(
            lambda tar: brainstate.random.normal(
                a,
                b * p,
                tar.shape,
                dtype=tar.dtype
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
            lambda tar: brainstate.random.normal(
                a,
                b * p,
                tar[indices].shape,
                dtype=tar.dtype
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
