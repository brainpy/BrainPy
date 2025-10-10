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

# -*- coding: utf-8 -*-

from typing import Callable

import brainstate
import braintools
import brainunit as u
import jax
from brainstate.typing import ArrayLike, Size

from ._base import Neuron

__all__ = [
    'IF', 'LIF', 'LIFRef', 'ALIF',
]


class IF(Neuron):
    r"""Integrate-and-Fire (IF) neuron model.

    This class implements the classic Integrate-and-Fire neuron model, one of the simplest
    spiking neuron models. It accumulates input current until the membrane potential reaches
    a threshold, at which point it fires a spike and resets the potential.

    The model is characterized by the following differential equation:

    $$
    \tau \frac{dV}{dt} = -V + R \cdot I(t)
    $$

    Spike condition:
    If $V \geq V_{th}$: emit spike and reset $V = V - V_{th}$ (soft reset) or $V = 0$ (hard reset)

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=1. * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=5. * u.ms
        Membrane time constant.
    V_th : ArrayLike, default=1. * u.mV
        Firing threshold voltage (should be positive).
    V_initializer : Callable
        Initializer for the membrane potential state.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function for the non-differentiable spike generation.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation:
        - 'soft': subtract threshold V = V - V_th
        - 'hard': strict reset using stop_gradient
    name : str, optional
        Name of the neuron layer.

    Attributes
    ----------
    V : HiddenState
        Membrane potential.

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create an IF neuron layer with 10 neurons
    >>> if_neuron = brainpy.state.IF(10, tau=8*u.ms, V_th=1.2*u.mV)
    >>>
    >>> # Initialize the state
    >>> if_neuron.init_state(batch_size=1)
    >>>
    >>> # Apply an input current and update the neuron state
    >>> spikes = if_neuron.update(x=2.0*u.mA)
    >>>
    >>> # Create a network with IF neurons
    >>> network = brainstate.nn.Sequential([
    ...     brainpy.state.IF(100, tau=5.0*u.ms),
    ...     brainstate.nn.Linear(100, 10)
    ... ])

    Notes
    -----
    - Unlike the LIF model, the IF model has no leak towards a resting potential.
    - The membrane potential decays exponentially with time constant tau in the absence of input.
    - The time-dependent dynamics are integrated using an exponential Euler method.
    - The IF model is perfect integrator in the sense that it accumulates input indefinitely
      until reaching threshold, without any leak current.

    References
    ----------
    .. [1] Lapicque, L. (1907). Recherches quantitatives sur l'excitation électrique
           des nerfs traitée comme une polarisation. Journal de Physiologie et de
           Pathologie Générale, 9, 620-635.
    .. [2] Abbott, L. F. (1999). Lapicque's introduction of the integrate-and-fire
           model neuron (1907). Brain Research Bulletin, 50(5-6), 303-304.
    .. [3] Burkitt, A. N. (2006). A review of the integrate-and-fire neuron model:
           I. Homogeneous synaptic input. Biological cybernetics, 95(1), 1-19.
    """

    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        R: brainstate.typing.ArrayLike = 1. * u.ohm,
        tau: brainstate.typing.ArrayLike = 5. * u.ms,
        V_th: brainstate.typing.ArrayLike = 1. * u.mV,  # should be positive
        V_initializer: Callable = braintools.init.Constant(0. * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = braintools.init.param(R, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_initializer = V_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)

    def get_spike(self, V=None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / self.V_th
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        # reset
        last_V = self.V.value
        last_spike = self.get_spike(self.V.value)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_V)
        V = last_V - V_th * last_spike
        # membrane potential
        dv = lambda v: (-v + self.R * self.sum_current_inputs(x, v)) / self.tau
        V = brainstate.nn.exp_euler_step(dv, V)
        V = self.sum_delta_inputs(V)
        self.V.value = V
        return self.get_spike(V)


class LIF(Neuron):
    r"""Leaky Integrate-and-Fire (LIF) neuron model.

    This class implements the Leaky Integrate-and-Fire neuron model, which extends the basic
    Integrate-and-Fire model by adding a leak term. The leak causes the membrane potential
    to decay towards a resting value in the absence of input, making the model more
    biologically plausible.

    The model is characterized by the following differential equation:

    $$
    \tau \frac{dV}{dt} = -(V - V_{rest}) + R \cdot I(t)
    $$

    Spike condition:
    If $V \geq V_{th}$: emit spike and reset $V = V_{reset}$

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=1. * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=5. * u.ms
        Membrane time constant.
    V_th : ArrayLike, default=1. * u.mV
        Firing threshold voltage.
    V_reset : ArrayLike, default=0. * u.mV
        Reset voltage after spike.
    V_rest : ArrayLike, default=0. * u.mV
        Resting membrane potential.
    V_initializer : Callable
        Initializer for the membrane potential state.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function for the non-differentiable spike generation.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation:
        - 'soft': subtract threshold V = V - V_th
        - 'hard': strict reset using stop_gradient
    name : str, optional
        Name of the neuron layer.

    Attributes
    ----------
    V : HiddenState
        Membrane potential.

    Examples
    --------
    >>> import brainpy.state as brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create a LIF neuron layer with 10 neurons
    >>> lif = brainpy.LIF(10, tau=10*u.ms, V_th=0.8*u.mV)
    >>>
    >>> # Initialize the state
    >>> lif.init_state(batch_size=1)
    >>>
    >>> # Apply an input current and update the neuron state
    >>> spikes = lif.update(x=1.5*u.mA)

    Notes
    -----
    - The leak term causes the membrane potential to decay exponentially towards V_rest
      with time constant tau when no input is present.
    - The time-dependent dynamics are integrated using an exponential Euler method.
    - Spike generation is non-differentiable, so surrogate gradients are used for
      backpropagation during training.

    References
    ----------
    .. [1] Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014).
           Neuronal dynamics: From single neurons to networks and models of cognition.
           Cambridge University Press.
    .. [2] Burkitt, A. N. (2006). A review of the integrate-and-fire neuron model:
           I. Homogeneous synaptic input. Biological cybernetics, 95(1), 1-19.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 1. * u.ohm,
        tau: ArrayLike = 5. * u.ms,
        V_th: ArrayLike = 1. * u.mV,
        V_reset: ArrayLike = 0. * u.mV,
        V_rest: ArrayLike = 0. * u.mV,
        V_initializer: Callable = braintools.init.Constant(0. * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = braintools.init.param(R, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_rest = braintools.init.param(V_rest, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.V_initializer = V_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        last_v = self.V.value
        lst_spk = self.get_spike(last_v)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        V = last_v - (V_th - self.V_reset) * lst_spk
        # membrane potential
        dv = lambda v: (-v + self.V_rest + self.R * self.sum_current_inputs(x, v)) / self.tau
        V = brainstate.nn.exp_euler_step(dv, V)
        V = self.sum_delta_inputs(V)
        self.V.value = V
        return self.get_spike(V)


class LIFRef(Neuron):
    r"""Leaky Integrate-and-Fire neuron model with refractory period.

    This class implements a Leaky Integrate-and-Fire neuron model that includes a
    refractory period after spiking, during which the neuron cannot fire regardless
    of input. This better captures the behavior of biological neurons that exhibit
    a recovery period after action potential generation.

    The model is characterized by the following equations:

    When not in refractory period:
    $$
    \tau \frac{dV}{dt} = -(V - V_{rest}) + R \cdot I(t)
    $$

    During refractory period:
    $$
    V = V_{reset}
    $$

    Spike condition:
    If $V \geq V_{th}$: emit spike, set $V = V_{reset}$, and enter refractory period for $\tau_{ref}$

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=1. * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=5. * u.ms
        Membrane time constant.
    tau_ref : ArrayLike, default=5. * u.ms
        Refractory period duration.
    V_th : ArrayLike, default=1. * u.mV
        Firing threshold voltage.
    V_reset : ArrayLike, default=0. * u.mV
        Reset voltage after spike.
    V_rest : ArrayLike, default=0. * u.mV
        Resting membrane potential.
    V_initializer : Callable
        Initializer for the membrane potential state.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function for the non-differentiable spike generation.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation:
        - 'soft': subtract threshold V = V - V_th
        - 'hard': strict reset using stop_gradient
    name : str, optional
        Name of the neuron layer.

    Attributes
    ----------
    V : HiddenState
        Membrane potential.
    last_spike_time : ShortTermState
        Time of the last spike, used to implement refractory period.

    Examples
    --------
    >>> import brainpy.state as brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create a LIFRef neuron layer with 10 neurons
    >>> lifref = brainpy.LIFRef(10,
    ...                         tau=10*u.ms,
    ...                         tau_ref=5*u.ms,
    ...                         V_th=0.8*u.mV)
    >>>
    >>> # Initialize the state
    >>> lifref.init_state(batch_size=1)
    >>>
    >>> # Apply an input current and update the neuron state
    >>> spikes = lifref.update(x=1.5*u.mA)
    >>>
    >>> # Create a network with refractory neurons
    >>> network = brainstate.nn.Sequential([
    ...     brainpy.LIFRef(100, tau_ref=4*u.ms),
    ...     brainstate.nn.Linear(100, 10)
    ... ])

    Notes
    -----
    - The refractory period is implemented by tracking the time of the last spike
      and preventing membrane potential updates if the elapsed time is less than tau_ref.
    - During the refractory period, the membrane potential remains at the reset value
      regardless of input current strength.
    - Refractory periods prevent high-frequency repetitive firing and are critical
      for realistic neural dynamics.
    - The time-dependent dynamics are integrated using an exponential Euler method.
    - The simulation environment time variable 't' is used to track the refractory state.

    References
    ----------
    .. [1] Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014).
           Neuronal dynamics: From single neurons to networks and models of cognition.
           Cambridge University Press.
    .. [2] Burkitt, A. N. (2006). A review of the integrate-and-fire neuron model:
           I. Homogeneous synaptic input. Biological cybernetics, 95(1), 1-19.
    .. [3] Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on
           neural networks, 14(6), 1569-1572.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 1. * u.ohm,
        tau: ArrayLike = 5. * u.ms,
        tau_ref: ArrayLike = 5. * u.ms,
        V_th: ArrayLike = 1. * u.mV,
        V_reset: ArrayLike = 0. * u.mV,
        V_rest: ArrayLike = 0. * u.mV,
        V_initializer: Callable = braintools.init.Constant(0. * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = braintools.init.param(R, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.tau_ref = braintools.init.param(tau_ref, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_rest = braintools.init.param(V_rest, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.V_initializer = V_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.last_spike_time = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        )

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        t = brainstate.environ.get('t')
        last_v = self.V.value
        lst_spk = self.get_spike(last_v)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        last_v = last_v - (V_th - self.V_reset) * lst_spk
        # membrane potential
        dv = lambda v: (-v + self.V_rest + self.R * self.sum_current_inputs(x, v)) / self.tau
        V = brainstate.nn.exp_euler_step(dv, last_v)
        V = self.sum_delta_inputs(V)
        self.V.value = u.math.where(t - self.last_spike_time.value < self.tau_ref, last_v, V)
        # spike time evaluation
        lst_spk_time = u.math.where(
            self.V.value >= self.V_th, brainstate.environ.get('t'), self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(lst_spk_time)
        return self.get_spike()


class ALIF(Neuron):
    r"""Adaptive Leaky Integrate-and-Fire (ALIF) neuron model.

    This class implements the Adaptive Leaky Integrate-and-Fire neuron model, which extends
    the basic LIF model by adding an adaptation variable. This adaptation mechanism increases
    the effective firing threshold after each spike, allowing the neuron to exhibit
    spike-frequency adaptation - a common feature in biological neurons that reduces
    firing rate during sustained stimulation.

    The model is characterized by the following differential equations:

    $$
    \tau \frac{dV}{dt} = -(V - V_{rest}) + R \cdot I(t)
    $$

    $$
    \tau_a \frac{da}{dt} = -a
    $$

    Spike condition:
    If $V \geq V_{th} + \beta \cdot a$: emit spike, set $V = V_{reset}$, and increment $a = a + 1$

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=1. * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=5. * u.ms
        Membrane time constant.
    tau_a : ArrayLike, default=100. * u.ms
        Adaptation time constant (typically much longer than tau).
    V_th : ArrayLike, default=1. * u.mV
        Base firing threshold voltage.
    V_reset : ArrayLike, default=0. * u.mV
        Reset voltage after spike.
    V_rest : ArrayLike, default=0. * u.mV
        Resting membrane potential.
    beta : ArrayLike, default=0.1 * u.mV
        Adaptation coupling parameter that scales the effect of the adaptation variable.
    spk_fun : Callable
        Surrogate gradient function for the non-differentiable spike generation.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation:

        - 'soft': subtract threshold V = V - V_th
        - 'hard': strict reset using stop_gradient
    V_initializer : Callable
        Initializer for the membrane potential state.
    a_initializer : Callable
        Initializer for the adaptation variable.
    name : str, optional
        Name of the neuron layer.

    Attributes
    ----------
    V : HiddenState
        Membrane potential.
    a : HiddenState
        Adaptation variable that increases after each spike and decays exponentially.

    Examples
    --------
    >>> import brainpy.state as brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create an ALIF neuron layer with 10 neurons
    >>> alif = brainpy.ALIF(10,
    ...                     tau=10*u.ms,
    ...                     tau_a=200*u.ms,
    ...                     beta=0.2*u.mV)
    >>>
    >>> # Initialize the state
    >>> alif.init_state(batch_size=1)
    >>>
    >>> # Apply an input current and update the neuron state
    >>> spikes = alif.update(x=1.5*u.mA)
    >>>
    >>> # Create a network with adaptation for burst detection
    >>> network = brainstate.nn.Sequential([
    ...     brainpy.ALIF(100, tau_a=150*u.ms, beta=0.3*u.mV),
    ...     brainstate.nn.Linear(100, 10)
    ... ])

    Notes
    -----
    - The adaptation variable 'a' increases by 1 with each spike and decays exponentially
      with time constant tau_a between spikes.
    - The effective threshold increases by beta*a, making it progressively harder for the
      neuron to fire when it has recently been active.
    - This adaptation mechanism creates spike-frequency adaptation, allowing the neuron
      to respond strongly to input onset but then reduce its firing rate even if the
      input remains constant.
    - The adaptation time constant tau_a is typically much larger than the membrane time
      constant tau, creating a longer-lasting adaptation effect.
    - The time-dependent dynamics are integrated using an exponential Euler method.

    References
    ----------
    .. [1] Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014).
           Neuronal dynamics: From single neurons to networks and models of cognition.
           Cambridge University Press.
    .. [2] Brette, R., & Gerstner, W. (2005). Adaptive exponential integrate-and-fire model
           as an effective description of neuronal activity. Journal of neurophysiology,
           94(5), 3637-3642.
    .. [3] Naud, R., Marcille, N., Clopath, C., & Gerstner, W. (2008). Firing patterns in
           the adaptive exponential integrate-and-fire model. Biological cybernetics,
           99(4), 335-347.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 1. * u.ohm,
        tau: ArrayLike = 5. * u.ms,
        tau_a: ArrayLike = 100. * u.ms,
        V_th: ArrayLike = 1. * u.mV,
        V_reset: ArrayLike = 0. * u.mV,
        V_rest: ArrayLike = 0. * u.mV,
        beta: ArrayLike = 0.1 * u.mV,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        V_initializer: Callable = braintools.init.Constant(0. * u.mV),
        a_initializer: Callable = braintools.init.Constant(0.),
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = braintools.init.param(R, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.tau_a = braintools.init.param(tau_a, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.V_rest = braintools.init.param(V_rest, self.varshape)
        self.beta = braintools.init.param(beta, self.varshape)

        # functions
        self.V_initializer = V_initializer
        self.a_initializer = a_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.a = brainstate.HiddenState(braintools.init.param(self.a_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.a.value = braintools.init.param(self.a_initializer, self.varshape, batch_size)

    def get_spike(self, V=None, a=None):
        V = self.V.value if V is None else V
        a = self.a.value if a is None else a
        v_scaled = (V - self.V_th - self.beta * a) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        last_v = self.V.value
        last_a = self.a.value
        lst_spk = self.get_spike(last_v, last_a)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        V = last_v - (V_th - self.V_reset) * lst_spk
        a = last_a + lst_spk
        # membrane potential
        dv = lambda v: (-v + self.V_rest + self.R * self.sum_current_inputs(x, v)) / self.tau
        da = lambda a: -a / self.tau_a
        V = brainstate.nn.exp_euler_step(dv, V)
        a = brainstate.nn.exp_euler_step(da, a)
        self.V.value = self.sum_delta_inputs(V)
        self.a.value = a
        return self.get_spike(self.V.value, self.a.value)
