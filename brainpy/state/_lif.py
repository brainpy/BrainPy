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
    'IF', 'LIF', 'ExpIF', 'ExpIFRef', 'AdExIF', 'AdExIFRef', 'LIFRef', 'ALIF',
    'QuaIF', 'AdQuaIF', 'AdQuaIFRef', 'Gif', 'GifRef',
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
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create a LIF neuron layer with 10 neurons
    >>> lif = brainpy.state.LIF(10, tau=10*u.ms, V_th=0.8*u.mV)
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
        last_spk = self.get_spike(last_v)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        V = last_v - (V_th - self.V_reset) * last_spk
        # membrane potential
        dv = lambda v: (-v + self.V_rest + self.R * self.sum_current_inputs(x, v)) / self.tau
        V = brainstate.nn.exp_euler_step(dv, V)
        V = self.sum_delta_inputs(V)
        self.V.value = V
        return self.get_spike(V)


class ExpIF(Neuron):
    r"""Exponential Integrate-and-Fire (ExpIF) neuron model.

    This model augments the LIF neuron by adding an exponential spike-initiation
    term, which provides a smooth approximation of the action potential onset
    and improves biological plausibility for cortical pyramidal cells.

    The membrane potential dynamics follow:

    $$
    \tau \frac{dV}{dt} = -(V - V_{rest}) + \Delta_T \exp\left(\frac{V - V_T}{\Delta_T}\right) + R \cdot I(t)
    $$

    Spike condition:
    If $V \geq V_{th}$: emit spike and reset $V = V_{reset}$ (hard reset) or
    $V = V - (V_{th} - V_{reset})$ (soft reset).

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=1. * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=10. * u.ms
        Membrane time constant.
    V_th : ArrayLike, default=-30. * u.mV
        Numerical firing threshold voltage.
    V_reset : ArrayLike, default=-68. * u.mV
        Reset voltage after spike.
    V_rest : ArrayLike, default=-65. * u.mV
        Resting membrane potential.
    V_T : ArrayLike, default=-59.9 * u.mV
        Threshold potential of the exponential term.
    delta_T : ArrayLike, default=3.48 * u.mV
        Spike slope factor controlling the sharpness of spike initiation.
    V_initializer : Callable
        Initializer for the membrane potential state.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function for the spike generation.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation.
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
    >>> # Create a ExpIF neuron layer with 10 neurons
    >>> expif = brainpy.state.ExpIF(10, tau=10*u.ms, V_th=-30*u.mV)
    >>>
    >>> # Initialize the state
    >>> expif.init_state(batch_size=1)
    >>>
    >>> # Apply an input current and update the neuron state
    >>> spikes = expif.update(x=1.5*u.mA)

    Notes
    -----
    - The model was first introduced by Nicolas Fourcaud-Trocmé, David Hansel, Carl van Vreeswijk
      and Nicolas Brunel [1]_. The exponential nonlinearity was later confirmed by Badel et al. [3]_.
      It is one of the prominent examples of a precise theoretical prediction in computational
      neuroscience that was later confirmed by experimental neuroscience.
    - The right-hand side of the above equation contains a nonlinearity
      that can be directly extracted from experimental data [3]_. In this sense the exponential
      nonlinearity is not an arbitrary choice but directly supported by experimental evidence.
    - Even though it is a nonlinear model, it is simple enough to calculate the firing
      rate for constant input, and the linear response to fluctuations, even in the presence
      of input noise [4]_.

    References
    ----------
    .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
           mechanisms determine the neuronal response to fluctuating
           inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
    .. [2] Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014).
           Neuronal dynamics: From single neurons to networks and models
           of cognition. Cambridge University Press.
    .. [3] Badel, Laurent, Sandrine Lefort, Romain Brette, Carl CH Petersen,
           Wulfram Gerstner, and Magnus JE Richardson. "Dynamic IV curves
           are reliable predictors of naturalistic pyramidal-neuron voltage
           traces." Journal of Neurophysiology 99, no. 2 (2008): 656-666.
    .. [4] Richardson, Magnus JE. "Firing-rate response of linear and nonlinear
           integrate-and-fire neurons to modulated current-based and
           conductance-based synaptic drive." Physical Review E 76, no. 2 (2007): 021919.
    .. [5] https://en.wikipedia.org/wiki/Exponential_integrate-and-fire
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 1. * u.ohm,
        tau: ArrayLike = 10. * u.ms,
        V_th: ArrayLike = -30. * u.mV,
        V_reset: ArrayLike = -68. * u.mV,
        V_rest: ArrayLike = -65. * u.mV,
        V_T: ArrayLike = -59.9 * u.mV,
        delta_T: ArrayLike = 3.48 * u.mV,
        V_initializer: Callable = braintools.init.Constant(-65. * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = braintools.init.param(R, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.V_rest = braintools.init.param(V_rest, self.varshape)
        self.V_T = braintools.init.param(V_T, self.varshape)
        self.delta_T = braintools.init.param(delta_T, self.varshape)
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
        last_spk = self.get_spike(last_v)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        V = last_v - (V_th - self.V_reset) * last_spk

        def dv(v):
            exp_term = self.delta_T * u.math.exp((v - self.V_T) / self.delta_T)
            return (-(v - self.V_rest) + exp_term + self.R * self.sum_current_inputs(x, v)) / self.tau

        V = brainstate.nn.exp_euler_step(dv, V)
        V = self.sum_delta_inputs(V)
        self.V.value = V
        return self.get_spike(V)


class ExpIFRef(Neuron):
    r"""Exponential Integrate-and-Fire neuron model with refractory mechanism.

    This neuron adds an absolute refractory period to :class:`ExpIF`. While the exponential
    spike-initiation term keeps the membrane potential dynamics smooth, the refractory
    mechanism prevents the neuron from firing within ``tau_ref`` after a spike.

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=1. * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=10. * u.ms
        Membrane time constant.
    tau_ref : ArrayLike, default=1.7 * u.ms
        Absolute refractory period duration.
    V_th : ArrayLike, default=-30. * u.mV
        Numerical firing threshold voltage.
    V_reset : ArrayLike, default=-68. * u.mV
        Reset voltage after spike.
    V_rest : ArrayLike, default=-65. * u.mV
        Resting membrane potential.
    V_T : ArrayLike, default=-59.9 * u.mV
        Threshold potential of the exponential term.
    delta_T : ArrayLike, default=3.48 * u.mV
        Spike slope factor controlling spike initiation sharpness.
    V_initializer : Callable
        Initializer for the membrane potential state.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function for the spike generation.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation.
    ref_var : bool, default=False
        Whether to expose a boolean refractory state variable.
    name : str, optional
        Name of the neuron layer.

    Attributes
    ----------
    V : HiddenState
        Membrane potential.
    last_spike_time : ShortTermState
        Last spike time recorder.
    refractory : HiddenState
        Neuron refractory state.

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create a ExpIF neuron layer with 10 neurons
    >>> expif = brainpy.state.ExpIF(10, tau=10*u.ms, V_th=-30*u.mV)
    >>>
    >>> # Initialize the state
    >>> expif.init_state(batch_size=1)
    >>>
    >>> # Generate inputs
    >>> time_steps = 100
    >>> inputs = brainstate.random.randn(time_steps, 1, 10) * u.mA
    >>>
    >>> # Apply an input current and update the neuron state
    >>> 
    >>> with brainstate.environ.context(dt=0.1 * u.ms):
    >>>     for t in range(time_steps):
    >>>         with brainstate.environ.context(t=t*0.1*u.ms):
    >>>             spikes = expif.update(x=inputs[t])
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 1. * u.ohm,
        tau: ArrayLike = 10. * u.ms,
        tau_ref: ArrayLike = 1.7 * u.ms,
        V_th: ArrayLike = -30. * u.mV,
        V_reset: ArrayLike = -68. * u.mV,
        V_rest: ArrayLike = -65. * u.mV,
        V_T: ArrayLike = -59.9 * u.mV,
        delta_T: ArrayLike = 3.48 * u.mV,
        V_initializer: Callable = braintools.init.Constant(-65. * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = braintools.init.param(R, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.tau_ref = braintools.init.param(tau_ref, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.V_rest = braintools.init.param(V_rest, self.varshape)
        self.V_T = braintools.init.param(V_T, self.varshape)
        self.delta_T = braintools.init.param(delta_T, self.varshape)
        self.V_initializer = V_initializer
        self.ref_var = ref_var

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.last_spike_time = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        )
        if self.ref_var:
            self.refractory = brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            )

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        if self.ref_var:
            self.refractory.value = braintools.init.param(
                braintools.init.Constant(False), self.varshape, batch_size
            )

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        t = brainstate.environ.get('t')
        last_v = self.V.value
        last_spk = self.get_spike(last_v)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        v_reset = last_v - (V_th - self.V_reset) * last_spk

        def dv(v):
            exp_term = self.delta_T * u.math.exp((v - self.V_T) / self.delta_T)
            return (-(v - self.V_rest) + exp_term + self.R * self.sum_current_inputs(x, v)) / self.tau

        V_candidate = brainstate.nn.exp_euler_step(dv, v_reset)
        V_candidate = self.sum_delta_inputs(V_candidate)

        refractory = (t - self.last_spike_time.value) < self.tau_ref
        self.V.value = u.math.where(refractory, v_reset, V_candidate)

        spike_cond = self.V.value >= self.V_th
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t, self.last_spike_time.value)
        )
        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(
                u.math.logical_or(refractory, spike_cond)
            )
        return self.get_spike()


class AdExIF(Neuron):
    r"""Adaptive exponential Integrate-and-Fire (AdExIF) neuron model.

    This model extends :class:`ExpIF` by adding an adaptation current ``w`` that is
    incremented after each spike and relaxes with time constant ``tau_w``. The membrane
    dynamics are governed by two coupled differential equations [1]_:

    $$
    \tau \frac{dV}{dt} = -(V - V_{rest}) + \Delta_T
    \exp\left(\frac{V - V_T}{\Delta_T}\right) - R w + R \cdot I(t)
    $$

    $$
    \tau_w \frac{dw}{dt} = a (V - V_{rest}) - w
    $$

    After each spike the membrane potential is reset and the adaptation current
    increases by ``b``. This simple mechanism generates rich firing patterns such
    as spike-frequency adaptation and bursting.

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=1. * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=10. * u.ms
        Membrane time constant.
    tau_w : ArrayLike, default=30. * u.ms
        Adaptation current time constant.
    V_th : ArrayLike, default=-55. * u.mV
        Spike threshold used for reset.
    V_reset : ArrayLike, default=-68. * u.mV
        Reset potential after spike.
    V_rest : ArrayLike, default=-65. * u.mV
        Resting membrane potential.
    V_T : ArrayLike, default=-59.9 * u.mV
        Threshold of the exponential term.
    delta_T : ArrayLike, default=3.48 * u.mV
        Spike slope factor controlling the sharpness of spike initiation.
    a : ArrayLike, default=1. * u.siemens
        Coupling strength from voltage to adaptation current.
    b : ArrayLike, default=1. * u.mA
        Increment of the adaptation current after a spike.
    V_initializer : Callable
        Initializer for the membrane potential state.
    w_initializer : Callable
        Initializer for the adaptation current.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function for the spike generation.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation.
    name : str, optional
        Name of the neuron layer.

    Attributes
    ----------
    V : HiddenState
        Membrane potential.
    w : HiddenState
        Adaptation current.

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create a AdExIF neuron layer with 10 neurons
    >>> adexif = brainpy.state.AdExIF(10, tau=10*u.ms)
    >>>
    >>> # Initialize the state
    >>> adexif.init_state(batch_size=1)
    >>>
    >>> # Apply an input current and update the neuron state
    >>> spikes = adexif.update(x=1.5*u.mA)

    References
    ----------
    .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
           mechanisms determine the neuronal response to fluctuating
           inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
    .. [2] http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model

    .. seealso::

       :class:`brainpy.dyn.AdExIF` for the dynamical-system counterpart.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 1. * u.ohm,
        tau: ArrayLike = 10. * u.ms,
        tau_w: ArrayLike = 30. * u.ms,
        V_th: ArrayLike = -55. * u.mV,
        V_reset: ArrayLike = -68. * u.mV,
        V_rest: ArrayLike = -65. * u.mV,
        V_T: ArrayLike = -59.9 * u.mV,
        delta_T: ArrayLike = 3.48 * u.mV,
        a: ArrayLike = 1. * u.siemens,
        b: ArrayLike = 1. * u.mA,
        V_initializer: Callable = braintools.init.Constant(-65. * u.mV),
        w_initializer: Callable = braintools.init.Constant(0. * u.mA),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = braintools.init.param(R, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.tau_w = braintools.init.param(tau_w, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.V_rest = braintools.init.param(V_rest, self.varshape)
        self.V_T = braintools.init.param(V_T, self.varshape)
        self.delta_T = braintools.init.param(delta_T, self.varshape)
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)

        # initializers
        self.V_initializer = V_initializer
        self.w_initializer = w_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.w = brainstate.HiddenState(braintools.init.param(self.w_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.w.value = braintools.init.param(self.w_initializer, self.varshape, batch_size)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        last_v = self.V.value
        last_w = self.w.value
        last_spk = self.get_spike(last_v)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        V = last_v - (V_th - self.V_reset) * last_spk
        w = last_w + self.b * last_spk

        def dv(v):
            exp_term = self.delta_T * u.math.exp((v - self.V_T) / self.delta_T)
            I_total = self.sum_current_inputs(x, v)
            return (-(v - self.V_rest) + exp_term - self.R * w + self.R * I_total) / self.tau

        V = brainstate.nn.exp_euler_step(dv, V)
        V = self.sum_delta_inputs(V)

        def dw_func(w_val):
            return (self.a * (V - self.V_rest) - w_val) / self.tau_w

        w = brainstate.nn.exp_euler_step(dw_func, w)
        self.V.value = V
        self.w.value = w
        return self.get_spike(self.V.value)


class AdExIFRef(Neuron):
    r"""Adaptive exponential Integrate-and-Fire neuron model with refractory mechanism.

    This model extends :class:`AdExIF` by adding an absolute refractory period. While the
    exponential spike-initiation term and adaptation current keep the membrane potential
    dynamics biologically realistic, the refractory mechanism prevents the neuron from
    firing within ``tau_ref`` after a spike.

    The membrane dynamics are governed by two coupled differential equations:

    $$
    \tau \frac{dV}{dt} = -(V - V_{rest}) + \Delta_T
    \exp\left(\frac{V - V_T}{\Delta_T}\right) - R w + R \cdot I(t)
    $$

    $$
    \tau_w \frac{dw}{dt} = a (V - V_{rest}) - w
    $$

    After each spike the membrane potential is reset and the adaptation current
    increases by ``b``. During the refractory period, the membrane potential
    remains at the reset value.

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=1. * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=10. * u.ms
        Membrane time constant.
    tau_w : ArrayLike, default=30. * u.ms
        Adaptation current time constant.
    tau_ref : ArrayLike, default=1.7 * u.ms
        Absolute refractory period duration.
    V_th : ArrayLike, default=-55. * u.mV
        Spike threshold used for reset.
    V_reset : ArrayLike, default=-68. * u.mV
        Reset potential after spike.
    V_rest : ArrayLike, default=-65. * u.mV
        Resting membrane potential.
    V_T : ArrayLike, default=-59.9 * u.mV
        Threshold of the exponential term.
    delta_T : ArrayLike, default=3.48 * u.mV
        Spike slope factor controlling the sharpness of spike initiation.
    a : ArrayLike, default=1. * u.siemens
        Coupling strength from voltage to adaptation current.
    b : ArrayLike, default=1. * u.mA
        Increment of the adaptation current after a spike.
    V_initializer : Callable
        Initializer for the membrane potential state.
    w_initializer : Callable
        Initializer for the adaptation current.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function for the spike generation.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation.
    ref_var : bool, default=False
        Whether to expose a boolean refractory state variable.
    name : str, optional
        Name of the neuron layer.

    Attributes
    ----------
    V : HiddenState
        Membrane potential.
    w : HiddenState
        Adaptation current.
    last_spike_time : ShortTermState
        Last spike time recorder.
    refractory : HiddenState
        Neuron refractory state (if ref_var=True).

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create an AdExIFRef neuron layer with 10 neurons
    >>> adexif_ref = brainpy.state.AdExIFRef(10, tau=10*u.ms, tau_ref=2*u.ms)
    >>>
    >>> # Initialize the state
    >>> adexif_ref.init_state(batch_size=1)
    >>>
    >>> # Generate inputs
    >>> time_steps = 100
    >>> inputs = brainstate.random.randn(time_steps, 1, 10) * u.mA
    >>>
    >>> # Apply input currents and update the neuron state
    >>> with brainstate.environ.context(dt=0.1 * u.ms):
    >>>     for t in range(time_steps):
    >>>         with brainstate.environ.context(t=t*0.1*u.ms):
    >>>             spikes = adexif_ref.update(x=inputs[t])

    References
    ----------
    .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
           mechanisms determine the neuronal response to fluctuating
           inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
    .. [2] http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model

    .. seealso::

       :class:`brainpy.dyn.AdExIFRef` for the dynamical-system counterpart.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 1. * u.ohm,
        tau: ArrayLike = 10. * u.ms,
        tau_w: ArrayLike = 30. * u.ms,
        tau_ref: ArrayLike = 1.7 * u.ms,
        V_th: ArrayLike = -55. * u.mV,
        V_reset: ArrayLike = -68. * u.mV,
        V_rest: ArrayLike = -65. * u.mV,
        V_T: ArrayLike = -59.9 * u.mV,
        delta_T: ArrayLike = 3.48 * u.mV,
        a: ArrayLike = 1. * u.siemens,
        b: ArrayLike = 1. * u.mA,
        V_initializer: Callable = braintools.init.Constant(-65. * u.mV),
        w_initializer: Callable = braintools.init.Constant(0. * u.mA),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = braintools.init.param(R, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.tau_w = braintools.init.param(tau_w, self.varshape)
        self.tau_ref = braintools.init.param(tau_ref, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.V_rest = braintools.init.param(V_rest, self.varshape)
        self.V_T = braintools.init.param(V_T, self.varshape)
        self.delta_T = braintools.init.param(delta_T, self.varshape)
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)

        # initializers
        self.V_initializer = V_initializer
        self.w_initializer = w_initializer
        self.ref_var = ref_var

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.w = brainstate.HiddenState(braintools.init.param(self.w_initializer, self.varshape, batch_size))
        self.last_spike_time = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        )
        if self.ref_var:
            self.refractory = brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            )

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.w.value = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        if self.ref_var:
            self.refractory.value = braintools.init.param(
                braintools.init.Constant(False), self.varshape, batch_size
            )

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        t = brainstate.environ.get('t')
        last_v = self.V.value
        last_w = self.w.value
        last_spk = self.get_spike(last_v)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        v_reset = last_v - (V_th - self.V_reset) * last_spk
        w_reset = last_w + self.b * last_spk

        def dv(v):
            exp_term = self.delta_T * u.math.exp((v - self.V_T) / self.delta_T)
            I_total = self.sum_current_inputs(x, v)
            return (-(v - self.V_rest) + exp_term - self.R * w_reset + self.R * I_total) / self.tau

        V_candidate = brainstate.nn.exp_euler_step(dv, v_reset)
        V_candidate = self.sum_delta_inputs(V_candidate)

        def dw_func(w_val):
            return (self.a * (V_candidate - self.V_rest) - w_val) / self.tau_w

        w_candidate = brainstate.nn.exp_euler_step(dw_func, w_reset)

        refractory = (t - self.last_spike_time.value) < self.tau_ref
        self.V.value = u.math.where(refractory, v_reset, V_candidate)
        self.w.value = u.math.where(refractory, w_reset, w_candidate)

        spike_cond = self.V.value >= self.V_th
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t, self.last_spike_time.value)
        )
        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(
                u.math.logical_or(refractory, spike_cond)
            )
        return self.get_spike()


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
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create a LIFRef neuron layer with 10 neurons
    >>> lifref = brainpy.state.LIFRef(10,
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
    ...     brainpy.state.LIFRef(100, tau_ref=4*u.ms),
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
        last_spk = self.get_spike(last_v)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        last_v = last_v - (V_th - self.V_reset) * last_spk
        # membrane potential
        dv = lambda v: (-v + self.V_rest + self.R * self.sum_current_inputs(x, v)) / self.tau
        V = brainstate.nn.exp_euler_step(dv, last_v)
        V = self.sum_delta_inputs(V)
        self.V.value = u.math.where(t - self.last_spike_time.value < self.tau_ref, last_v, V)
        # spike time evaluation
        last_spk_time = u.math.where(
            self.V.value >= self.V_th, brainstate.environ.get('t'), self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spk_time)
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
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create an ALIF neuron layer with 10 neurons
    >>> alif = brainpy.state.ALIF(10,
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
    ...     brainpy.state.ALIF(100, tau_a=150*u.ms, beta=0.3*u.mV),
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


class QuaIF(Neuron):
    r"""Quadratic Integrate-and-Fire (QuaIF) neuron model.

    This model extends the basic integrate-and-fire neuron by adding a quadratic
    nonlinearity in the voltage dynamics. The quadratic term creates a soft spike
    initiation, making the model more biologically realistic than the linear IF model.

    The model is characterized by the following differential equation:

    $$
    \tau \frac{dV}{dt} = c(V - V_{rest})(V - V_c) + R \cdot I(t)
    $$

    Spike condition:
    If $V \geq V_{th}$: emit spike and reset $V = V_{reset}$

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=1. * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=10. * u.ms
        Membrane time constant.
    V_th : ArrayLike, default=-30. * u.mV
        Firing threshold voltage.
    V_reset : ArrayLike, default=-68. * u.mV
        Reset voltage after spike.
    V_rest : ArrayLike, default=-65. * u.mV
        Resting membrane potential.
    V_c : ArrayLike, default=-50. * u.mV
        Critical voltage for spike initiation. Must be larger than V_rest.
    c : ArrayLike, default=0.07 / u.mV
        Coefficient describing membrane potential update. Larger than 0.
    V_initializer : Callable
        Initializer for the membrane potential state.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function for the spike generation.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation.
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
    >>> # Create a QuaIF neuron layer with 10 neurons
    >>> quaif = brainpy.state.QuaIF(10, tau=10*u.ms, V_th=-30*u.mV, V_c=-50*u.mV)
    >>>
    >>> # Initialize the state
    >>> quaif.init_state(batch_size=1)
    >>>
    >>> # Apply an input current and update the neuron state
    >>> spikes = quaif.update(x=2.5*u.mA)
    >>>
    >>> # Create a network with QuaIF neurons
    >>> network = brainstate.nn.Sequential([
    ...     brainpy.state.QuaIF(100, tau=10.0*u.ms),
    ...     brainstate.nn.Linear(100, 10)
    ... ])

    Notes
    -----
    - The quadratic nonlinearity provides a more realistic spike initiation compared to LIF.
    - The critical voltage V_c determines the onset of spike generation.
    - When V approaches V_c, the quadratic term causes rapid acceleration toward threshold.
    - This model can exhibit Type I excitability (continuous f-I curve).

    References
    ----------
    .. [1] P. E. Latham, B.J. Richmond, P. Nelson and S. Nirenberg
           (2000) Intrinsic dynamics in neuronal networks. I. Theory.
           J. Neurophysiology 83, pp. 808–827.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 1. * u.ohm,
        tau: ArrayLike = 10. * u.ms,
        V_th: ArrayLike = -30. * u.mV,
        V_reset: ArrayLike = -68. * u.mV,
        V_rest: ArrayLike = -65. * u.mV,
        V_c: ArrayLike = -50. * u.mV,
        c: ArrayLike = 0.07 / u.mV,
        V_initializer: Callable = braintools.init.Constant(-65. * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = braintools.init.param(R, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.V_rest = braintools.init.param(V_rest, self.varshape)
        self.V_c = braintools.init.param(V_c, self.varshape)
        self.c = braintools.init.param(c, self.varshape)
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
        last_spk = self.get_spike(last_v)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        V = last_v - (V_th - self.V_reset) * last_spk

        def dv(v):
            return (self.c * (v - self.V_rest) * (v - self.V_c) + self.R * self.sum_current_inputs(x, v)) / self.tau

        V = brainstate.nn.exp_euler_step(dv, V)
        V = self.sum_delta_inputs(V)
        self.V.value = V
        return self.get_spike(V)


class AdQuaIF(Neuron):
    r"""Adaptive Quadratic Integrate-and-Fire (AdQuaIF) neuron model.

    This model extends the QuaIF model by adding an adaptation current that increases
    after each spike and decays exponentially between spikes. The adaptation mechanism
    produces spike-frequency adaptation and enables the neuron to exhibit various
    firing patterns.

    The model is characterized by the following differential equations:

    $$
    \tau \frac{dV}{dt} = c(V - V_{rest})(V - V_c) - w + R \cdot I(t)
    $$

    $$
    \tau_w \frac{dw}{dt} = a(V - V_{rest}) - w
    $$

    After a spike: $V \rightarrow V_{reset}$, $w \rightarrow w + b$

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=1. * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=10. * u.ms
        Membrane time constant.
    tau_w : ArrayLike, default=10. * u.ms
        Adaptation current time constant.
    V_th : ArrayLike, default=-30. * u.mV
        Firing threshold voltage.
    V_reset : ArrayLike, default=-68. * u.mV
        Reset voltage after spike.
    V_rest : ArrayLike, default=-65. * u.mV
        Resting membrane potential.
    V_c : ArrayLike, default=-50. * u.mV
        Critical voltage for spike initiation.
    c : ArrayLike, default=0.07 / u.mV
        Coefficient describing membrane potential update.
    a : ArrayLike, default=1. * u.siemens
        Coupling strength from voltage to adaptation current.
    b : ArrayLike, default=0.1 * u.mA
        Increment of adaptation current after a spike.
    V_initializer : Callable
        Initializer for the membrane potential state.
    w_initializer : Callable
        Initializer for the adaptation current.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation.
    name : str, optional
        Name of the neuron layer.

    Attributes
    ----------
    V : HiddenState
        Membrane potential.
    w : HiddenState
        Adaptation current.

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create an AdQuaIF neuron layer with 10 neurons
    >>> adquaif = brainpy.state.AdQuaIF(10, tau=10*u.ms, tau_w=100*u.ms,
    ...                                 a=1.0*u.siemens, b=0.1*u.mA)
    >>>
    >>> # Initialize the state
    >>> adquaif.init_state(batch_size=1)
    >>>
    >>> # Apply an input current and observe spike-frequency adaptation
    >>> spikes = adquaif.update(x=3.0*u.mA)
    >>>
    >>> # Create a network with adaptive neurons
    >>> network = brainstate.nn.Sequential([
    ...     brainpy.state.AdQuaIF(100, tau=10.0*u.ms, tau_w=100.0*u.ms),
    ...     brainstate.nn.Linear(100, 10)
    ... ])

    Notes
    -----
    - The adaptation current w provides negative feedback, reducing firing rate.
    - Parameter 'a' controls subthreshold adaptation (coupling from V to w).
    - Parameter 'b' controls spike-triggered adaptation (increment after spike).
    - With appropriate parameters, can exhibit regular spiking, bursting, etc.
    - The adaptation time constant tau_w determines adaptation speed.

    References
    ----------
    .. [1] Izhikevich, E. M. (2004). Which model to use for cortical spiking
           neurons?. IEEE transactions on neural networks, 15(5), 1063-1070.
    .. [2] Touboul, Jonathan. "Bifurcation analysis of a general class of
           nonlinear integrate-and-fire neurons." SIAM Journal on Applied
           Mathematics 68, no. 4 (2008): 1045-1079.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 1. * u.ohm,
        tau: ArrayLike = 10. * u.ms,
        tau_w: ArrayLike = 10. * u.ms,
        V_th: ArrayLike = -30. * u.mV,
        V_reset: ArrayLike = -68. * u.mV,
        V_rest: ArrayLike = -65. * u.mV,
        V_c: ArrayLike = -50. * u.mV,
        c: ArrayLike = 0.07 / u.mV,
        a: ArrayLike = 1. * u.siemens,
        b: ArrayLike = 0.1 * u.mA,
        V_initializer: Callable = braintools.init.Constant(-65. * u.mV),
        w_initializer: Callable = braintools.init.Constant(0. * u.mA),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = braintools.init.param(R, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.tau_w = braintools.init.param(tau_w, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.V_rest = braintools.init.param(V_rest, self.varshape)
        self.V_c = braintools.init.param(V_c, self.varshape)
        self.c = braintools.init.param(c, self.varshape)
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)
        self.V_initializer = V_initializer
        self.w_initializer = w_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.w = brainstate.HiddenState(braintools.init.param(self.w_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.w.value = braintools.init.param(self.w_initializer, self.varshape, batch_size)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        last_v = self.V.value
        last_w = self.w.value
        last_spk = self.get_spike(last_v)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        V = last_v - (V_th - self.V_reset) * last_spk
        w = last_w + self.b * last_spk

        def dv(v):
            return (self.c * (v - self.V_rest) * (v - self.V_c) - self.R * w + self.R * self.sum_current_inputs(x, v)) / self.tau

        def dw_func(w_val):
            return (self.a * (V - self.V_rest) - w_val) / self.tau_w

        V = brainstate.nn.exp_euler_step(dv, V)
        V = self.sum_delta_inputs(V)
        w = brainstate.nn.exp_euler_step(dw_func, w)

        self.V.value = V
        self.w.value = w
        return self.get_spike(V)


class AdQuaIFRef(Neuron):
    r"""Adaptive Quadratic Integrate-and-Fire neuron model with refractory mechanism.

    This model extends AdQuaIF by adding an absolute refractory period during which
    the neuron cannot fire regardless of input. The combination of adaptation and
    refractory period creates realistic firing patterns.

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=1. * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=10. * u.ms
        Membrane time constant.
    tau_w : ArrayLike, default=10. * u.ms
        Adaptation current time constant.
    tau_ref : ArrayLike, default=1.7 * u.ms
        Absolute refractory period duration.
    V_th : ArrayLike, default=-30. * u.mV
        Firing threshold voltage.
    V_reset : ArrayLike, default=-68. * u.mV
        Reset voltage after spike.
    V_rest : ArrayLike, default=-65. * u.mV
        Resting membrane potential.
    V_c : ArrayLike, default=-50. * u.mV
        Critical voltage for spike initiation.
    c : ArrayLike, default=0.07 / u.mV
        Coefficient describing membrane potential update.
    a : ArrayLike, default=1. * u.siemens
        Coupling strength from voltage to adaptation current.
    b : ArrayLike, default=0.1 * u.mA
        Increment of adaptation current after a spike.
    V_initializer : Callable
        Initializer for the membrane potential state.
    w_initializer : Callable
        Initializer for the adaptation current.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation.
    ref_var : bool, default=False
        Whether to expose a boolean refractory state variable.
    name : str, optional
        Name of the neuron layer.

    Attributes
    ----------
    V : HiddenState
        Membrane potential.
    w : HiddenState
        Adaptation current.
    last_spike_time : ShortTermState
        Last spike time recorder.
    refractory : HiddenState
        Neuron refractory state (if ref_var=True).

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create an AdQuaIFRef neuron layer with refractory period
    >>> adquaif_ref = brainpy.state.AdQuaIFRef(10, tau=10*u.ms, tau_w=100*u.ms,
    ...                                        tau_ref=2.0*u.ms, ref_var=True)
    >>>
    >>> # Initialize the state
    >>> adquaif_ref.init_state(batch_size=1)
    >>>
    >>> # Apply input and observe refractory behavior
    >>> with brainstate.environ.context(dt=0.1*u.ms, t=0.0*u.ms):
    ...     spikes = adquaif_ref.update(x=3.0*u.mA)
    >>>
    >>> # Create a network with refractory adaptive neurons
    >>> network = brainstate.nn.Sequential([
    ...     brainpy.state.AdQuaIFRef(100, tau=10.0*u.ms, tau_ref=2.0*u.ms),
    ...     brainstate.nn.Linear(100, 10)
    ... ])

    Notes
    -----
    - Combines spike-frequency adaptation with absolute refractory period.
    - During refractory period, neuron state is held at reset values.
    - Set ref_var=True to track refractory state as a boolean variable.
    - Refractory period prevents unrealistically high firing rates.
    - More biologically realistic than AdQuaIF without refractory period.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 1. * u.ohm,
        tau: ArrayLike = 10. * u.ms,
        tau_w: ArrayLike = 10. * u.ms,
        tau_ref: ArrayLike = 1.7 * u.ms,
        V_th: ArrayLike = -30. * u.mV,
        V_reset: ArrayLike = -68. * u.mV,
        V_rest: ArrayLike = -65. * u.mV,
        V_c: ArrayLike = -50. * u.mV,
        c: ArrayLike = 0.07 / u.mV,
        a: ArrayLike = 1. * u.siemens,
        b: ArrayLike = 0.1 * u.mA,
        V_initializer: Callable = braintools.init.Constant(-65. * u.mV),
        w_initializer: Callable = braintools.init.Constant(0. * u.mA),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = braintools.init.param(R, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.tau_w = braintools.init.param(tau_w, self.varshape)
        self.tau_ref = braintools.init.param(tau_ref, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.V_rest = braintools.init.param(V_rest, self.varshape)
        self.V_c = braintools.init.param(V_c, self.varshape)
        self.c = braintools.init.param(c, self.varshape)
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)
        self.V_initializer = V_initializer
        self.w_initializer = w_initializer
        self.ref_var = ref_var

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.w = brainstate.HiddenState(braintools.init.param(self.w_initializer, self.varshape, batch_size))
        self.last_spike_time = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        )
        if self.ref_var:
            self.refractory = brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            )

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.w.value = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        if self.ref_var:
            self.refractory.value = braintools.init.param(
                braintools.init.Constant(False), self.varshape, batch_size
            )

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        t = brainstate.environ.get('t')
        last_v = self.V.value
        last_w = self.w.value
        last_spk = self.get_spike(last_v)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        v_reset = last_v - (V_th - self.V_reset) * last_spk
        w_reset = last_w + self.b * last_spk

        def dv(v):
            return (self.c * (v - self.V_rest) * (v - self.V_c) - self.R * w_reset + self.R * self.sum_current_inputs(x, v)) / self.tau

        V_candidate = brainstate.nn.exp_euler_step(dv, v_reset)
        V_candidate = self.sum_delta_inputs(V_candidate)

        def dw_func(w_val):
            return (self.a * (V_candidate - self.V_rest) - w_val) / self.tau_w

        w_candidate = brainstate.nn.exp_euler_step(dw_func, w_reset)

        refractory = (t - self.last_spike_time.value) < self.tau_ref
        self.V.value = u.math.where(refractory, v_reset, V_candidate)
        self.w.value = u.math.where(refractory, w_reset, w_candidate)

        spike_cond = self.V.value >= self.V_th
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t, self.last_spike_time.value)
        )
        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(
                u.math.logical_or(refractory, spike_cond)
            )
        return self.get_spike()


class Gif(Neuron):
    r"""Generalized Integrate-and-Fire (Gif) neuron model.

    This model extends the basic integrate-and-fire neuron by adding internal
    currents and a dynamic threshold. The model can reproduce diverse firing
    patterns observed in biological neurons.

    The model is characterized by the following equations:

    $$
    \frac{dI_1}{dt} = -k_1 I_1
    $$

    $$
    \frac{dI_2}{dt} = -k_2 I_2
    $$

    $$
    \tau \frac{dV}{dt} = -(V - V_{rest}) + R(I_1 + I_2 + I(t))
    $$

    $$
    \frac{dV_{th}}{dt} = a(V - V_{rest}) - b(V_{th} - V_{th\infty})
    $$

    When $V \geq V_{th}$:
    - $I_1 \leftarrow R_1 I_1 + A_1$
    - $I_2 \leftarrow R_2 I_2 + A_2$
    - $V \leftarrow V_{reset}$
    - $V_{th} \leftarrow \max(V_{th_{reset}}, V_{th})$

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=20. * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=20. * u.ms
        Membrane time constant.
    V_rest : ArrayLike, default=-70. * u.mV
        Resting potential.
    V_reset : ArrayLike, default=-70. * u.mV
        Reset potential after spike.
    V_th_inf : ArrayLike, default=-50. * u.mV
        Target value of threshold potential updating.
    V_th_reset : ArrayLike, default=-60. * u.mV
        Free parameter, should be larger than V_reset.
    V_th_initializer : Callable
        Initializer for the threshold potential.
    a : ArrayLike, default=0. / u.ms
        Coefficient describes dependence of V_th on membrane potential.
    b : ArrayLike, default=0.01 / u.ms
        Coefficient describes V_th update.
    k1 : ArrayLike, default=0.2 / u.ms
        Constant of I1.
    k2 : ArrayLike, default=0.02 / u.ms
        Constant of I2.
    R1 : ArrayLike, default=0.
        Free parameter describing dependence of I1 reset value on I1 before spiking.
    R2 : ArrayLike, default=1.
        Free parameter describing dependence of I2 reset value on I2 before spiking.
    A1 : ArrayLike, default=0. * u.mA
        Free parameter.
    A2 : ArrayLike, default=0. * u.mA
        Free parameter.
    V_initializer : Callable
        Initializer for the membrane potential state.
    I1_initializer : Callable
        Initializer for internal current I1.
    I2_initializer : Callable
        Initializer for internal current I2.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation.
    name : str, optional
        Name of the neuron layer.

    Attributes
    ----------
    V : HiddenState
        Membrane potential.
    I1 : HiddenState
        Internal current 1.
    I2 : HiddenState
        Internal current 2.
    V_th : HiddenState
        Spiking threshold potential.

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create a Gif neuron layer with dynamic threshold
    >>> gif = brainpy.state.Gif(10, tau=20*u.ms, k1=0.2/u.ms, k2=0.02/u.ms,
    ...                         a=0.005/u.ms, b=0.01/u.ms)
    >>>
    >>> # Initialize the state
    >>> gif.init_state(batch_size=1)
    >>>
    >>> # Apply input and observe diverse firing patterns
    >>> spikes = gif.update(x=1.5*u.mA)
    >>>
    >>> # Create a network with Gif neurons
    >>> network = brainstate.nn.Sequential([
    ...     brainpy.state.Gif(100, tau=20.0*u.ms),
    ...     brainstate.nn.Linear(100, 10)
    ... ])

    Notes
    -----
    - The Gif model uses internal currents (I1, I2) for complex dynamics.
    - Dynamic threshold V_th adapts based on membrane potential and its own dynamics.
    - Can reproduce diverse firing patterns: regular spiking, bursting, adaptation.
    - Parameters a and b control threshold adaptation.
    - Parameters k1, k2, R1, R2, A1, A2 control internal current dynamics.
    - More flexible than simpler IF models for matching biological data.

    References
    ----------
    .. [1] Mihalaş, Ştefan, and Ernst Niebur. "A generalized linear
           integrate-and-fire neural model produces diverse spiking
           behaviors." Neural computation 21.3 (2009): 704-718.
    .. [2] Teeter, Corinne, Ramakrishnan Iyer, Vilas Menon, Nathan
           Gouwens, David Feng, Jim Berg, Aaron Szafer et al. "Generalized
           leaky integrate-and-fire models classify multiple neuron types."
           Nature communications 9, no. 1 (2018): 1-15.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 20. * u.ohm,
        tau: ArrayLike = 20. * u.ms,
        V_rest: ArrayLike = -70. * u.mV,
        V_reset: ArrayLike = -70. * u.mV,
        V_th_inf: ArrayLike = -50. * u.mV,
        V_th_reset: ArrayLike = -60. * u.mV,
        V_th_initializer: Callable = braintools.init.Constant(-50. * u.mV),
        a: ArrayLike = 0. / u.ms,
        b: ArrayLike = 0.01 / u.ms,
        k1: ArrayLike = 0.2 / u.ms,
        k2: ArrayLike = 0.02 / u.ms,
        R1: ArrayLike = 0.,
        R2: ArrayLike = 1.,
        A1: ArrayLike = 0. * u.mA,
        A2: ArrayLike = 0. * u.mA,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
        I1_initializer: Callable = braintools.init.Constant(0. * u.mA),
        I2_initializer: Callable = braintools.init.Constant(0. * u.mA),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = braintools.init.param(R, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.V_rest = braintools.init.param(V_rest, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.V_th_inf = braintools.init.param(V_th_inf, self.varshape)
        self.V_th_reset = braintools.init.param(V_th_reset, self.varshape)
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)
        self.k1 = braintools.init.param(k1, self.varshape)
        self.k2 = braintools.init.param(k2, self.varshape)
        self.R1 = braintools.init.param(R1, self.varshape)
        self.R2 = braintools.init.param(R2, self.varshape)
        self.A1 = braintools.init.param(A1, self.varshape)
        self.A2 = braintools.init.param(A2, self.varshape)
        self.V_initializer = V_initializer
        self.I1_initializer = I1_initializer
        self.I2_initializer = I2_initializer
        self.V_th_initializer = V_th_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.I1 = brainstate.HiddenState(braintools.init.param(self.I1_initializer, self.varshape, batch_size))
        self.I2 = brainstate.HiddenState(braintools.init.param(self.I2_initializer, self.varshape, batch_size))
        self.V_th = brainstate.HiddenState(braintools.init.param(self.V_th_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.I1.value = braintools.init.param(self.I1_initializer, self.varshape, batch_size)
        self.I2.value = braintools.init.param(self.I2_initializer, self.varshape, batch_size)
        self.V_th.value = braintools.init.param(self.V_th_initializer, self.varshape, batch_size)

    def get_spike(self, V: ArrayLike = None, V_th: ArrayLike = None):
        V = self.V.value if V is None else V
        V_th = self.V_th.value if V_th is None else V_th
        v_scaled = (V - V_th) / (V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        last_v = self.V.value
        last_i1 = self.I1.value
        last_i2 = self.I2.value
        last_v_th = self.V_th.value
        last_spk = self.get_spike(last_v, last_v_th)

        # Apply spike effects
        V_th_val = last_v_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        V = last_v - (V_th_val - self.V_reset) * last_spk
        I1 = last_i1 + last_spk * (self.R1 * last_i1 + self.A1 - last_i1)
        I2 = last_i2 + last_spk * (self.R2 * last_i2 + self.A2 - last_i2)
        V_th = last_v_th + last_spk * (u.math.maximum(self.V_th_reset, last_v_th) - last_v_th)

        # Update dynamics
        def dI1(i1):
            return -self.k1 * i1

        def dI2(i2):
            return -self.k2 * i2

        def dV_th_func(v_th):
            return self.a * (V - self.V_rest) - self.b * (v_th - self.V_th_inf)

        def dv(v):
            return (-(v - self.V_rest) + self.R * (I1 + I2 + self.sum_current_inputs(x, v))) / self.tau

        I1 = brainstate.nn.exp_euler_step(dI1, I1)
        I2 = brainstate.nn.exp_euler_step(dI2, I2)
        V_th = brainstate.nn.exp_euler_step(dV_th_func, V_th)
        V = brainstate.nn.exp_euler_step(dv, V)
        V = self.sum_delta_inputs(V)

        self.V.value = V
        self.I1.value = I1
        self.I2.value = I2
        self.V_th.value = V_th
        return self.get_spike(V, V_th)


class GifRef(Neuron):
    r"""Generalized Integrate-and-Fire neuron model with refractory mechanism.

    This model extends Gif by adding an absolute refractory period during which
    the neuron cannot fire. This creates more realistic firing patterns and
    prevents unrealistic high-frequency firing.

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=20. * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=20. * u.ms
        Membrane time constant.
    tau_ref : ArrayLike, default=1.7 * u.ms
        Absolute refractory period duration.
    V_rest : ArrayLike, default=-70. * u.mV
        Resting potential.
    V_reset : ArrayLike, default=-70. * u.mV
        Reset potential after spike.
    V_th_inf : ArrayLike, default=-50. * u.mV
        Target value of threshold potential updating.
    V_th_reset : ArrayLike, default=-60. * u.mV
        Free parameter, should be larger than V_reset.
    V_th_initializer : Callable
        Initializer for the threshold potential.
    a : ArrayLike, default=0. / u.ms
        Coefficient describes dependence of V_th on membrane potential.
    b : ArrayLike, default=0.01 / u.ms
        Coefficient describes V_th update.
    k1 : ArrayLike, default=0.2 / u.ms
        Constant of I1.
    k2 : ArrayLike, default=0.02 / u.ms
        Constant of I2.
    R1 : ArrayLike, default=0.
        Free parameter.
    R2 : ArrayLike, default=1.
        Free parameter.
    A1 : ArrayLike, default=0. * u.mA
        Free parameter.
    A2 : ArrayLike, default=0. * u.mA
        Free parameter.
    V_initializer : Callable
        Initializer for the membrane potential state.
    I1_initializer : Callable
        Initializer for internal current I1.
    I2_initializer : Callable
        Initializer for internal current I2.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation.
    ref_var : bool, default=False
        Whether to expose a boolean refractory state variable.
    name : str, optional
        Name of the neuron layer.

    Attributes
    ----------
    V : HiddenState
        Membrane potential.
    I1 : HiddenState
        Internal current 1.
    I2 : HiddenState
        Internal current 2.
    V_th : HiddenState
        Spiking threshold potential.
    last_spike_time : ShortTermState
        Last spike time recorder.
    refractory : HiddenState
        Neuron refractory state (if ref_var=True).

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create a GifRef neuron layer with refractory period
    >>> gif_ref = brainpy.state.GifRef(10, tau=20*u.ms, tau_ref=2.0*u.ms,
    ...                                k1=0.2/u.ms, k2=0.02/u.ms, ref_var=True)
    >>>
    >>> # Initialize the state
    >>> gif_ref.init_state(batch_size=1)
    >>>
    >>> # Apply input and observe refractory behavior
    >>> with brainstate.environ.context(dt=0.1*u.ms, t=0.0*u.ms):
    ...     spikes = gif_ref.update(x=1.5*u.mA)
    >>>
    >>> # Create a network with refractory Gif neurons
    >>> network = brainstate.nn.Sequential([
    ...     brainpy.state.GifRef(100, tau=20.0*u.ms, tau_ref=2.0*u.ms),
    ...     brainstate.nn.Linear(100, 10)
    ... ])

    Notes
    -----
    - Combines Gif model's rich dynamics with absolute refractory period.
    - During refractory period, all state variables are held at reset values.
    - Set ref_var=True to track refractory state as a boolean variable.
    - More biologically realistic than Gif without refractory mechanism.
    - Can still exhibit diverse firing patterns: regular, bursting, adaptation.
    - Refractory period prevents unrealistically high firing rates.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 20. * u.ohm,
        tau: ArrayLike = 20. * u.ms,
        tau_ref: ArrayLike = 1.7 * u.ms,
        V_rest: ArrayLike = -70. * u.mV,
        V_reset: ArrayLike = -70. * u.mV,
        V_th_inf: ArrayLike = -50. * u.mV,
        V_th_reset: ArrayLike = -60. * u.mV,
        V_th_initializer: Callable = braintools.init.Constant(-50. * u.mV),
        a: ArrayLike = 0. / u.ms,
        b: ArrayLike = 0.01 / u.ms,
        k1: ArrayLike = 0.2 / u.ms,
        k2: ArrayLike = 0.02 / u.ms,
        R1: ArrayLike = 0.,
        R2: ArrayLike = 1.,
        A1: ArrayLike = 0. * u.mA,
        A2: ArrayLike = 0. * u.mA,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
        I1_initializer: Callable = braintools.init.Constant(0. * u.mA),
        I2_initializer: Callable = braintools.init.Constant(0. * u.mA),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = braintools.init.param(R, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.tau_ref = braintools.init.param(tau_ref, self.varshape)
        self.V_rest = braintools.init.param(V_rest, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.V_th_inf = braintools.init.param(V_th_inf, self.varshape)
        self.V_th_reset = braintools.init.param(V_th_reset, self.varshape)
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)
        self.k1 = braintools.init.param(k1, self.varshape)
        self.k2 = braintools.init.param(k2, self.varshape)
        self.R1 = braintools.init.param(R1, self.varshape)
        self.R2 = braintools.init.param(R2, self.varshape)
        self.A1 = braintools.init.param(A1, self.varshape)
        self.A2 = braintools.init.param(A2, self.varshape)
        self.V_initializer = V_initializer
        self.I1_initializer = I1_initializer
        self.I2_initializer = I2_initializer
        self.V_th_initializer = V_th_initializer
        self.ref_var = ref_var

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.I1 = brainstate.HiddenState(braintools.init.param(self.I1_initializer, self.varshape, batch_size))
        self.I2 = brainstate.HiddenState(braintools.init.param(self.I2_initializer, self.varshape, batch_size))
        self.V_th = brainstate.HiddenState(braintools.init.param(self.V_th_initializer, self.varshape, batch_size))
        self.last_spike_time = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        )
        if self.ref_var:
            self.refractory = brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            )

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.I1.value = braintools.init.param(self.I1_initializer, self.varshape, batch_size)
        self.I2.value = braintools.init.param(self.I2_initializer, self.varshape, batch_size)
        self.V_th.value = braintools.init.param(self.V_th_initializer, self.varshape, batch_size)
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        if self.ref_var:
            self.refractory.value = braintools.init.param(
                braintools.init.Constant(False), self.varshape, batch_size
            )

    def get_spike(self, V: ArrayLike = None, V_th: ArrayLike = None):
        V = self.V.value if V is None else V
        V_th = self.V_th.value if V_th is None else V_th
        v_scaled = (V - V_th) / (V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        t = brainstate.environ.get('t')
        last_v = self.V.value
        last_i1 = self.I1.value
        last_i2 = self.I2.value
        last_v_th = self.V_th.value
        last_spk = self.get_spike(last_v, last_v_th)

        # Apply spike effects
        V_th_val = last_v_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        v_reset = last_v - (V_th_val - self.V_reset) * last_spk
        i1_reset = last_i1 + last_spk * (self.R1 * last_i1 + self.A1 - last_i1)
        i2_reset = last_i2 + last_spk * (self.R2 * last_i2 + self.A2 - last_i2)
        v_th_reset = last_v_th + last_spk * (u.math.maximum(self.V_th_reset, last_v_th) - last_v_th)

        # Update dynamics
        def dI1(i1):
            return -self.k1 * i1

        def dI2(i2):
            return -self.k2 * i2

        def dV_th_func(v_th):
            return self.a * (v_reset - self.V_rest) - self.b * (v_th - self.V_th_inf)

        def dv(v):
            return (-(v - self.V_rest) + self.R * (i1_reset + i2_reset + self.sum_current_inputs(x, v))) / self.tau

        I1_candidate = brainstate.nn.exp_euler_step(dI1, i1_reset)
        I2_candidate = brainstate.nn.exp_euler_step(dI2, i2_reset)
        V_th_candidate = brainstate.nn.exp_euler_step(dV_th_func, v_th_reset)
        V_candidate = brainstate.nn.exp_euler_step(dv, v_reset)
        V_candidate = self.sum_delta_inputs(V_candidate)

        refractory = (t - self.last_spike_time.value) < self.tau_ref
        self.V.value = u.math.where(refractory, v_reset, V_candidate)
        self.I1.value = u.math.where(refractory, i1_reset, I1_candidate)
        self.I2.value = u.math.where(refractory, i2_reset, I2_candidate)
        self.V_th.value = u.math.where(refractory, v_th_reset, V_th_candidate)

        spike_cond = self.V.value >= self.V_th.value
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t, self.last_spike_time.value)
        )
        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(
                u.math.logical_or(refractory, spike_cond)
            )
        return self.get_spike()
