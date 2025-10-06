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

# -*- coding: utf-8 -*-


from typing import Optional, Callable

import braintools
import brainunit as u

import brainstate
from brainstate.typing import Size, ArrayLike
from ._base import Synapse

__all__ = [
    'Alpha', 'AMPA', 'GABAa',
]


class Alpha(Synapse):
    r"""
    Alpha synapse model.

    This class implements the alpha function synapse model, which produces
    a smooth, biologically realistic synaptic conductance waveform.
    The model is characterized by the differential equation system:

    dh/dt = -h/tau
    dg/dt = -g/tau + h/tau

    This produces a response that rises and then falls with a characteristic
    time constant $\tau$, with peak amplitude occurring at time $t = \tau$.

    Parameters
    ----------
    in_size : Size
        Size of the input.
    name : str, optional
        Name of the synapse instance.
    tau : ArrayLike, default=8.0*u.ms
        Time constant of the alpha function in milliseconds.
    g_initializer : ArrayLike or Callable, default=init.Constant(0. * u.mS)
        Initial value or initializer for synaptic conductance.

    Attributes
    ----------
    g : HiddenState
        Synaptic conductance state variable.
    h : HiddenState
        Auxiliary state variable for implementing the alpha function.
    tau : Parameter
        Time constant of the alpha function.

    Notes
    -----
    The alpha function is defined as g(t) = (t/tau) * exp(1-t/tau) for t ≥ 0.
    This implementation uses an exponential Euler integration method.
    The output of this synapse is the conductance value.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        name: Optional[str] = None,
        tau: ArrayLike = 8.0 * u.ms,
        g_initializer: ArrayLike | Callable = braintools.init.Constant(0. * u.mS),
    ):
        super().__init__(name=name, in_size=in_size)

        # parameters
        self.tau = braintools.init.param(tau, self.varshape)
        self.g_initializer = g_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.g = brainstate.HiddenState(braintools.init.param(self.g_initializer, self.varshape, batch_size))
        self.h = brainstate.HiddenState(braintools.init.param(self.g_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.g.value = braintools.init.param(self.g_initializer, self.varshape, batch_size)
        self.h.value = braintools.init.param(self.g_initializer, self.varshape, batch_size)

    def update(self, x=None):
        h = brainstate.nn.exp_euler_step(lambda h: -h / self.tau, self.h.value)
        self.g.value = brainstate.nn.exp_euler_step(
            lambda g, h: -g / self.tau + h / self.tau, self.g.value, self.h.value)
        self.h.value = self.sum_delta_inputs(h)
        if x is not None:
            self.h.value += x
        return self.g.value


class AMPA(Synapse):
    r"""AMPA receptor synapse model.

    This class implements a kinetic model of AMPA (α-amino-3-hydroxy-5-methyl-4-isoxazolepropionic acid)
    receptor-mediated synaptic transmission. AMPA receptors are ionotropic glutamate receptors that mediate
    fast excitatory synaptic transmission in the central nervous system.

    The model uses a Markov process approach to describe the state transitions of AMPA receptors
    between closed and open states, governed by neurotransmitter binding:

    $$
    \frac{dg}{dt} = \alpha [T] (1-g) - \beta g
    $$

    $$
    I_{syn} = g_{max} \cdot g \cdot (V - E)
    $$

    where:
    - $g$ represents the fraction of receptors in the open state
    - $\alpha$ is the binding rate constant [ms^-1 mM^-1]
    - $\beta$ is the unbinding rate constant [ms^-1]
    - $[T]$ is the neurotransmitter concentration [mM]
    - $I_{syn}$ is the resulting synaptic current
    - $g_{max}$ is the maximum conductance
    - $V$ is the membrane potential
    - $E$ is the reversal potential

    The neurotransmitter concentration $[T]$ follows a square pulse of amplitude T and
    duration T_dur after each presynaptic spike.

    Parameters
    ----------
    in_size : Size
        Size of the input.
    name : str, optional
        Name of the synapse instance.
    alpha : ArrayLike, default=0.98/(u.ms*u.mM)
        Binding rate constant [ms^-1 mM^-1].
    beta : ArrayLike, default=0.18/u.ms
        Unbinding rate constant [ms^-1].
    T : ArrayLike, default=0.5*u.mM
        Peak neurotransmitter concentration when released [mM].
    T_dur : ArrayLike, default=0.5*u.ms
        Duration of neurotransmitter presence in the synaptic cleft [ms].
    g_initializer : ArrayLike or Callable, default=init.Constant(0. * u.mS)
        Initial value or initializer for the synaptic conductance.

    Attributes
    ----------
    g : HiddenState
        Fraction of receptors in the open state.
    spike_arrival_time : ShortTermState
        Time of the most recent presynaptic spike.

    Notes
    -----
    - The model captures the fast-rising and relatively fast-decaying excitatory currents
      characteristic of AMPA receptor-mediated transmission.
    - The time course of the synaptic conductance is determined by both the binding and
      unbinding rate constants and the duration of transmitter presence.
    - This implementation uses an exponential Euler integration method.

    References
    ----------
    .. [1] Destexhe, A., Mainen, Z. F., & Sejnowski, T. J. (1994). Synthesis of models for
           excitable membranes, synaptic transmission and neuromodulation using a common
           kinetic formalism. Journal of computational neuroscience, 1(3), 195-230.
    .. [2] Vijayan, S., & Kopell, N. J. (2012). Thalamic model of awake alpha oscillations
           and implications for stimulus processing. Proceedings of the National Academy
           of Sciences, 109(45), 18553-18558.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        name: Optional[str] = None,
        alpha: ArrayLike = 0.98 / (u.ms * u.mM),
        beta: ArrayLike = 0.18 / u.ms,
        T: ArrayLike = 0.5 * u.mM,
        T_dur: ArrayLike = 0.5 * u.ms,
        g_initializer: ArrayLike | Callable = braintools.init.Constant(0. * u.mS),
    ):
        super().__init__(name=name, in_size=in_size)

        # parameters
        self.alpha = braintools.init.param(alpha, self.varshape)
        self.beta = braintools.init.param(beta, self.varshape)
        self.T = braintools.init.param(T, self.varshape)
        self.T_duration = braintools.init.param(T_dur, self.varshape)
        self.g_initializer = g_initializer

    def init_state(self, batch_size=None):
        self.g = brainstate.HiddenState(braintools.init.param(self.g_initializer, self.varshape, batch_size))
        self.spike_arrival_time = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        )

    def reset_state(self, batch_or_mode=None, **kwargs):
        self.g.value = braintools.init.param(self.g_initializer, self.varshape, batch_or_mode)
        self.spike_arrival_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_or_mode
        )

    def update(self, pre_spike):
        t = brainstate.environ.get('t')
        self.spike_arrival_time.value = u.math.where(pre_spike, t, self.spike_arrival_time.value)
        TT = ((t - self.spike_arrival_time.value) < self.T_duration) * self.T
        dg = lambda g: self.alpha * TT * (1 * u.get_unit(g) - g) - self.beta * g
        self.g.value = brainstate.nn.exp_euler_step(dg, self.g.value)
        return self.g.value


class GABAa(AMPA):
    r"""GABAa receptor synapse model.

    This class implements a kinetic model of GABAa (gamma-aminobutyric acid type A)
    receptor-mediated synaptic transmission. GABAa receptors are ionotropic chloride channels
    that mediate fast inhibitory synaptic transmission in the central nervous system.

    The model uses the same Markov process approach as the AMPA model but with different
    kinetic parameters appropriate for GABAa receptors:

    $$
    \frac{dg}{dt} = \alpha [T] (1-g) - \beta g
    $$

    $$
    I_{syn} = - g_{max} \cdot g \cdot (V - E)
    $$

    where:

    - $g$ represents the fraction of receptors in the open state
    - $\alpha$ is the binding rate constant [ms^-1 mM^-1], typically slower than AMPA
    - $\beta$ is the unbinding rate constant [ms^-1]
    - $[T]$ is the neurotransmitter (GABA) concentration [mM]
    - $I_{syn}$ is the resulting synaptic current (note the negative sign indicating inhibition)
    - $g_{max}$ is the maximum conductance
    - $V$ is the membrane potential
    - $E$ is the reversal potential (typically around -80 mV for chloride)

    The neurotransmitter concentration $[T]$ follows a square pulse of amplitude T and
    duration T_dur after each presynaptic spike.

    Parameters
    ----------
    in_size : Size
        Size of the input.
    name : str, optional
        Name of the synapse instance.
    alpha : ArrayLike, default=0.53/(u.ms*u.mM)
        Binding rate constant [ms^-1 mM^-1]. Typically slower than AMPA receptors.
    beta : ArrayLike, default=0.18/u.ms
        Unbinding rate constant [ms^-1].
    T : ArrayLike, default=1.0*u.mM
        Peak neurotransmitter concentration when released [mM]. Higher than AMPA.
    T_dur : ArrayLike, default=1.0*u.ms
        Duration of neurotransmitter presence in the synaptic cleft [ms]. Longer than AMPA.
    g_initializer : ArrayLike or Callable, default=init.Constant(0. * u.mS)
        Initial value or initializer for the synaptic conductance.

    Attributes
    ----------
    Inherits all attributes from AMPA class.

    Notes
    -----
    - GABAa receptors typically produce slower-rising and longer-lasting currents compared to AMPA receptors.
    - The inhibitory nature of GABAa receptors is reflected in the convention of using a negative sign in the
      synaptic current equation.
    - The reversal potential for GABAa receptors is typically around -80 mV (due to chloride), making them
      inhibitory for neurons with resting potentials more positive than this value.
    - This model does not include desensitization, which can be significant for prolonged GABA exposure.

    References
    ----------
    .. [1] Destexhe, A., Mainen, Z. F., & Sejnowski, T. J. (1994). Synthesis of models for
           excitable membranes, synaptic transmission and neuromodulation using a common
           kinetic formalism. Journal of computational neuroscience, 1(3), 195-230.
    .. [2] Destexhe, A., & Paré, D. (1999). Impact of network activity on the integrative
           properties of neocortical pyramidal neurons in vivo. Journal of neurophysiology,
           81(4), 1531-1547.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        name: Optional[str] = None,
        alpha: ArrayLike = 0.53 / (u.ms * u.mM),
        beta: ArrayLike = 0.18 / u.ms,
        T: ArrayLike = 1.0 * u.mM,
        T_dur: ArrayLike = 1.0 * u.ms,
        g_initializer: ArrayLike | Callable = braintools.init.Constant(0. * u.mS),
    ):
        super().__init__(
            alpha=alpha,
            beta=beta,
            T=T,
            T_dur=T_dur,
            name=name,
            in_size=in_size,
            g_initializer=g_initializer
        )
