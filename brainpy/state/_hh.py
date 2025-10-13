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
    'HH', 'MorrisLecar', 'WangBuzsakiHH',
]


class HH(Neuron):
    r"""Hodgkin–Huxley neuron model.

    **Model Descriptions**

    The Hodgkin-Huxley (HH; Hodgkin & Huxley, 1952) model for the generation of
    the nerve action potential is one of the most successful mathematical models of
    a complex biological process that has ever been formulated. The basic concepts
    expressed in the model have proved a valid approach to the study of bio-electrical
    activity from the most primitive single-celled organisms such as *Paramecium*,
    right through to the neurons within our own brains.

    Mathematically, the model is given by,

    $$
    C \frac {dV} {dt} = -(\bar{g}_{Na} m^3 h (V-E_{Na})
    + \bar{g}_K n^4 (V-E_K) + g_{leak} (V - E_{leak})) + I(t)
    $$

    $$
    \frac {dx} {dt} = \alpha_x (1-x) - \beta_x, \quad x\in {\rm{\{m, h, n\}}}
    $$

    where

    $$
    \alpha_m(V) = \frac {0.1(V+40)}{1-\exp(\frac{-(V + 40)} {10})}
    $$

    $$
    \beta_m(V) = 4.0 \exp(\frac{-(V + 65)} {18})
    $$

    $$
    \alpha_h(V) = 0.07 \exp(\frac{-(V+65)}{20})
    $$

    $$
    \beta_h(V) = \frac 1 {1 + \exp(\frac{-(V + 35)} {10})}
    $$

    $$
    \alpha_n(V) = \frac {0.01(V+55)}{1-\exp(-(V+55)/10)}
    $$

    $$
    \beta_n(V) = 0.125 \exp(\frac{-(V + 65)} {80})
    $$

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    ENa : ArrayLike, default=50. * u.mV
        Reversal potential of sodium.
    gNa : ArrayLike, default=120. * u.msiemens
        Maximum conductance of sodium channel.
    EK : ArrayLike, default=-77. * u.mV
        Reversal potential of potassium.
    gK : ArrayLike, default=36. * u.msiemens
        Maximum conductance of potassium channel.
    EL : ArrayLike, default=-54.387 * u.mV
        Reversal potential of leak channel.
    gL : ArrayLike, default=0.03 * u.msiemens
        Conductance of leak channel.
    V_th : ArrayLike, default=20. * u.mV
        Threshold of the membrane spike.
    C : ArrayLike, default=1.0 * u.ufarad
        Membrane capacitance.
    V_initializer : Callable
        Initializer for membrane potential.
    m_initializer : Callable, optional
        Initializer for m channel. If None, uses steady state.
    h_initializer : Callable, optional
        Initializer for h channel. If None, uses steady state.
    n_initializer : Callable, optional
        Initializer for n channel. If None, uses steady state.
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
    m : HiddenState
        Sodium activation variable.
    h : HiddenState
        Sodium inactivation variable.
    n : HiddenState
        Potassium activation variable.

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create an HH neuron layer with 10 neurons
    >>> hh = brainpy.state.HH(10)
    >>>
    >>> # Initialize the state
    >>> hh.init_state(batch_size=1)
    >>>
    >>> # Apply an input current and update the neuron state
    >>> spikes = hh.update(x=10.*u.uA)

    References
    ----------
    .. [1] Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description
           of membrane current and its application to conduction and excitation
           in nerve." The Journal of physiology 117.4 (1952): 500.
    .. [2] https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model
    """

    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        ENa: ArrayLike = 50. * u.mV,
        gNa: ArrayLike = 120. * u.msiemens,
        EK: ArrayLike = -77. * u.mV,
        gK: ArrayLike = 36. * u.msiemens,
        EL: ArrayLike = -54.387 * u.mV,
        gL: ArrayLike = 0.03 * u.msiemens,
        V_th: ArrayLike = 20. * u.mV,
        C: ArrayLike = 1.0 * u.ufarad,
        V_initializer: Callable = braintools.init.Uniform(-70. * u.mV, -60. * u.mV),
        m_initializer: Callable = None,
        h_initializer: Callable = None,
        n_initializer: Callable = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.ENa = braintools.init.param(ENa, self.varshape)
        self.EK = braintools.init.param(EK, self.varshape)
        self.EL = braintools.init.param(EL, self.varshape)
        self.gNa = braintools.init.param(gNa, self.varshape)
        self.gK = braintools.init.param(gK, self.varshape)
        self.gL = braintools.init.param(gL, self.varshape)
        self.C = braintools.init.param(C, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)

        # initializers
        self.V_initializer = V_initializer
        self.m_initializer = m_initializer
        self.h_initializer = h_initializer
        self.n_initializer = n_initializer

    def m_alpha(self, V):
        return 1. / u.math.exprel(-(V + 40. * u.mV) / (10. * u.mV)) / u.ms

    def m_beta(self, V):
        return 4.0 / u.ms * u.math.exp(-(V + 65. * u.mV) / (18. * u.mV))

    def m_inf(self, V):
        return self.m_alpha(V) / (self.m_alpha(V) + self.m_beta(V))

    def h_alpha(self, V):
        return 0.07 / u.ms * u.math.exp(-(V + 65. * u.mV) / (20. * u.mV))

    def h_beta(self, V):
        return 1. / u.ms / (1. + u.math.exp(-(V + 35. * u.mV) / (10. * u.mV)))

    def h_inf(self, V):
        return self.h_alpha(V) / (self.h_alpha(V) + self.h_beta(V))

    def n_alpha(self, V):
        return 0.1 / u.ms / u.math.exprel(-(V + 55. * u.mV) / (10. * u.mV))

    def n_beta(self, V):
        return 0.125 / u.ms * u.math.exp(-(V + 65. * u.mV) / (80. * u.mV))

    def n_inf(self, V):
        return self.n_alpha(V) / (self.n_alpha(V) + self.n_beta(V))

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        if self.m_initializer is None:
            self.m = brainstate.HiddenState(self.m_inf(self.V.value))
        else:
            self.m = brainstate.HiddenState(braintools.init.param(self.m_initializer, self.varshape, batch_size))
        if self.h_initializer is None:
            self.h = brainstate.HiddenState(self.h_inf(self.V.value))
        else:
            self.h = brainstate.HiddenState(braintools.init.param(self.h_initializer, self.varshape, batch_size))
        if self.n_initializer is None:
            self.n = brainstate.HiddenState(self.n_inf(self.V.value))
        else:
            self.n = brainstate.HiddenState(braintools.init.param(self.n_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        if self.m_initializer is None:
            self.m.value = self.m_inf(self.V.value)
        else:
            self.m.value = braintools.init.param(self.m_initializer, self.varshape, batch_size)
        if self.h_initializer is None:
            self.h.value = self.h_inf(self.V.value)
        else:
            self.h.value = braintools.init.param(self.h_initializer, self.varshape, batch_size)
        if self.n_initializer is None:
            self.n.value = self.n_inf(self.V.value)
        else:
            self.n.value = braintools.init.param(self.n_initializer, self.varshape, batch_size)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / self.V_th
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.uA):
        last_V = self.V.value
        last_m = self.m.value
        last_h = self.h.value
        last_n = self.n.value

        # Ionic currents
        I_Na = (self.gNa * last_m ** 3 * last_h) * (last_V - self.ENa)
        I_K = (self.gK * last_n ** 4) * (last_V - self.EK)
        I_leak = self.gL * (last_V - self.EL)

        # Voltage dynamics
        I_total = self.sum_current_inputs(x, last_V)
        dV = lambda V: (-I_Na - I_K - I_leak + I_total) / self.C

        # Gating variable dynamics
        dm = lambda m: self.m_alpha(last_V) * (1. - m) - self.m_beta(last_V) * m
        dh = lambda h: self.h_alpha(last_V) * (1. - h) - self.h_beta(last_V) * h
        dn = lambda n: self.n_alpha(last_V) * (1. - n) - self.n_beta(last_V) * n

        V = brainstate.nn.exp_euler_step(dV, last_V)
        V = self.sum_delta_inputs(V)
        m = brainstate.nn.exp_euler_step(dm, last_m)
        h = brainstate.nn.exp_euler_step(dh, last_h)
        n = brainstate.nn.exp_euler_step(dn, last_n)

        self.V.value = V
        self.m.value = m
        self.h.value = h
        self.n.value = n
        return self.get_spike(V)


class MorrisLecar(Neuron):
    r"""The Morris-Lecar neuron model.

    **Model Descriptions**

    The Morris-Lecar model (Also known as :math:`I_{Ca}+I_K`-model)
    is a two-dimensional "reduced" excitation model applicable to
    systems having two non-inactivating voltage-sensitive conductances.
    This model was named after Cathy Morris and Harold Lecar, who
    derived it in 1981. Because it is two-dimensional, the Morris-Lecar
    model is one of the favorite conductance-based models in computational neuroscience.

    The original form of the model employed an instantaneously
    responding voltage-sensitive Ca2+ conductance for excitation and a delayed
    voltage-dependent K+ conductance for recovery. The equations of the model are:

    $$
    \begin{aligned}
    C\frac{dV}{dt} =& - g_{Ca} M_{\infty} (V - V_{Ca})- g_{K} W(V - V_{K}) -
                      g_{Leak} (V - V_{Leak}) + I_{ext} \\
    \frac{dW}{dt} =& \frac{W_{\infty}(V) - W}{ \tau_W(V)}
    \end{aligned}
    $$

    Here, :math:`V` is the membrane potential, :math:`W` is the "recovery variable",
    which is almost invariably the normalized :math:`K^+`-ion conductance, and
    :math:`I_{ext}` is the applied current stimulus.

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    V_Ca : ArrayLike, default=130. * u.mV
        Equilibrium potential of Ca+.
    g_Ca : ArrayLike, default=4.4 * u.msiemens
        Maximum conductance of Ca+.
    V_K : ArrayLike, default=-84. * u.mV
        Equilibrium potential of K+.
    g_K : ArrayLike, default=8. * u.msiemens
        Maximum conductance of K+.
    V_leak : ArrayLike, default=-60. * u.mV
        Equilibrium potential of leak current.
    g_leak : ArrayLike, default=2. * u.msiemens
        Conductance of leak current.
    C : ArrayLike, default=20. * u.ufarad
        Membrane capacitance.
    V1 : ArrayLike, default=-1.2 * u.mV
        Potential at which M_inf = 0.5.
    V2 : ArrayLike, default=18. * u.mV
        Reciprocal of slope of voltage dependence of M_inf.
    V3 : ArrayLike, default=2. * u.mV
        Potential at which W_inf = 0.5.
    V4 : ArrayLike, default=30. * u.mV
        Reciprocal of slope of voltage dependence of W_inf.
    phi : ArrayLike, default=0.04 / u.ms
        Temperature factor.
    V_th : ArrayLike, default=10. * u.mV
        Spike threshold.
    V_initializer : Callable
        Initializer for membrane potential.
    W_initializer : Callable
        Initializer for recovery variable.
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
    W : HiddenState
        Recovery variable.

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create a Morris-Lecar neuron layer with 10 neurons
    >>> ml = brainpy.state.MorrisLecar(10)
    >>>
    >>> # Initialize the state
    >>> ml.init_state(batch_size=1)
    >>>
    >>> # Apply an input current and update the neuron state
    >>> spikes = ml.update(x=100.*u.uA)

    References
    ----------
    .. [1] Lecar, Harold. "Morris-lecar model." Scholarpedia 2.10 (2007): 1333.
    .. [2] http://www.scholarpedia.org/article/Morris-Lecar_model
    .. [3] https://en.wikipedia.org/wiki/Morris%E2%80%93Lecar_model
    """

    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        V_Ca: ArrayLike = 130. * u.mV,
        g_Ca: ArrayLike = 4.4 * u.msiemens,
        V_K: ArrayLike = -84. * u.mV,
        g_K: ArrayLike = 8. * u.msiemens,
        V_leak: ArrayLike = -60. * u.mV,
        g_leak: ArrayLike = 2. * u.msiemens,
        C: ArrayLike = 20. * u.ufarad,
        V1: ArrayLike = -1.2 * u.mV,
        V2: ArrayLike = 18. * u.mV,
        V3: ArrayLike = 2. * u.mV,
        V4: ArrayLike = 30. * u.mV,
        phi: ArrayLike = 0.04 / u.ms,
        V_th: ArrayLike = 10. * u.mV,
        V_initializer: Callable = braintools.init.Uniform(-70. * u.mV, -60. * u.mV),
        W_initializer: Callable = braintools.init.Constant(0.02),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.V_Ca = braintools.init.param(V_Ca, self.varshape)
        self.g_Ca = braintools.init.param(g_Ca, self.varshape)
        self.V_K = braintools.init.param(V_K, self.varshape)
        self.g_K = braintools.init.param(g_K, self.varshape)
        self.V_leak = braintools.init.param(V_leak, self.varshape)
        self.g_leak = braintools.init.param(g_leak, self.varshape)
        self.C = braintools.init.param(C, self.varshape)
        self.V1 = braintools.init.param(V1, self.varshape)
        self.V2 = braintools.init.param(V2, self.varshape)
        self.V3 = braintools.init.param(V3, self.varshape)
        self.V4 = braintools.init.param(V4, self.varshape)
        self.phi = braintools.init.param(phi, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)

        # initializers
        self.V_initializer = V_initializer
        self.W_initializer = W_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.W = brainstate.HiddenState(braintools.init.param(self.W_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.W.value = braintools.init.param(self.W_initializer, self.varshape, batch_size)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / self.V_th
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.uA):
        last_V = self.V.value
        last_W = self.W.value

        # Steady states
        M_inf = 0.5 * (1. + u.math.tanh((last_V - self.V1) / self.V2))
        W_inf = 0.5 * (1. + u.math.tanh((last_V - self.V3) / self.V4))
        tau_W = 1. / (self.phi * u.math.cosh((last_V - self.V3) / (2. * self.V4)))

        # Ionic currents
        I_Ca = self.g_Ca * M_inf * (last_V - self.V_Ca)
        I_K = self.g_K * last_W * (last_V - self.V_K)
        I_leak = self.g_leak * (last_V - self.V_leak)

        # Dynamics
        I_total = self.sum_current_inputs(x, last_V)
        dV = lambda V: (-I_Ca - I_K - I_leak + I_total) / self.C
        dW = lambda W: (W_inf - W) / tau_W

        V = brainstate.nn.exp_euler_step(dV, last_V)
        V = self.sum_delta_inputs(V)
        W = brainstate.nn.exp_euler_step(dW, last_W)

        self.V.value = V
        self.W.value = W
        return self.get_spike(V)


class WangBuzsakiHH(Neuron):
    r"""Wang-Buzsaki model, an implementation of a modified Hodgkin-Huxley model.

    Each model is described by a single compartment and obeys the current balance equation:

    $$
    C_{m} \frac{d V}{d t}=-I_{\mathrm{Na}}-I_{\mathrm{K}}-I_{\mathrm{L}}+I_{\mathrm{app}}
    $$

    where :math:`C_{m}=1 \mu \mathrm{F} / \mathrm{cm}^{2}` and :math:`I_{\mathrm{app}}` is the
    injected current (in :math:`\mu \mathrm{A} / \mathrm{cm}^{2}` ). The leak current
    :math:`I_{\mathrm{L}}=g_{\mathrm{L}}\left(V-E_{\mathrm{L}}\right)` has a conductance
    :math:`g_{\mathrm{L}}=0.1 \mathrm{mS} / \mathrm{cm}^{2}`.

    The spike-generating :math:`\mathrm{Na}^{+}` and :math:`\mathrm{K}^{+}` voltage-dependent ion
    currents are of the Hodgkin-Huxley type. The transient sodium current
    :math:`I_{\mathrm{Na}}=g_{\mathrm{Na}} m_{\infty}^{3} h\left(V-E_{\mathrm{Na}}\right)`,
    where the activation variable :math:`m` is assumed fast and substituted by its steady-state
    function :math:`m_{\infty}=\alpha_{m} /\left(\alpha_{m}+\beta_{m}\right)`;
    :math:`\alpha_{m}(V)=-0.1(V+35) /(\exp (-0.1(V+35))-1)`, :math:`\beta_{m}(V)=4 \exp (-(V+60) / 18)`.

    The inactivation variable :math:`h` obeys:

    $$
    \frac{d h}{d t}=\phi\left(\alpha_{h}(1-h)-\beta_{h} h\right)
    $$

    where :math:`\alpha_{h}(V)=0.07 \exp (-(V+58) / 20)` and
    :math:`\beta_{h}(V)=1 /(\exp (-0.1(V+28)) +1)`.

    The delayed rectifier :math:`I_{\mathrm{K}}=g_{\mathrm{K}} n^{4}\left(V-E_{\mathrm{K}}\right)`,
    where the activation variable :math:`n` obeys:

    $$
    \frac{d n}{d t}=\phi\left(\alpha_{n}(1-n)-\beta_{n} n\right)
    $$

    with :math:`\alpha_{n}(V)=-0.01(V+34) /(\exp (-0.1(V+34))-1)` and
    :math:`\beta_{n}(V)=0.125\exp (-(V+44) / 80)`.

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    ENa : ArrayLike, default=55. * u.mV
        Reversal potential of sodium.
    gNa : ArrayLike, default=35. * u.msiemens
        Maximum conductance of sodium channel.
    EK : ArrayLike, default=-90. * u.mV
        Reversal potential of potassium.
    gK : ArrayLike, default=9. * u.msiemens
        Maximum conductance of potassium channel.
    EL : ArrayLike, default=-65. * u.mV
        Reversal potential of leak channel.
    gL : ArrayLike, default=0.1 * u.msiemens
        Conductance of leak channel.
    V_th : ArrayLike, default=20. * u.mV
        Threshold of the membrane spike.
    phi : ArrayLike, default=5.0
        Temperature regulator constant.
    C : ArrayLike, default=1.0 * u.ufarad
        Membrane capacitance.
    V_initializer : Callable
        Initializer for membrane potential.
    h_initializer : Callable
        Initializer for h channel.
    n_initializer : Callable
        Initializer for n channel.
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
    h : HiddenState
        Sodium inactivation variable.
    n : HiddenState
        Potassium activation variable.

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create a WangBuzsakiHH neuron layer with 10 neurons
    >>> wb = brainpy.state.WangBuzsakiHH(10)
    >>>
    >>> # Initialize the state
    >>> wb.init_state(batch_size=1)
    >>>
    >>> # Apply an input current and update the neuron state
    >>> spikes = wb.update(x=1.*u.uA)

    References
    ----------
    .. [1] Wang, X.J. and Buzsaki, G., (1996) Gamma oscillation by synaptic
           inhibition in a hippocampal interneuronal network model. Journal of
           neuroscience, 16(20), pp.6402-6413.
    """

    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        ENa: ArrayLike = 55. * u.mV,
        gNa: ArrayLike = 35. * u.msiemens,
        EK: ArrayLike = -90. * u.mV,
        gK: ArrayLike = 9. * u.msiemens,
        EL: ArrayLike = -65. * u.mV,
        gL: ArrayLike = 0.1 * u.msiemens,
        V_th: ArrayLike = 20. * u.mV,
        phi: ArrayLike = 5.0,
        C: ArrayLike = 1.0 * u.ufarad,
        V_initializer: Callable = braintools.init.Constant(-65. * u.mV),
        h_initializer: Callable = braintools.init.Constant(0.6),
        n_initializer: Callable = braintools.init.Constant(0.32),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.ENa = braintools.init.param(ENa, self.varshape)
        self.EK = braintools.init.param(EK, self.varshape)
        self.EL = braintools.init.param(EL, self.varshape)
        self.gNa = braintools.init.param(gNa, self.varshape)
        self.gK = braintools.init.param(gK, self.varshape)
        self.gL = braintools.init.param(gL, self.varshape)
        self.phi = braintools.init.param(phi, self.varshape)
        self.C = braintools.init.param(C, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)

        # initializers
        self.V_initializer = V_initializer
        self.h_initializer = h_initializer
        self.n_initializer = n_initializer

    def m_inf(self, V):
        alpha = 1. / u.math.exprel(-0.1 * (V + 35. * u.mV) / u.mV) / u.ms
        beta = 4. / u.ms * u.math.exp(-(V + 60. * u.mV) / (18. * u.mV))
        return alpha / (alpha + beta)

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.h = brainstate.HiddenState(braintools.init.param(self.h_initializer, self.varshape, batch_size))
        self.n = brainstate.HiddenState(braintools.init.param(self.n_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.h.value = braintools.init.param(self.h_initializer, self.varshape, batch_size)
        self.n.value = braintools.init.param(self.n_initializer, self.varshape, batch_size)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / self.V_th
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.uA):
        last_V = self.V.value
        last_h = self.h.value
        last_n = self.n.value

        # Ionic currents
        m_inf_val = self.m_inf(last_V)
        I_Na = self.gNa * m_inf_val ** 3 * last_h * (last_V - self.ENa)
        I_K = self.gK * last_n ** 4 * (last_V - self.EK)
        I_L = self.gL * (last_V - self.EL)

        # Voltage dynamics
        I_total = self.sum_current_inputs(x, last_V)
        dV = lambda V: (-I_Na - I_K - I_L + I_total) / self.C

        # Gating variable dynamics
        h_alpha = 0.07 / u.ms * u.math.exp(-(last_V + 58. * u.mV) / (20. * u.mV))
        h_beta = 1. / u.ms / (u.math.exp(-0.1 * (last_V + 28. * u.mV) / u.mV) + 1.)
        dh = lambda h: self.phi * (h_alpha * (1. - h) - h_beta * h)

        n_alpha = 1. / u.ms / u.math.exprel(-0.1 * (last_V + 34. * u.mV) / u.mV)
        n_beta = 0.125 / u.ms * u.math.exp(-(last_V + 44. * u.mV) / (80. * u.mV))
        dn = lambda n: self.phi * (n_alpha * (1. - n) - n_beta * n)

        V = brainstate.nn.exp_euler_step(dV, last_V)
        V = self.sum_delta_inputs(V)
        h = brainstate.nn.exp_euler_step(dh, last_h)
        n = brainstate.nn.exp_euler_step(dn, last_n)

        self.V.value = V
        self.h.value = h
        self.n.value = n
        return self.get_spike(V)
