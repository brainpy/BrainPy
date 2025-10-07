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


import numbers
from typing import Callable

import braintools
import brainunit as u
import jax

import brainstate
from brainstate.typing import Size, ArrayLike
from ._base import Neuron

__all__ = [
    'LeakyRateReadout',
    'LeakySpikeReadout',
]


class LeakyRateReadout(brainstate.nn.Module):
    r"""
    Leaky dynamics for the read-out module.

    This module implements a leaky integrator with the following dynamics:

    .. math::
        r_{t} = \alpha r_{t-1} + x_{t} W

    where:
      - :math:`r_{t}` is the output at time t
      - :math:`\alpha = e^{-\Delta t / \tau}` is the decay factor
      - :math:`x_{t}` is the input at time t
      - :math:`W` is the weight matrix

    The leaky integrator acts as a low-pass filter, allowing the network
    to maintain memory of past inputs with an exponential decay determined
    by the time constant tau.

    Parameters
    ----------
    in_size : int or sequence of int
        Size of the input dimension(s)
    out_size : int or sequence of int
        Size of the output dimension(s)
    tau : ArrayLike, optional
        Time constant of the leaky dynamics, by default 5ms
    w_init : Callable, optional
        Weight initialization function, by default KaimingNormal()
    name : str, optional
        Name of the module, by default None

    Attributes
    ----------
    decay : float
        Decay factor computed as exp(-dt/tau)
    weight : ParamState
        Weight matrix connecting input to output
    r : HiddenState
        Hidden state representing the output values
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        tau: ArrayLike = 5. * u.ms,
        w_init: Callable = braintools.init.KaimingNormal(),
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self.in_size = (in_size,) if isinstance(in_size, numbers.Integral) else tuple(in_size)
        self.out_size = (out_size,) if isinstance(out_size, numbers.Integral) else tuple(out_size)
        self.tau = braintools.init.param(tau, self.in_size)
        self.decay = u.math.exp(-brainstate.environ.get_dt() / self.tau)

        # weights
        self.weight = brainstate.ParamState(brainstate.init.param(w_init, (self.in_size[0], self.out_size[0])))

    def init_state(self, batch_size=None, **kwargs):
        self.r = brainstate.HiddenState(
            brainstate.init.param(brainstate.init.Constant(0.), self.out_size, batch_size)
        )

    def reset_state(self, batch_size=None, **kwargs):
        self.r.value = brainstate.init.param(
            brainstate.init.Constant(0.), self.out_size, batch_size
        )

    def update(self, x):
        self.r.value = self.decay * self.r.value + x @ self.weight.value
        return self.r.value


class LeakySpikeReadout(Neuron):
    r"""
    Integrate-and-fire neuron model with leaky dynamics for readout functionality.

    This class implements a spiking neuron with the following dynamics:

    .. math::
        \frac{dV}{dt} = \frac{-V + I_{in}}{\tau}

    where:
      - :math:`V` is the membrane potential
      - :math:`\tau` is the membrane time constant
      - :math:`I_{in}` is the input current

    Spike generation occurs when :math:`V > V_{th}` according to:

    .. math::
        S_t = \text{surrogate}\left(\frac{V - V_{th}}{V_{th}}\right)

    After spiking, the membrane potential is reset according to the reset mode:
      - Soft reset: :math:`V \leftarrow V - V_{th} \cdot S_t`
      - Hard reset: :math:`V \leftarrow V - V_t \cdot S_t` (where :math:`V_t` is detached)

    Parameters
    ----------
    in_size : Size
        Size of the input dimension
    tau : ArrayLike, optional
        Membrane time constant, by default 5ms
    V_th : ArrayLike, optional
        Spike threshold, by default 1mV
    w_init : Callable, optional
        Weight initialization function, by default KaimingNormal(unit=mV)
    V_initializer : ArrayLike, optional
        Initial membrane potential, by default Constant(0. * u.mV)
    spk_fun : Callable, optional
        Surrogate gradient function for spike generation, by default ReluGrad()
    spk_reset : str, optional
        Reset mechanism after spike ('soft' or 'hard'), by default 'soft'
    name : str, optional
        Name of the module, by default None

    Attributes
    ----------
    V : HiddenState
        Membrane potential state variable
    weight : ParamState
        Synaptic weight matrix
    """

    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        tau: ArrayLike = 5. * u.ms,
        V_th: ArrayLike = 1. * u.mV,
        w_init: Callable = braintools.init.KaimingNormal(unit=u.mV),
        V_initializer: ArrayLike = braintools.init.Constant(0. * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.tau = braintools.init.param(tau, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_initializer = V_initializer

        # weights
        self.weight = brainstate.ParamState(braintools.init.param(w_init, (self.in_size[-1], self.out_size[-1])))

    def init_state(self, batch_size, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)

    @property
    def spike(self):
        return self.get_spike(self.V.value)

    def get_spike(self, V):
        v_scaled = (V - self.V_th) / self.V_th
        return self.spk_fun(v_scaled)

    def update(self, spk):
        # reset
        last_V = self.V.value
        last_spike = self.get_spike(last_V)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_V)
        V = last_V - V_th * last_spike
        # membrane potential
        x = spk @ self.weight.value
        dv = lambda v: (-v + self.sum_current_inputs(x, v)) / self.tau
        V = brainstate.nn.exp_euler_step(dv, V)
        self.V.value = self.sum_delta_inputs(V)
        return self.get_spike(V)
