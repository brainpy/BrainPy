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

from brainstate.typing import Size, ArrayLike
import brainstate
import braintools
import brainunit as u
from ._base import Synapse

__all__ = [
     'Expon', 'DualExpon',
]


class Expon(Synapse, brainstate.mixin.AlignPost):
    r"""
    Exponential decay synapse model.

    This class implements a simple first-order exponential decay synapse model where
    the synaptic conductance g decays exponentially with time constant tau:

    $$
    dg/dt = -g/\tau + \text{input}
    $$

    The model is widely used for basic synaptic transmission modeling.

    Parameters
    ----------
    in_size : Size
        Size of the input.
    name : str, optional
        Name of the synapse instance.
    tau : ArrayLike, default=8.0*u.ms
        Time constant of decay in milliseconds.
    g_initializer : ArrayLike or Callable, default=init.Constant(0. * u.mS)
        Initial value or initializer for synaptic conductance.

    Attributes
    ----------
    g : HiddenState
        Synaptic conductance state variable.
    tau : Parameter
        Time constant of decay.

    Notes
    -----
    The implementation uses an exponential Euler integration method.
    The output of this synapse is the conductance value.

    This class inherits from :py:class:`AlignPost`, which means it can be used in projection patterns
    where synaptic variables are aligned with post-synaptic neurons, enabling event-driven
    computation and more efficient handling of sparse connectivity patterns.
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

    def reset_state(self, batch_size: int = None, **kwargs):
        self.g.value = braintools.init.param(self.g_initializer, self.varshape, batch_size)

    def update(self, x=None):
        g = brainstate.nn.exp_euler_step(lambda g: self.sum_current_inputs(-g) / self.tau, self.g.value)
        self.g.value = self.sum_delta_inputs(g)
        if x is not None: self.g.value += x
        return self.g.value


class DualExpon(Synapse, brainstate.mixin.AlignPost):
    r"""
    Dual exponential synapse model.

    This class implements a synapse model with separate rise and decay time constants,
    which produces a more biologically realistic conductance waveform than a single
    exponential model. The model is characterized by the differential equation system:

    dg_rise/dt = -g_rise/tau_rise
    dg_decay/dt = -g_decay/tau_decay
    g = a * (g_decay - g_rise)

    where $a$ is a normalization factor that ensures the peak conductance reaches
    the desired amplitude.

    Parameters
    ----------
    in_size : Size
        Size of the input.
    name : str, optional
        Name of the synapse instance.
    tau_decay : ArrayLike, default=10.0*u.ms
        Time constant of decay in milliseconds.
    tau_rise : ArrayLike, default=1.0*u.ms
        Time constant of rise in milliseconds.
    A : ArrayLike, optional
        Amplitude scaling factor. If None, a scaling factor is automatically
        calculated to normalize the peak amplitude.
    g_initializer : ArrayLike or Callable, default=init.Constant(0. * u.mS)
        Initial value or initializer for synaptic conductance.

    Attributes
    ----------
    g_rise : HiddenState
        Rise component of synaptic conductance.
    g_decay : HiddenState
        Decay component of synaptic conductance.
    tau_rise : Parameter
        Time constant of rise phase.
    tau_decay : Parameter
        Time constant of decay phase.
    a : Parameter
        Normalization factor calculated from tau_rise, tau_decay, and A.

    Notes
    -----
    The dual exponential model produces a conductance waveform that is more
    physiologically realistic than a simple exponential decay, with a finite
    rise time followed by a slower decay.

    The implementation uses an exponential Euler integration method.
    The output of this synapse is the normalized difference between decay and rise components.

    This class inherits from :py:class:`AlignPost`, which means it can be used in projection patterns
    where synaptic variables are aligned with post-synaptic neurons, enabling event-driven
    computation and more efficient handling of sparse connectivity patterns.
    """
    __module__ = 'brainpy'

    def __init__(
        self,
        in_size: Size,
        name: Optional[str] = None,
        tau_decay: ArrayLike = 10.0 * u.ms,
        tau_rise: ArrayLike = 1.0 * u.ms,
        A: Optional[ArrayLike] = None,
        g_initializer: ArrayLike | Callable = braintools.init.Constant(0. * u.mS),
    ):
        super().__init__(name=name, in_size=in_size)

        # parameters
        self.tau_decay = braintools.init.param(tau_decay, self.varshape)
        self.tau_rise = braintools.init.param(tau_rise, self.varshape)
        A = self._format_dual_exp_A(A)
        self.a = (self.tau_decay - self.tau_rise) / self.tau_rise / self.tau_decay * A
        self.g_initializer = g_initializer

    def _format_dual_exp_A(self, A):
        A = braintools.init.param(A, sizes=self.varshape, allow_none=True)
        if A is None:
            A = (
                self.tau_decay / (self.tau_decay - self.tau_rise) *
                u.math.float_power(self.tau_rise / self.tau_decay,
                                   self.tau_rise / (self.tau_rise - self.tau_decay))
            )
        return A

    def init_state(self, batch_size: int = None, **kwargs):
        self.g_rise = brainstate.HiddenState(braintools.init.param(self.g_initializer, self.varshape, batch_size))
        self.g_decay = brainstate.HiddenState(braintools.init.param(self.g_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.g_rise.value = braintools.init.param(self.g_initializer, self.varshape, batch_size)
        self.g_decay.value = braintools.init.param(self.g_initializer, self.varshape, batch_size)

    def update(self, x=None):
        g_rise = brainstate.nn.exp_euler_step(lambda h: -h / self.tau_rise, self.g_rise.value)
        g_decay = brainstate.nn.exp_euler_step(lambda g: -g / self.tau_decay, self.g_decay.value)
        self.g_rise.value = self.sum_delta_inputs(g_rise)
        self.g_decay.value = self.sum_delta_inputs(g_decay)
        if x is not None:
            self.g_rise.value += x
            self.g_decay.value += x
        return self.a * (self.g_decay.value - self.g_rise.value)
