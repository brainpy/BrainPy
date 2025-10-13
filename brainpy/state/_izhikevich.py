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
    'Izhikevich', 'IzhikevichRef',
]


class Izhikevich(Neuron):
    r"""Izhikevich neuron model.

    This class implements the Izhikevich neuron model, a two-dimensional spiking neuron
    model that can reproduce a wide variety of neuronal firing patterns observed in
    biological neurons. The model combines computational efficiency with biological
    plausibility through a quadratic voltage dynamics and a linear recovery variable.

    The model is characterized by the following differential equations:

    $$
    \frac{dV}{dt} = 0.04 V^2 + 5V + 140 - u + I(t)
    $$

    $$
    \frac{du}{dt} = a(bV - u)
    $$

    Spike condition:
    If $V \geq V_{th}$: emit spike, set $V = c$ and $u = u + d$

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    a : ArrayLike, default=0.02 / u.ms
        Time scale of the recovery variable u. Smaller values result in slower recovery.
    b : ArrayLike, default=0.2 / u.ms
        Sensitivity of the recovery variable u to the membrane potential V.
    c : ArrayLike, default=-65. * u.mV
        After-spike reset value of the membrane potential.
    d : ArrayLike, default=8. * u.mV / u.ms
        After-spike increment of the recovery variable u.
    V_th : ArrayLike, default=30. * u.mV
        Spike threshold voltage.
    V_initializer : Callable
        Initializer for the membrane potential state.
    u_initializer : Callable
        Initializer for the recovery variable state.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function for the non-differentiable spike generation.
    spk_reset : str, default='hard'
        Reset mechanism after spike generation:
        - 'soft': subtract threshold V = V - V_th
        - 'hard': strict reset using stop_gradient
    name : str, optional
        Name of the neuron layer.

    Attributes
    ----------
    V : HiddenState
        Membrane potential.
    u : HiddenState
        Recovery variable.

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create an Izhikevich neuron layer with 10 neurons
    >>> izh = brainpy.state.Izhikevich(10)
    >>>
    >>> # Initialize the state
    >>> izh.init_state(batch_size=1)
    >>>
    >>> # Apply an input current and update the neuron state
    >>> spikes = izh.update(x=10.*u.mV/u.ms)

    Notes
    -----
    - The quadratic term in the voltage equation (0.04*V^2) provides a sharp spike
      upstroke similar to biological neurons.
    - Different combinations of parameters (a, b, c, d) can reproduce various neuronal
      behaviors including regular spiking, intrinsically bursting, chattering, and
      fast spiking.
    - The model uses a hard reset mechanism where V is set to c and u is incremented
      by d when a spike occurs.
    - Parameter ranges: a ∈ [0.01, 0.1], b ∈ [0.2, 0.3], c ∈ [-65, -50], d ∈ [0.1, 10]

    References
    ----------
    .. [1] Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions
           on neural networks, 14(6), 1569-1572.
    .. [2] Izhikevich, E. M. (2004). Which model to use for cortical spiking neurons?.
           IEEE transactions on neural networks, 15(5), 1063-1070.
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        a: ArrayLike = 0.02 / u.ms,
        b: ArrayLike = 0.2 / u.ms,
        c: ArrayLike = -65. * u.mV,
        d: ArrayLike = 8. * u.mV / u.ms,
        V_th: ArrayLike = 30. * u.mV,
        V_initializer: Callable = braintools.init.Constant(-65. * u.mV),
        u_initializer: Callable = braintools.init.Constant(0. * u.mV / u.ms),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)
        self.c = braintools.init.param(c, self.varshape)
        self.d = braintools.init.param(d, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)

        # pre-computed coefficients for quadratic equation
        self.p1 = 0.04 / (u.ms * u.mV)
        self.p2 = 5. / u.ms
        self.p3 = 140. * u.mV / u.ms

        # initializers
        self.V_initializer = V_initializer
        self.u_initializer = u_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.u = brainstate.HiddenState(braintools.init.param(self.u_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.u.value = braintools.init.param(self.u_initializer, self.varshape, batch_size)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / self.V_th
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mV / u.ms):
        last_v = self.V.value
        last_u = self.u.value
        last_spk = self.get_spike(last_v)

        # Izhikevich uses hard reset: V → c, u → u + d
        V = u.math.where(last_spk > 0., self.c, last_v)
        u_val = last_u + self.d * last_spk

        # voltage dynamics: dV/dt = 0.04*V^2 + 5*V + 140 - u + I
        def dv(v):
            I_total = self.sum_current_inputs(x, v)
            return self.p1 * v * v + self.p2 * v + self.p3 - u_val + I_total

        # recovery dynamics: du/dt = a(bV - u)
        def du(u_):
            return self.a * (self.b * V - u_)

        V = brainstate.nn.exp_euler_step(dv, V)
        V = self.sum_delta_inputs(V)
        u_val = brainstate.nn.exp_euler_step(du, u_val)

        self.V.value = V
        self.u.value = u_val
        return self.get_spike(V)


class IzhikevichRef(Neuron):
    r"""Izhikevich neuron model with refractory period.

    This class implements the Izhikevich neuron model with an absolute refractory period.
    During the refractory period after a spike, the neuron cannot fire regardless of input,
    which better captures the behavior of biological neurons that exhibit a recovery period
    after action potential generation.

    The model is characterized by the following equations:

    When not in refractory period:

    $$
    \frac{dV}{dt} = 0.04 V^2 + 5V + 140 - u + I(t)
    $$

    $$
    \frac{du}{dt} = a(bV - u)
    $$

    During refractory period:

    $$
    V = c, \quad u = u
    $$

    Spike condition:
    If $V \geq V_{th}$ and not in refractory period: emit spike, set $V = c$, $u = u + d$,
    and enter refractory period for $\tau_{ref}$

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    a : ArrayLike, default=0.02 / u.ms
        Time scale of the recovery variable u.
    b : ArrayLike, default=0.2 / u.ms
        Sensitivity of the recovery variable u to the membrane potential V.
    c : ArrayLike, default=-65. * u.mV
        After-spike reset value of the membrane potential.
    d : ArrayLike, default=8. * u.mV / u.ms
        After-spike increment of the recovery variable u.
    V_th : ArrayLike, default=30. * u.mV
        Spike threshold voltage.
    tau_ref : ArrayLike, default=0. * u.ms
        Refractory period duration.
    V_initializer : Callable
        Initializer for the membrane potential state.
    u_initializer : Callable
        Initializer for the recovery variable state.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function for the non-differentiable spike generation.
    spk_reset : str, default='hard'
        Reset mechanism after spike generation.
    ref_var : bool, default=False
        Whether to expose a boolean refractory state variable.
    name : str, optional
        Name of the neuron layer.

    Attributes
    ----------
    V : HiddenState
        Membrane potential.
    u : HiddenState
        Recovery variable.
    last_spike_time : ShortTermState
        Time of the last spike, used to implement refractory period.
    refractory : HiddenState
        Neuron refractory state (if ref_var=True).

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create an IzhikevichRef neuron layer with 10 neurons
    >>> izh_ref = brainpy.state.IzhikevichRef(10, tau_ref=2.*u.ms)
    >>>
    >>> # Initialize the state
    >>> izh_ref.init_state(batch_size=1)
    >>>
    >>> # Generate inputs and run simulation
    >>> time_steps = 100
    >>> inputs = brainstate.random.randn(time_steps, 1, 10) * u.mV / u.ms
    >>>
    >>> with brainstate.environ.context(dt=0.1 * u.ms):
    >>>     for t in range(time_steps):
    >>>         with brainstate.environ.context(t=t*0.1*u.ms):
    >>>             spikes = izh_ref.update(x=inputs[t])

    Notes
    -----
    - The refractory period is implemented by tracking the time of the last spike
      and preventing membrane potential updates if the elapsed time is less than tau_ref.
    - During the refractory period, the membrane potential remains at the reset value c
      regardless of input current strength.
    - Refractory periods prevent high-frequency repetitive firing and are critical
      for realistic neural dynamics.
    - The simulation environment time variable 't' is used to track the refractory state.

    References
    ----------
    .. [1] Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions
           on neural networks, 14(6), 1569-1572.
    .. [2] Izhikevich, E. M. (2004). Which model to use for cortical spiking neurons?.
           IEEE transactions on neural networks, 15(5), 1063-1070.
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        a: ArrayLike = 0.02 / u.ms,
        b: ArrayLike = 0.2 / u.ms,
        c: ArrayLike = -65. * u.mV,
        d: ArrayLike = 8. * u.mV / u.ms,
        V_th: ArrayLike = 30. * u.mV,
        tau_ref: ArrayLike = 0. * u.ms,
        V_initializer: Callable = braintools.init.Constant(-65. * u.mV),
        u_initializer: Callable = braintools.init.Constant(0. * u.mV / u.ms),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)
        self.c = braintools.init.param(c, self.varshape)
        self.d = braintools.init.param(d, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.tau_ref = braintools.init.param(tau_ref, self.varshape)

        # pre-computed coefficients for quadratic equation
        self.p1 = 0.04 / (u.ms * u.mV)
        self.p2 = 5. / u.ms
        self.p3 = 140. * u.mV / u.ms

        # initializers
        self.V_initializer = V_initializer
        self.u_initializer = u_initializer
        self.ref_var = ref_var

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.u = brainstate.HiddenState(braintools.init.param(self.u_initializer, self.varshape, batch_size))
        self.last_spike_time = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        )
        if self.ref_var:
            self.refractory = brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            )

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.u.value = braintools.init.param(self.u_initializer, self.varshape, batch_size)
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        if self.ref_var:
            self.refractory.value = braintools.init.param(
                braintools.init.Constant(False), self.varshape, batch_size
            )

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / self.V_th
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mV / u.ms):
        t = brainstate.environ.get('t')
        last_v = self.V.value
        last_u = self.u.value
        last_spk = self.get_spike(last_v)

        # Izhikevich uses hard reset: V → c, u → u + d
        v_reset = u.math.where(last_spk > 0., self.c, last_v)
        u_reset = last_u + self.d * last_spk

        # voltage dynamics: dV/dt = 0.04*V^2 + 5*V + 140 - u + I
        def dv(v):
            I_total = self.sum_current_inputs(x, v)
            return self.p1 * v * v + self.p2 * v + self.p3 - u_reset + I_total

        # recovery dynamics: du/dt = a(bV - u)
        def du(u_):
            return self.a * (self.b * V_candidate - u_)

        V_candidate = brainstate.nn.exp_euler_step(dv, v_reset)
        V_candidate = self.sum_delta_inputs(V_candidate)
        u_candidate = brainstate.nn.exp_euler_step(du, u_reset)

        # apply refractory period
        refractory = (t - self.last_spike_time.value) < self.tau_ref
        self.V.value = u.math.where(refractory, v_reset, V_candidate)
        self.u.value = u.math.where(refractory, u_reset, u_candidate)

        # spike time evaluation
        spike_cond = self.V.value >= self.V_th
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t, self.last_spike_time.value)
        )
        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(
                u.math.logical_or(refractory, spike_cond)
            )
        return self.get_spike()
