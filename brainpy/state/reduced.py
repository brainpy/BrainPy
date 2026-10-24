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

from collections.abc import Callable

import brainstate
import braintools
import brainunit as u
from brainstate.typing import ArrayLike, Size

from brainpy.state._base import Neuron

__all__ = [
    'FitzHughNagumo', 'HindmarshRose',
]


class FitzHughNagumo(Neuron):
    r"""FitzHugh-Nagumo neuron model.

    A two-dimensional reduction of Hodgkin-Huxley dynamics describing an
    excitable membrane with a fast voltage-like variable :math:`V` and a slow
    recovery variable :math:`w`:

    .. math::

        \tau \frac{dV}{dt}    &= V - \frac{V^3}{3} - w + I(t) \\
        \tau_w \frac{dw}{dt}  &= V + a - b\, w

    The state variables are dimensionless; the time constants ``tau`` and
    ``tau_w`` set the physical timescale. Like the Hodgkin-Huxley family, the
    model has **no reset** — :math:`V` rides a continuous limit cycle — so the
    spike output uses rising-edge detection (one spike per upward threshold
    crossing) rather than a per-step threshold test.

    Parameters
    ----------
    in_size : Size
        Size of the neuron group.
    a : ArrayLike, default=0.7
        Recovery offset.
    b : ArrayLike, default=0.8
        Recovery coupling.
    tau : ArrayLike, default=1. * u.ms
        Fast (voltage) time constant.
    tau_w : ArrayLike, default=12.5 * u.ms
        Slow (recovery) time constant.
    V_th : ArrayLike, default=1.0
        Threshold (dimensionless) used for rising-edge spike detection.
    V_initializer, w_initializer : Callable
        Initializers for the membrane and recovery variables.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function.
    spk_reset : str, default='soft'
        Retained for API compatibility; the model does not reset.
    name : str, optional
        Name of the neuron layer.

    See Also
    --------
    HindmarshRose : 3-D bursting reduced model.
    HH : Full Hodgkin-Huxley model.

    References
    ----------
    .. [1] FitzHugh, R. (1961). Impulses and physiological states in theoretical
           models of nerve membrane. Biophysical Journal, 1(6), 445-466.
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        a: ArrayLike = 0.7,
        b: ArrayLike = 0.8,
        tau: ArrayLike = 1. * u.ms,
        tau_w: ArrayLike = 12.5 * u.ms,
        V_th: ArrayLike = 1.0,
        V_initializer: Callable = braintools.init.Constant(-1.2),
        w_initializer: Callable = braintools.init.Constant(-0.62),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str | None = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.tau_w = braintools.init.param(tau_w, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_initializer = V_initializer
        self.w_initializer = w_initializer

    def init_state(self, batch_size: int | None = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.w = brainstate.HiddenState(braintools.init.param(self.w_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int | None = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.w.value = braintools.init.param(self.w_initializer, self.varshape, batch_size)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / self.V_th
        return self.spk_fun(v_scaled)

    def update(self, x=0.):
        last_V = self.V.value
        last_w = self.w.value
        I_total = self.sum_current_inputs(x, last_V)
        dV = lambda V: (V - V ** 3 / 3. - last_w + I_total) / self.tau
        dw = lambda w: (last_V + self.a - self.b * w) / self.tau_w
        V = brainstate.nn.exp_euler_step(dV, last_V)
        w = brainstate.nn.exp_euler_step(dw, last_w)
        V = self.sum_delta_inputs(V)
        self.V.value = V
        self.w.value = w
        # Rising-edge detection: emit a spike only when V crosses threshold upward.
        # The model never resets, so V stays above threshold for the whole spike;
        # a plain per-step threshold test would report a spike on every such step.
        return self.get_spike(V) * (1. - self.get_spike(last_V))


class HindmarshRose(Neuron):
    r"""Hindmarsh-Rose neuron model.

    A three-dimensional reduced model capable of **bursting**, with a fast
    voltage-like variable :math:`V`, a fast recovery variable :math:`y`, and a
    slow adaptation current :math:`z`:

    .. math::

        \tau \frac{dV}{dt} &= y - a V^3 + b V^2 - z + I(t) \\
        \tau \frac{dy}{dt} &= c - d V^2 - y \\
        \tau \frac{dz}{dt} &= r\,(s\,(V - V_r) - z)

    The state variables are dimensionless; ``tau`` sets the physical timescale
    and the small rate ``r`` makes :math:`z` slow, producing bursts. As with the
    Hodgkin-Huxley family the model has **no reset**, so spike output uses
    rising-edge detection. ``V`` is the fast voltage-like variable (``x`` in the
    classic Hindmarsh-Rose notation).

    Parameters
    ----------
    in_size : Size
        Size of the neuron group.
    a, b, c, d : ArrayLike
        Polynomial coefficients (defaults ``1, 3, 1, 5``).
    r : ArrayLike, default=0.006
        Slow-variable rate; smaller values give slower bursting.
    s : ArrayLike, default=4.0
        Adaptation coupling.
    V_r : ArrayLike, default=-1.6
        Resting value of the fast variable in the adaptation equation.
    tau : ArrayLike, default=1. * u.ms
        Time constant setting the physical timescale.
    V_th : ArrayLike, default=1.0
        Threshold (dimensionless) used for rising-edge spike detection.
    V_initializer, y_initializer, z_initializer : Callable
        Initializers for the three state variables.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function.
    spk_reset : str, default='soft'
        Retained for API compatibility; the model does not reset.
    name : str, optional
        Name of the neuron layer.

    See Also
    --------
    FitzHughNagumo : 2-D excitable reduced model.
    Izhikevich : 2-D quadratic model with reset.

    References
    ----------
    .. [1] Hindmarsh, J. L., & Rose, R. M. (1984). A model of neuronal bursting
           using three coupled first order differential equations. Proceedings
           of the Royal Society of London. Series B, 221(1222), 87-102.
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        a: ArrayLike = 1.0,
        b: ArrayLike = 3.0,
        c: ArrayLike = 1.0,
        d: ArrayLike = 5.0,
        r: ArrayLike = 0.006,
        s: ArrayLike = 4.0,
        V_r: ArrayLike = -1.6,
        tau: ArrayLike = 1. * u.ms,
        V_th: ArrayLike = 1.0,
        V_initializer: Callable = braintools.init.Constant(-1.6),
        y_initializer: Callable = braintools.init.Constant(-10.),
        z_initializer: Callable = braintools.init.Constant(0.),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str | None = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)
        self.c = braintools.init.param(c, self.varshape)
        self.d = braintools.init.param(d, self.varshape)
        self.r = braintools.init.param(r, self.varshape)
        self.s = braintools.init.param(s, self.varshape)
        self.V_r = braintools.init.param(V_r, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_initializer = V_initializer
        self.y_initializer = y_initializer
        self.z_initializer = z_initializer

    def init_state(self, batch_size: int | None = None, **kwargs):
        self.V = brainstate.HiddenState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.y = brainstate.HiddenState(braintools.init.param(self.y_initializer, self.varshape, batch_size))
        self.z = brainstate.HiddenState(braintools.init.param(self.z_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int | None = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.y.value = braintools.init.param(self.y_initializer, self.varshape, batch_size)
        self.z.value = braintools.init.param(self.z_initializer, self.varshape, batch_size)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / self.V_th
        return self.spk_fun(v_scaled)

    def update(self, x=0.):
        last_V = self.V.value
        last_y = self.y.value
        last_z = self.z.value
        I_total = self.sum_current_inputs(x, last_V)
        dV = lambda V: (last_y - self.a * V ** 3 + self.b * V ** 2 - last_z + I_total) / self.tau
        dy = lambda y: (self.c - self.d * last_V ** 2 - y) / self.tau
        dz = lambda z: (self.r * (self.s * (last_V - self.V_r) - z)) / self.tau
        V = brainstate.nn.exp_euler_step(dV, last_V)
        y = brainstate.nn.exp_euler_step(dy, last_y)
        z = brainstate.nn.exp_euler_step(dz, last_z)
        V = self.sum_delta_inputs(V)
        self.V.value = V
        self.y.value = y
        self.z.value = z
        # Rising-edge detection (no reset; see FitzHughNagumo.update).
        return self.get_spike(V) * (1. - self.get_spike(last_V))
