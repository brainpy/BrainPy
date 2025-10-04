# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

from typing import Optional

import brainunit as u

from brainstate._state import HiddenState
from brainstate.typing import ArrayLike, Size
from braintools import init as init
from brainstate.nn import exp_euler_step

__all__ = [
    'STP', 'STD',
]


class STP(ShortTermPlasticity):
    r"""
    Synapse with short-term plasticity.

    This class implements a synapse model with short-term plasticity (STP), which captures
    activity-dependent changes in synaptic efficacy that occur over milliseconds to seconds.
    The model simultaneously accounts for both short-term facilitation and depression
    based on the formulation by Tsodyks & Markram (1998).

    The model is characterized by the following equations:

    $$
    \frac{du}{dt} = -\frac{u}{\tau_f} + U \cdot (1 - u) \cdot \delta(t - t_{spike})
    $$

    $$
    \frac{dx}{dt} = \frac{1 - x}{\tau_d} - u \cdot x \cdot \delta(t - t_{spike})
    $$

    $$
    g_{syn} = u \cdot x
    $$

    where:
    - $u$ represents the utilization of synaptic efficacy (facilitation variable)
    - $x$ represents the available synaptic resources (depression variable)
    - $\tau_f$ is the facilitation time constant
    - $\tau_d$ is the depression time constant
    - $U$ is the baseline utilization parameter
    - $\delta(t - t_{spike})$ is the Dirac delta function representing presynaptic spikes
    - $g_{syn}$ is the effective synaptic conductance

    Parameters
    ----------
    in_size : Size
        Size of the input.
    name : str, optional
        Name of the synapse instance.
    U : ArrayLike, default=0.15
        Baseline utilization parameter (fraction of resources used per action potential).
    tau_f : ArrayLike, default=1500.*u.ms
        Time constant of short-term facilitation in milliseconds.
    tau_d : ArrayLike, default=200.*u.ms
        Time constant of short-term depression (recovery of synaptic resources) in milliseconds.

    Attributes
    ----------
    u : HiddenState
        Utilization of synaptic efficacy (facilitation variable).
    x : HiddenState
        Available synaptic resources (depression variable).

    Notes
    -----
    - Larger values of tau_f produce stronger facilitation effects.
    - Larger values of tau_d lead to slower recovery from depression.
    - The parameter U controls the initial release probability.
    - The effective synaptic strength is the product of u and x.

    References
    ----------
    .. [1] Tsodyks, M. V., & Markram, H. (1997). The neural code between neocortical
           pyramidal neurons depends on neurotransmitter release probability.
           Proceedings of the National Academy of Sciences, 94(2), 719-723.
    .. [2] Tsodyks, M., Pawelzik, K., & Markram, H. (1998). Neural networks with dynamic
           synapses. Neural computation, 10(4), 821-835.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        name: Optional[str] = None,
        U: ArrayLike = 0.15,
        tau_f: ArrayLike = 1500. * u.ms,
        tau_d: ArrayLike = 200. * u.ms,
    ):
        super().__init__(name=name, in_size=in_size)

        # parameters
        self.tau_f = init.param(tau_f, self.varshape)
        self.tau_d = init.param(tau_d, self.varshape)
        self.U = init.param(U, self.varshape)

    def init_state(self, batch_size: int = None, **kwargs):
        self.x = HiddenState(init.param(init.Constant(1.), self.varshape, batch_size))
        self.u = HiddenState(init.param(init.Constant(self.U), self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.x.value = init.param(init.Constant(1.), self.varshape, batch_size)
        self.u.value = init.param(init.Constant(self.U), self.varshape, batch_size)

    def update(self, pre_spike):
        u = exp_euler_step(lambda u: - u / self.tau_f, self.u.value)
        x = exp_euler_step(lambda x: (1 - x) / self.tau_d, self.x.value)

        # --- original code:
        #   if pre_spike.dtype == jax.numpy.bool_:
        #     u = bm.where(pre_spike, u + self.U * (1 - self.u), u)
        #     x = bm.where(pre_spike, x - u * self.x, x)
        #   else:
        #     u = pre_spike * (u + self.U * (1 - self.u)) + (1 - pre_spike) * u
        #     x = pre_spike * (x - u * self.x) + (1 - pre_spike) * x

        # --- simplified code:
        u = u + pre_spike * self.U * (1 - self.u.value)
        x = x - pre_spike * u * self.x.value

        self.u.value = u
        self.x.value = x
        return u * x * pre_spike


class STD(ShortTermPlasticity):
    r"""
    Synapse with short-term depression.

    This class implements a synapse model with short-term depression (STD), which captures
    activity-dependent reduction in synaptic efficacy, typically caused by depletion of
    neurotransmitter vesicles following repeated stimulation.

    The model is characterized by the following equation:

    $$
    \frac{dx}{dt} = \frac{1 - x}{\tau} - U \cdot x \cdot \delta(t - t_{spike})
    $$

    $$
    g_{syn} = x
    $$

    where:
    - $x$ represents the available synaptic resources (depression variable)
    - $\tau$ is the depression recovery time constant
    - $U$ is the utilization parameter (fraction of resources depleted per spike)
    - $\delta(t - t_{spike})$ is the Dirac delta function representing presynaptic spikes
    - $g_{syn}$ is the effective synaptic conductance

    Parameters
    ----------
    in_size : Size
        Size of the input.
    name : str, optional
        Name of the synapse instance.
    tau : ArrayLike, default=200.*u.ms
        Time constant governing recovery of synaptic resources in milliseconds.
    U : ArrayLike, default=0.07
        Utilization parameter (fraction of resources used per action potential).

    Attributes
    ----------
    x : HiddenState
        Available synaptic resources (depression variable).

    Notes
    -----
    - Larger values of tau lead to slower recovery from depression.
    - Larger values of U cause stronger depression with each spike.
    - This model is a simplified version of the STP model that only includes depression.

    References
    ----------
    .. [1] Abbott, L. F., Varela, J. A., Sen, K., & Nelson, S. B. (1997). Synaptic
           depression and cortical gain control. Science, 275(5297), 220-224.
    .. [2] Tsodyks, M. V., & Markram, H. (1997). The neural code between neocortical
           pyramidal neurons depends on neurotransmitter release probability.
           Proceedings of the National Academy of Sciences, 94(2), 719-723.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        name: Optional[str] = None,
        # synapse parameters
        tau: ArrayLike = 200. * u.ms,
        U: ArrayLike = 0.07,
    ):
        super().__init__(name=name, in_size=in_size)

        # parameters
        self.tau = init.param(tau, self.varshape)
        self.U = init.param(U, self.varshape)

    def init_state(self, batch_size: int = None, **kwargs):
        self.x = HiddenState(init.param(init.Constant(1.), self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.x.value = init.param(init.Constant(1.), self.varshape, batch_size)

    def update(self, pre_spike):
        x = exp_euler_step(lambda x: (1 - x) / self.tau, self.x.value)

        # --- original code:
        # self.x.value = bm.where(pre_spike, x - self.U * self.x, x)

        # --- simplified code:
        self.x.value = x - pre_spike * self.U * self.x.value

        return self.x.value * pre_spike
