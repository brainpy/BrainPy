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
from collections.abc import Callable

import brainstate
import braintools
import brainunit as u
from brainstate.typing import Size, ArrayLike

__all__ = [
    'LeakyRateReadout',
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

    See Also
    --------
    Notes
    -----
    - The decay factor :math:`\alpha = e^{-\Delta t / \tau}` is computed once
      at construction time using the current environment ``dt``.
    - The weight matrix is initialized with Kaiming Normal initialization,
      which is suitable for layers that follow ReLU-like activations.
    - This module does not produce spikes; it outputs continuous rate values,
      making it suitable as the final readout layer in spiking networks
      trained with surrogate gradients.

    References
    ----------
    .. [1] Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient
           learning in spiking neural networks. IEEE Signal Processing
           Magazine, 36(6), 51-63.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>> with brainstate.environ.context(dt=1. * u.ms):
        ...     readout = brainpy.state.LeakyRateReadout(128, 10, tau=5.*u.ms)
        ...     readout.init_state(batch_size=1)
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        tau: ArrayLike = 5. * u.ms,
        w_init: Callable = braintools.init.KaimingNormal(),
        name: str | None = None,
    ):
        super().__init__(name=name)

        # parameters
        self.in_size = (in_size,) if isinstance(in_size, numbers.Integral) else tuple(in_size)
        self.out_size = (out_size,) if isinstance(out_size, numbers.Integral) else tuple(out_size)
        # ``tau`` parameterises the leaky read-out state ``r``, which lives in
        # the output space, so it is sized to ``out_size`` (not ``in_size``).
        self.tau = braintools.init.param(tau, self.out_size)
        self.decay = u.math.exp(-brainstate.environ.get_dt() / self.tau)

        # weights
        self.weight = brainstate.ParamState(braintools.init.param(w_init, (self.in_size[0], self.out_size[0])))

    def init_state(self, batch_size=None, **kwargs):
        self.r = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(0.), self.out_size, batch_size)
        )

    def reset_state(self, batch_size=None, **kwargs):
        self.r.value = braintools.init.param(
            braintools.init.Constant(0.), self.out_size, batch_size
        )

    def update(self, x):
        self.r.value = self.decay * self.r.value + x @ self.weight.value
        return self.r.value
