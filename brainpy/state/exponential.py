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
from brainstate.typing import Size, ArrayLike

from brainpy.state._base import Synapse
from brainpy.state._mixin import AlignPost

__all__ = [
    'Expon', 'DualExpon',
]


class Expon(Synapse, AlignPost):
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

    See Also
    --------
    DualExpon : Dual-exponential synapse with separate rise and decay.
    Alpha : Alpha-function synapse with equal rise and decay time constants.

    Notes
    -----
    The implementation uses an exponential Euler integration method.
    The output of this synapse is the conductance value.

    This class inherits from :py:class:`AlignPost`, which means it can be used in projection patterns
    where synaptic variables are aligned with post-synaptic neurons, enabling event-driven
    computation and more efficient handling of sparse connectivity patterns.

    References
    ----------
    .. [1] Roth, A., & van Rossum, M. C. W. (2009). Modeling synapses.
           In Computational Modeling Methods for Neuroscientists (pp. 139-160).
           MIT Press.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>> # Create a simple exponential synapse with 8 ms decay
        >>> syn = brainpy.state.Expon(100, tau=8.*u.ms)
        >>> syn.init_state(batch_size=1)
        >>> # Step the synapse (conductance decays exponentially)
        >>> g = syn.update()
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        name: str | None = None,
        tau: ArrayLike = 8.0 * u.ms,
        g_initializer: ArrayLike | Callable = braintools.init.Constant(0. * u.mS),
    ):
        super().__init__(name=name, in_size=in_size)

        # parameters
        self.tau = braintools.init.param(tau, self.varshape)
        self.g_initializer = g_initializer

    def init_state(self, batch_size: int | None = None, **kwargs):
        self.g = brainstate.HiddenState.init(self.g_initializer, self.varshape, batch_size)

    def reset_state(self, batch_size: int | None = None, **kwargs):
        self.g.value = braintools.init.param(self.g_initializer, self.varshape, batch_size)

    def update(self, x=None):
        g = brainstate.nn.exp_euler_step(lambda g: self.sum_current_inputs(-g) / self.tau, self.g.value)
        self.g.value = self.sum_delta_inputs(g)
        if x is not None: self.g.value += x
        return self.g.value


class DualExpon(Synapse, AlignPost):
    r"""
    Dual exponential synapse model.

    This class implements a synapse model with separate rise and decay time constants,
    which produces a more biologically realistic conductance waveform than a single
    exponential model. The model is characterized by the differential equation system:

    dg_rise/dt = -g_rise/tau_rise
    dg_decay/dt = -g_decay/tau_decay
    g = a * (g_decay - g_rise)

    where $a$ is a scaling factor for the output waveform.

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
    normalize : bool, default=True
        Whether to use peak normalization for the dual-exponential waveform.
    amplitude : ArrayLike, default=1.
        Output amplitude scaling factor. When ``normalize=True``, the waveform is
        peak-normalized to 1 and then scaled by ``amplitude``. When ``normalize=False``,
        the raw difference waveform is scaled directly by ``amplitude``.
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
    normalize : bool
        Whether peak normalization is enabled.
    amplitude : Parameter
        Output amplitude scaling factor.

    See Also
    --------
    Expon : Single-exponential decay synapse.
    Alpha : Alpha-function synapse (special case where tau_rise == tau_decay).

    Notes
    -----
    The dual exponential model produces a conductance waveform that is more
    physiologically realistic than a simple exponential decay, with a finite
    rise time followed by a slower decay.

    The implementation uses an exponential Euler integration method.
    The output of this synapse is the difference between decay and rise components,
    optionally peak-normalized and scaled by ``amplitude``.

    If ``normalize=True``, the peak of the waveform is normalized to ``amplitude``.
    If ``normalize=False``, the raw waveform is scaled directly by ``amplitude``.

    This class inherits from :py:class:`AlignPost`, which means it can be used in projection patterns
    where synaptic variables are aligned with post-synaptic neurons, enabling event-driven
    computation and more efficient handling of sparse connectivity patterns.

    References
    ----------
    .. [1] Roth, A., & van Rossum, M. C. W. (2009). Modeling synapses.
           In Computational Modeling Methods for Neuroscientists (pp. 139-160).
           MIT Press.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> import brainpy
        >>> with brainstate.environ.context(dt=0.1 * u.ms):
        ...     syn = brainpy.state.DualExpon(in_size=1, tau_rise=0.5 * u.ms, tau_decay=5.0 * u.ms)
        ...     syn.init_state()
        ...     T, t0 = 300, 50
        ...     g = []
        ...     for t in range(T):
        ...         x = (1.0 * u.mS if t == t0 else 0.0 * u.mS)
        ...         y = u.get_magnitude(syn.update(x=x) / u.mS)
        ...         g.append(float(y[0]))
        
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        name: str | None = None,
        tau_decay: ArrayLike = 10.0 * u.ms,
        tau_rise: ArrayLike = 1.0 * u.ms,
        amplitude: ArrayLike = 1.0,
        normalize: bool = True,
        g_initializer: ArrayLike | Callable = braintools.init.Constant(0. * u.mS),
    ):
        super().__init__(name=name, in_size=in_size)

        # parameters
        self.tau_decay = braintools.init.param(tau_decay, self.varshape)
        self.tau_rise = braintools.init.param(tau_rise, self.varshape)
        self.amplitude = braintools.init.param(amplitude, self.varshape)
        self.normalize = normalize
        self.g_initializer = g_initializer

        # validate tau_rise != tau_decay when normalize is enabled
        if self.normalize:
            brainstate.transform.jit_error_if(
                u.math.any(u.get_magnitude(self.tau_decay - self.tau_rise) == 0.),
                'tau_decay must differ from tau_rise when normalize=True. '
                'Use Alpha synapse for equal time constants.'
            )

    def _dual_exp_normalization(self):
        return (
            self.tau_decay / (self.tau_decay - self.tau_rise) *
            u.math.float_power(
                self.tau_rise / self.tau_decay,
                self.tau_rise / (self.tau_rise - self.tau_decay)
            )
        )

    def init_state(self, batch_size: int | None = None, **kwargs):
        self.g_rise = brainstate.HiddenState.init(self.g_initializer, self.varshape, batch_size)
        self.g_decay = brainstate.HiddenState.init(self.g_initializer, self.varshape, batch_size)

    def reset_state(self, batch_size: int | None = None, **kwargs):
        self.g_rise.value = braintools.init.param(self.g_initializer, self.varshape, batch_size)
        self.g_decay.value = braintools.init.param(self.g_initializer, self.varshape, batch_size)

    def update(self, x=None):
        g_rise = brainstate.nn.exp_euler_step(lambda h: -h / self.tau_rise, self.g_rise.value)
        g_decay = brainstate.nn.exp_euler_step(lambda g: -g / self.tau_decay, self.g_decay.value)

        delta0 = u.math.zeros_like(self.g_rise.value)
        delta = self.sum_delta_inputs(delta0)

        self.g_rise.value = g_rise + delta
        self.g_decay.value = g_decay + delta

        if x is not None:
            self.g_rise.value += x
            self.g_decay.value += x

        scale = self.amplitude
        if self.normalize:
            scale = scale * self._dual_exp_normalization()

        return scale * (self.g_decay.value - self.g_rise.value)
