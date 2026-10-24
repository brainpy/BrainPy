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

import brainstate
import brainunit as u

from brainpy.state._mixin import BindCondData

__all__ = [
    'SynOut', 'COBA', 'CUBA', 'MgBlock',
]


class SynOut(brainstate.nn.Module, BindCondData):
    r"""
    Base class for synaptic output modules.

    ``SynOut`` defines the interface for converting synaptic conductance values
    into post-synaptic currents. Subclasses implement specific biophysical
    models (conductance-based, current-based, magnesium-blocked) by overriding
    the :meth:`update` method.

    Before calling, conductance data for the current time step must be bound
    via :meth:`~BindCondData.bind_cond`.

    See Also
    --------
    COBA : Conductance-based synaptic output.
    CUBA : Current-based synaptic output.
    MgBlock : Synaptic output with voltage-dependent magnesium block.

    Notes
    -----
    This class also inherits from :class:`~brainpy.state._mixin.BindCondData`,
    which provides the :meth:`bind_cond` method for temporarily storing
    conductance data between computation steps within a single time step.

    Subclasses must implement ``update(self, conductance, potential)`` which
    receives the bound conductance and the post-synaptic membrane potential,
    and returns the resulting synaptic current.
    """

    __module__ = 'brainpy.state'

    def __init__(self, ):
        super().__init__()
        self._conductance = None

    def __call__(self, *args, **kwargs):
        if self._conductance is None:
            raise ValueError(
                f'Please first pack conductance data at the current step using '
                f'".{BindCondData.bind_cond.__name__}(data)". {self}'
            )
        ret = self.update(self._conductance, *args, **kwargs)
        return ret

    def update(self, conductance, potential):
        raise NotImplementedError


class COBA(SynOut):
    r"""
    Conductance-based (COBA) synaptic output.

    Converts synaptic conductance into post-synaptic current using the
    driving-force model. The output current depends on the difference between
    the membrane potential and the reversal potential:

    .. math::

       I_{\mathrm{syn}}(t) = g_{\mathrm{syn}}(t) \, (E - V(t))

    where :math:`g_{\mathrm{syn}}` is the instantaneous synaptic conductance,
    :math:`E` is the reversal potential, and :math:`V` is the post-synaptic
    membrane potential.

    Parameters
    ----------
    E : ArrayLike
        Reversal potential of the synapse. Typical values: 0 mV for
        excitatory (AMPA/NMDA) and -80 mV for inhibitory (GABAa) synapses.

    See Also
    --------
    CUBA : Current-based synaptic output (potential-independent).
    MgBlock : Conductance-based output with voltage-dependent Mg2+ block.

    Notes
    -----
    - When :math:`V < E`, the driving force :math:`(E - V)` is positive,
      producing a depolarizing (excitatory) current. When :math:`V > E`,
      the current is hyperpolarizing (inhibitory).
    - The COBA model is more biologically realistic than CUBA because the
      synaptic current magnitude depends on the membrane potential, providing
      automatic gain control and preventing reversal of current direction
      past the equilibrium potential [1]_.

    References
    ----------
    .. [1] Destexhe, A., Mainen, Z. F., & Sejnowski, T. J. (1994). Synthesis
           of models for excitable membranes, synaptic transmission and
           neuromodulation using a common kinetic formalism. Journal of
           Computational Neuroscience, 1(3), 195-230.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy
        >>> import brainunit as u
        >>> # Excitatory COBA synapse with reversal at 0 mV
        >>> coba_exc = brainpy.state.COBA(E=0. * u.mV)
        >>> # Inhibitory COBA synapse with reversal at -80 mV
        >>> coba_inh = brainpy.state.COBA(E=-80. * u.mV)
    """
    __module__ = 'brainpy.state'

    def __init__(self, E: brainstate.typing.ArrayLike):
        super().__init__()

        self.E = E

    def update(self, conductance, potential):
        return conductance * (self.E - potential)


class CUBA(SynOut):
    r"""Current-based (CUBA) synaptic output.

    Converts synaptic conductance directly into post-synaptic current by
    applying a fixed scaling factor, independent of the membrane potential:

    .. math::

       I_{\mathrm{syn}}(t) = g_{\mathrm{syn}}(t) \cdot \text{scale}

    where :math:`g_{\mathrm{syn}}` is the instantaneous synaptic conductance
    and ``scale`` is a constant conversion factor.

    Parameters
    ----------
    scale : ArrayLike, default=u.volt
        Scaling factor applied to the conductance to obtain the current.
        The default value of ``u.volt`` converts conductance in siemens
        to current in amperes.

    See Also
    --------
    COBA : Conductance-based synaptic output (potential-dependent).

    Notes
    -----
    - The CUBA model ignores the post-synaptic membrane potential entirely,
      making the synaptic current a fixed function of the presynaptic
      activity alone.
    - This simplification is computationally cheaper than COBA and can be
      appropriate when the membrane potential remains close to rest or when
      modeling abstract spiking networks [1]_.
    - In practice, ``scale`` should be chosen so that
      ``conductance * scale`` has units compatible with the target neuron's
      current input (e.g., millivolts for dimensionless models, or amperes
      for biophysical models).

    References
    ----------
    .. [1] Brette, R., et al. (2007). Simulation of networks of spiking
           neurons: a review of tools and strategies. Journal of Computational
           Neuroscience, 23(3), 349-398.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy
        >>> import brainunit as u
        >>> # Default CUBA synapse
        >>> cuba = brainpy.state.CUBA()
        >>> # CUBA with custom scaling
        >>> cuba_scaled = brainpy.state.CUBA(scale=0.5 * u.volt)
    """
    __module__ = 'brainpy.state'

    def __init__(self, scale: brainstate.typing.ArrayLike = u.volt):
        super().__init__()
        self.scale = scale

    def update(self, conductance, potential=None):
        return conductance * self.scale


class MgBlock(SynOut):
    r"""Synaptic output with voltage-dependent magnesium (Mg2+) block.

    Models NMDA-receptor-mediated synaptic transmission where extracellular
    magnesium ions block the receptor channel pore at hyperpolarized potentials.
    The output current is:

    .. math::

       I_{\mathrm{syn}}(t) = g_{\mathrm{syn}}(t) \, (E - V(t)) \,
       g_{\infty}(V, [{Mg}^{2+}]_o)

    where the fraction of unblocked channels is:

    .. math::

       g_{\infty}(V, [{Mg}^{2+}]_o) = \left(
       1 + \frac{[{Mg}^{2+}]_o}{\beta} \,
       e^{-\alpha \, (V - V_{\mathrm{offset}})}
       \right)^{-1}

    Here :math:`[{Mg}^{2+}]_o` is the extracellular magnesium concentration,
    :math:`\alpha` and :math:`\beta` are kinetic constants, and
    :math:`V_{\mathrm{offset}}` is an optional voltage offset.

    Parameters
    ----------
    E : ArrayLike, default=0.
        Reversal potential of the NMDA synapse (mV).
    cc_Mg : ArrayLike, default=1.2
        Extracellular magnesium concentration (mM).
    alpha : ArrayLike, default=0.062
        Voltage sensitivity of the Mg2+ block (/mV).
    beta : ArrayLike, default=3.57
        Mg2+ unbinding constant (mM).
    V_offset : ArrayLike, default=0.
        Voltage offset applied before computing the block factor (mV).

    See Also
    --------
    COBA : Conductance-based output without voltage-dependent block.
    BioNMDA : Biophysical NMDA receptor synapse dynamics.

    Notes
    -----
    - At resting potential (~-65 mV), the Mg2+ block is nearly complete and
      the NMDA conductance contributes little current. As the membrane
      depolarizes (e.g., via AMPA input), the block is progressively relieved,
      creating a voltage-dependent coincidence detection mechanism [1]_.
    - The default parameters (``alpha=0.062``, ``beta=3.57``,
      ``cc_Mg=1.2``) correspond to the widely used fit from
      Jahr & Stevens (1990) [2]_.
    - This module is typically paired with :class:`BioNMDA` or a slow
      exponential synapse model to capture full NMDA receptor dynamics.

    References
    ----------
    .. [1] Mayer, M. L., Westbrook, G. L., & Guthrie, P. B. (1984).
           Voltage-dependent block by Mg2+ of NMDA responses in spinal cord
           neurones. Nature, 309(5965), 261-263.
    .. [2] Jahr, C. E., & Stevens, C. F. (1990). Voltage dependence of
           NMDA-activated macroscopic conductances predicted by single-channel
           kinetics. Journal of Neuroscience, 10(9), 3178-3182.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy
        >>> import brainunit as u
        >>> # Standard NMDA Mg2+ block
        >>> mg_block = brainpy.state.MgBlock(E=0. * u.mV, cc_Mg=1.2)
        >>> # Reduced Mg2+ concentration (e.g., Mg-free solution)
        >>> mg_free = brainpy.state.MgBlock(E=0. * u.mV, cc_Mg=0.0)
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        E: brainstate.typing.ArrayLike = 0.,
        cc_Mg: brainstate.typing.ArrayLike = 1.2,
        alpha: brainstate.typing.ArrayLike = 0.062,
        beta: brainstate.typing.ArrayLike = 3.57,
        V_offset: brainstate.typing.ArrayLike = 0.,
    ):
        super().__init__()

        self.E = E
        self.V_offset = V_offset
        self.cc_Mg = cc_Mg
        self.alpha = alpha
        self.beta = beta

    def update(self, conductance, potential):
        norm = (1 + self.cc_Mg / self.beta * u.math.exp(self.alpha * (self.V_offset - potential)))
        return conductance * (self.E - potential) / norm
