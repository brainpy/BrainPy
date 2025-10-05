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

import brainstate
import brainunit as u
import jax.numpy as jnp

__all__ = [
    'SynOut', 'COBA', 'CUBA', 'MgBlock',
]


class SynOut(brainstate.nn.Module, brainstate.mixin.BindCondData):
    """
    Base class for synaptic outputs.

    :py:class:`~.SynOut` is also subclass of :py:class:`~.ParamDesc` and :py:class:`~.BindCondData`.
    """

    __module__ = 'brainstate.nn'

    def __init__(self, ):
        super().__init__()
        self._conductance = None

    def __call__(self, *args, **kwargs):
        if self._conductance is None:
            raise ValueError(
                f'Please first pack conductance data at the current step using '
                f'".{brainstate.mixin.BindCondData.bind_cond.__name__}(data)". {self}'
            )
        ret = self.update(self._conductance, *args, **kwargs)
        return ret

    def update(self, conductance, potential):
        raise NotImplementedError


class COBA(SynOut):
    r"""
    Conductance-based synaptic output.

    Given the synaptic conductance, the model output the post-synaptic current with

    .. math::

       I_{syn}(t) = g_{\mathrm{syn}}(t) (E - V(t))

    Parameters
    ----------
    E: ArrayLike
      The reversal potential.

    See Also
    --------
    CUBA
    """
    __module__ = 'brainpy'

    def __init__(self, E: brainstate.typing.ArrayLike):
        super().__init__()

        self.E = E

    def update(self, conductance, potential):
        return conductance * (self.E - potential)


class CUBA(SynOut):
    r"""Current-based synaptic output.

    Given the conductance, this model outputs the post-synaptic current with a identity function:

    .. math::

       I_{\mathrm{syn}}(t) = g_{\mathrm{syn}}(t)

    Parameters
    ----------
    scale: ArrayLike
      The scaling factor for the conductance. Default 1. [mV]

    See Also
    --------
    COBA
    """
    __module__ = 'brainpy'

    def __init__(self, scale: brainstate.typing.ArrayLike = u.volt):
        super().__init__()
        self.scale = scale

    def update(self, conductance, potential=None):
        return conductance * self.scale


class MgBlock(SynOut):
    r"""Synaptic output based on Magnesium blocking.

    Given the synaptic conductance, the model output the post-synaptic current with

    .. math::

       I_{syn}(t) = g_{\mathrm{syn}}(t) (E - V(t)) g_{\infty}(V,[{Mg}^{2+}]_{o})

    where The fraction of channels :math:`g_{\infty}` that are not blocked by magnesium can be fitted to

    .. math::

       g_{\infty}(V,[{Mg}^{2+}]_{o}) = (1+{e}^{-\alpha V} \frac{[{Mg}^{2+}]_{o}} {\beta})^{-1}

    Here :math:`[{Mg}^{2+}]_{o}` is the extracellular magnesium concentration.

    Parameters
    ----------
    E: ArrayLike
      The reversal potential for the synaptic current. [mV]
    alpha: ArrayLike
      Binding constant. Default 0.062
    beta: ArrayLike
      Unbinding constant. Default 3.57
    cc_Mg: ArrayLike
      Concentration of Magnesium ion. Default 1.2 [mM].
    V_offset: ArrayLike
      The offset potential. Default 0. [mV]
    """
    __module__ = 'brainpy'

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
        norm = (1 + self.cc_Mg / self.beta * jnp.exp(self.alpha * (self.V_offset - potential)))
        return conductance * (self.E - potential) / norm
