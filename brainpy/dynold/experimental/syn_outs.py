# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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
from typing import Union

from brainpy.dynold.experimental.base import SynOutNS
from brainpy.math import exp
from brainpy.types import ArrayType

__all__ = [
    'COBA',
    'CUBA',
    'MgBlock',
]


class COBA(SynOutNS):
    r"""Conductance-based synaptic output.

    Given the synaptic conductance, the model output the post-synaptic current with

    .. math::

       I_{syn}(t) = g_{\mathrm{syn}}(t) (E - V(t))

    Parameters::

    E: float, ArrayType, ndarray
      The reversal potential.
    name: str
      The model name.

    See Also::

    CUBA
    """

    def __init__(self, E: Union[float, ArrayType] = 0., name: str = None, ):
        super().__init__(name=name)
        self.E = E

    def update(self, conductance, potential):
        return conductance * (self.E - potential)


class CUBA(SynOutNS):
    r"""Current-based synaptic output.

    Given the conductance, this model outputs the post-synaptic current with a identity function:

    .. math::

       I_{\mathrm{syn}}(t) = g_{\mathrm{syn}}(t)

    Parameters::

    name: str
      The model name.


    See Also::

    COBA
    """

    def __init__(self, name: str = None, ):
        super().__init__(name=name)

    def update(self, conductance, potential=None):
        return conductance


class MgBlock(SynOutNS):
    r"""Synaptic output based on Magnesium blocking.

    Given the synaptic conductance, the model output the post-synaptic current with

    .. math::

       I_{syn}(t) = g_{\mathrm{syn}}(t) (E - V(t)) g_{\infty}(V,[{Mg}^{2+}]_{o})

    where The fraction of channels :math:`g_{\infty}` that are not blocked by magnesium can be fitted to

    .. math::

       g_{\infty}(V,[{Mg}^{2+}]_{o}) = (1+{e}^{-\alpha V} \frac{[{Mg}^{2+}]_{o}} {\beta})^{-1}

    Here :math:`[{Mg}^{2+}]_{o}` is the extracellular magnesium concentration.

    Parameters::

    E: float, ArrayType
      The reversal potential for the synaptic current. [mV]
    alpha: float, ArrayType
      Binding constant. Default 0.062
    beta: float, ArrayType
      Unbinding constant. Default 3.57
    cc_Mg: float, ArrayType
      Concentration of Magnesium ion. Default 1.2 [mM].
    name: str
      The model name.
    """

    def __init__(
        self,
        E: Union[float, ArrayType] = 0.,
        cc_Mg: Union[float, ArrayType] = 1.2,
        alpha: Union[float, ArrayType] = 0.062,
        beta: Union[float, ArrayType] = 3.57,
        name: str = None,
    ):
        super().__init__(name=name)
        self.E = E
        self.cc_Mg = cc_Mg
        self.alpha = alpha
        self.beta = beta

    def update(self, conductance, potential):
        return conductance * (self.E - potential) / (1 + self.cc_Mg / self.beta * exp(-self.alpha * potential))
