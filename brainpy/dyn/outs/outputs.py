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
from typing import Union, Optional, Sequence

import numpy as np

from brainpy import math as bm, initialize as init
from brainpy.types import ArrayType
from .base import SynOut

__all__ = [
    'COBA',
    'CUBA',
    'MgBlock'
]


class COBA(SynOut):
    r"""Conductance-based synaptic output.

    Given the synaptic conductance, the model output the post-synaptic current with

    .. math::

       I_{syn}(t) = g_{\mathrm{syn}}(t) (E - V(t))

    Parameters::

    E: float, ArrayType, ndarray
      The reversal potential.
    sharding: sequence of str
      The axis names for variable for parallelization.
    name: str
      The model name.
    scaling: brainpy.Scaling
      The scaling object.

    See Also::

    CUBA
    """

    def __init__(
        self,
        E: Union[float, ArrayType],
        sharding: Optional[Sequence[str]] = None,
        name: Optional[str] = None,
        scaling: Optional[bm.Scaling] = None,
    ):
        super().__init__(name=name, scaling=scaling)

        self.sharding = sharding
        self.E = self.offset_scaling(init.parameter(E, np.shape(E), sharding=sharding))

    def update(self, conductance, potential):
        return conductance * (self.E - potential)


class CUBA(SynOut):
    r"""Current-based synaptic output.

    Given the conductance, this model outputs the post-synaptic current with a identity function:

    .. math::

       I_{\mathrm{syn}}(t) = g_{\mathrm{syn}}(t)

    Parameters::

    name: str
      The model name.
    scaling: brainpy.Scaling
      The scaling object.

    See Also::

    COBA
    """

    def __init__(
        self,
        name: Optional[str] = None,
        scaling: Optional[bm.Scaling] = None,
    ):
        super().__init__(name=name, scaling=scaling)

    def update(self, conductance, potential=None):
        return conductance


class MgBlock(SynOut):
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
    sharding: sequence of str
      The axis names for variable for parallelization.
    name: str
      The model name.
    """

    def __init__(
        self,
        E: Union[float, ArrayType] = 0.,
        cc_Mg: Union[float, ArrayType] = 1.2,
        alpha: Union[float, ArrayType] = 0.062,
        beta: Union[float, ArrayType] = 3.57,
        V_offset: Union[float, ArrayType] = 0.,
        sharding: Optional[Sequence[str]] = None,
        name: Optional[str] = None,
        scaling: Optional[bm.Scaling] = None,
    ):
        super().__init__(name=name, scaling=scaling)

        self.sharding = sharding
        self.E = self.offset_scaling(init.parameter(E, np.shape(E), sharding=sharding))
        self.V_offset = self.offset_scaling(init.parameter(V_offset, np.shape(V_offset), sharding=sharding))
        self.cc_Mg = init.parameter(cc_Mg, np.shape(cc_Mg), sharding=sharding)
        self.alpha = self.inv_scaling(init.parameter(alpha, np.shape(alpha), sharding=sharding))
        self.beta = init.parameter(beta, np.shape(beta), sharding=sharding)

    def update(self, conductance, potential):
        norm = (1 + self.cc_Mg / self.beta * bm.exp(self.alpha * (self.V_offset - potential)))
        return conductance * (self.E - potential) / norm
