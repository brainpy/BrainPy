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
from typing import Union, Callable, Optional

import jax.numpy as jnp

import brainpy.math as bm
from brainpy.dynold.synapses.base import _SynOut
from brainpy.initialize import parameter, Initializer
from brainpy.types import ArrayType

__all__ = [
    'MgBlock',
]


class MgBlock(_SynOut):
    r"""Synaptic output based on Magnesium blocking.

    Given the synaptic conductance, the model output the post-synaptic current with

    .. math::

       I_{syn}(t) = g_{\mathrm{syn}}(t) (E - V(t)) g_{\infty}(V,[{Mg}^{2+}]_{o})

    where The fraction of channels :math:`g_{\infty}` that are not blocked by magnesium can be fitted to

    .. math::

       g_{\infty}(V,[{Mg}^{2+}]_{o}) = (1+{e}^{-\alpha V} \frac{[{Mg}^{2+}]_{o}} {\beta})^{-1}

    Here :math:`[{Mg}^{2+}]_{o}` is the extracellular magnesium concentration.

    Parameters::

    E: float, ArrayType, callable, Initializer
      The reversal potential for the synaptic current. [mV]
    alpha: float, ArrayType
      Binding constant. Default 0.062
    beta: float, ArrayType, callable, Initializer
      Unbinding constant. Default 3.57
    cc_Mg: float, ArrayType, callable, Initializer
      Concentration of Magnesium ion. Default 1.2 [mM].
    name: str
      The model name.
    """

    def __init__(
        self,
        E: Union[float, ArrayType, Callable, Initializer] = 0.,
        cc_Mg: Union[float, ArrayType, Callable, Initializer] = 1.2,
        alpha: Union[float, ArrayType, Callable, Initializer] = 0.062,
        beta: Union[float, ArrayType, Callable, Initializer] = 3.57,
        target_var: Optional[Union[str, bm.Variable]] = 'input',
        membrane_var: Union[str, bm.Variable] = 'V',
        name: str = None,
    ):
        super().__init__(name=name, target_var=target_var)
        self._E = E
        self._cc_Mg = cc_Mg
        self._alpha = alpha
        self._beta = beta
        self._target_var = target_var
        self._membrane_var = membrane_var

    def register_master(self, master):
        super().register_master(master)

        self.E = parameter(self._E, self.master.post.num, allow_none=False)
        self.cc_Mg = parameter(self._cc_Mg, self.master.post.num, allow_none=False)
        self.alpha = parameter(self._alpha, self.master.post.num, allow_none=False)
        self.beta = parameter(self._beta, self.master.post.num, allow_none=False)
        if isinstance(self._membrane_var, str):
            if not hasattr(self.master.post, self._membrane_var):
                raise KeyError(f'Post-synaptic group does not have membrane variable: {self._membrane_var}')
            self.membrane_var = getattr(self.master.post, self._membrane_var)
        elif isinstance(self._membrane_var, bm.Variable):
            self.membrane_var = self._membrane_var
        else:
            raise TypeError('"membrane_var" must be instance of string or Variable. '
                            f'But we got {type(self._membrane_var)}')

    def filter(self, g):
        V = self.membrane_var.value
        I = g * (self.E - V) / (1 + self.cc_Mg / self.beta * jnp.exp(-self.alpha * V))
        return super(MgBlock, self).filter(I)

    def clone(self):
        return MgBlock(E=self._E,
                       cc_Mg=self._cc_Mg,
                       alpha=self._alpha,
                       beta=self._beta,
                       target_var=self._target_var,
                       membrane_var=self._membrane_var)
