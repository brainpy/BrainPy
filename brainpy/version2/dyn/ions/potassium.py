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

import brainpy.version2.math as bm
from brainpy.version2.dyn.base import IonChaDyn
from brainpy.version2.initialize import Initializer
from brainpy.version2.types import Shape, ArrayType
from .base import Ion

__all__ = [
    'Potassium',
    'PotassiumFixed',
]


class Potassium(Ion):
    """Base class for modeling Potassium ion."""
    pass


class PotassiumFixed(Potassium):
    """Fixed Sodium dynamics.

    This calcium model has no dynamics. It holds fixed reversal
    potential :math:`E` and concentration :math:`C`.
    """

    def __init__(
        self,
        size: Shape,
        keep_size: bool = False,
        E: Union[float, ArrayType, Initializer, Callable] = -950.,
        C: Union[float, ArrayType, Initializer, Callable] = 0.0400811,
        method: str = 'exp_auto',
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
        **channels
    ):
        super().__init__(size,
                         keep_size=keep_size,
                         method=method,
                         name=name,
                         mode=mode,
                         **channels)
        self.E = self.init_param(E, self.varshape)
        self.C = self.init_param(C, self.varshape)

    def reset_state(self, V, C=None, E=None, batch_size=None):
        C = self.C if C is None else C
        E = self.E if E is None else E
        nodes = self.nodes(level=1, include_self=False).unique().subset(IonChaDyn).values()
        self.check_hierarchies(type(self), *tuple(nodes))
        for node in nodes:
            node.reset_state(V, C, E, batch_size)
