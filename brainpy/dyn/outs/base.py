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
from typing import Optional

import brainpy.math as bm
from brainpy.dynsys import DynamicalSystem
from brainpy.mixin import ParamDesc, BindCondData

__all__ = [
    'SynOut'
]


class SynOut(DynamicalSystem, ParamDesc, BindCondData):
    """Base class for synaptic outputs.

    :py:class:`~.SynOut` is also subclass of :py:class:`~.ParamDesc` and :pu:class:`~.BindCondData`.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 scaling: Optional[bm.Scaling] = None):
        super().__init__(name=name)
        self._conductance = None
        if scaling is None:
            self.scaling = bm.get_membrane_scaling()
        else:
            self.scaling = scaling

    def __call__(self, *args, **kwargs):
        if self._conductance is None:
            raise ValueError(f'Please first pack conductance data at the current step using '
                             f'".{BindCondData.bind_cond.__name__}(data)". {self}')
        ret = self.update(self._conductance, *args, **kwargs)
        return ret

    def reset_state(self, *args, **kwargs):
        pass

    def offset_scaling(self, x, bias=None, scale=None):
        s = self.scaling.offset_scaling(x, bias=bias, scale=scale)
        if isinstance(x, bm.Array):
            x.value = s
            return x
        return s

    def std_scaling(self, x, scale=None):
        s = self.scaling.std_scaling(x, scale=scale)
        if isinstance(x, bm.Array):
            x.value = s
            return x
        return s

    def inv_scaling(self, x, scale=None):
        s = self.scaling.inv_scaling(x, scale=scale)
        if isinstance(x, bm.Array):
            x.value = s
            return x
        return s
