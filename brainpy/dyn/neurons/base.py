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
from typing import Sequence, Union, Callable, Any, Optional

import brainpy.math as bm
from brainpy.check import is_callable
from brainpy.dyn._docs import pneu_doc, dpneu_doc
from brainpy.dyn.base import NeuDyn

__all__ = ['GradNeuDyn']


class GradNeuDyn(NeuDyn):
    """Differentiable and Parallelizable Neuron Group.

    Args:
      {pneu}
      {dpneu}
    """

    supported_modes = (bm.TrainingMode, bm.NonBatchingMode)

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        sharding: Any = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        method: str = 'exp_auto',
        scaling: Optional[bm.Scaling] = None,

        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: str = 'soft',
        detach_spk: bool = False,
    ):
        super().__init__(size=size,
                         mode=mode,
                         keep_size=keep_size,
                         name=name,
                         sharding=sharding,
                         method=method)

        self.spk_reset = spk_reset
        self.spk_fun = is_callable(spk_fun)
        self.detach_spk = detach_spk
        self._spk_dtype = spk_dtype
        if scaling is None:
            self.scaling = bm.get_membrane_scaling()
        else:
            self.scaling = scaling

    @property
    def spk_dtype(self):
        if self._spk_dtype is None:
            return bm.float_ if isinstance(self.mode, bm.TrainingMode) else bm.bool_
        else:
            return self._spk_dtype

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


GradNeuDyn.__doc__ = GradNeuDyn.__doc__.format(pneu=pneu_doc, dpneu=dpneu_doc)
