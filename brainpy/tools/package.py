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
from typing import Any, Callable, Optional

import numpy as np

try:
    import numba
except (ImportError, ModuleNotFoundError):
    numba = None

__all__ = [
    'numba_jit',
    'numba_seed',
    'numba_range',
    'SUPPORT_NUMBA',
]

SUPPORT_NUMBA: bool = numba is not None


def numba_jit(f: Optional[Callable[..., Any]] = None, **kwargs: Any) -> Callable[..., Any]:
    if f is None:
        return lambda f: (f if (numba is None) else numba.njit(f, **kwargs))
    else:
        if numba is None:
            return f
        else:
            jitted: Callable[..., Any] = numba.njit(f)
            return jitted


@numba_jit
def _seed(seed: int) -> None:
    np.random.seed(seed)


def numba_seed(seed: Optional[int]) -> None:
    if numba is not None and seed is not None:
        _seed(seed)


numba_range: Callable[..., Any] = numba.prange if SUPPORT_NUMBA else range
