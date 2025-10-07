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

SUPPORT_NUMBA = numba is not None


def numba_jit(f=None, **kwargs):
    if f is None:
        return lambda f: (f if (numba is None) else numba.njit(f, **kwargs))
    else:
        if numba is None:
            return f
        else:
            return numba.njit(f)


@numba_jit
def _seed(seed):
    np.random.seed(seed)


def numba_seed(seed):
    if numba is not None and seed is not None:
        _seed(seed)


numba_range = numba.prange if SUPPORT_NUMBA else range
