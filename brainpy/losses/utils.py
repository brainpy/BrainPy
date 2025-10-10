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
import jax
from jax.tree_util import tree_flatten

import brainpy.math as bm
from brainpy._errors import UnsupportedError

_reduction_error = 'Only support reduction of "mean", "sum" and "none", but we got "%s".'


def _is_leaf(x):
    return isinstance(x, bm.Array)


def _reduce(outputs, reduction, axis=None):
    if reduction == 'mean':
        return outputs.mean(axis)
    elif reduction == 'sum':
        return outputs.sum(axis)
    elif reduction == 'none':
        return outputs
    else:
        raise UnsupportedError(_reduction_error % reduction)


def _multi_return(r):
    if isinstance(r, jax.Array):
        return r
    elif isinstance(r, bm.Array):
        return r.value
    else:
        leaves = tree_flatten(r)[0]
        r = leaves[0]
        for leaf in leaves[1:]:
            r += leaf
        return r
