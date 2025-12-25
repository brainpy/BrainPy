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
from functools import wraps
from typing import Dict

import brainstate
import jax.tree

from .base import BrainPyObject, ArrayCollector

__all__ = [
    'infer_dyn_vars',
    'get_brainpy_object',
]


def infer_dyn_vars(target):
    if isinstance(target, BrainPyObject):
        dyn_vars = target.vars().unique()
    elif hasattr(target, '__self__') and isinstance(target.__self__, BrainPyObject):
        dyn_vars = target.__self__.vars().unique()
    else:
        dyn_vars = ArrayCollector()
    return dyn_vars


def get_brainpy_object(target) -> Dict[str, BrainPyObject]:
    if isinstance(target, BrainPyObject):
        return {target.name: target}
    elif hasattr(target, '__self__') and isinstance(target.__self__, BrainPyObject):
        target = target.__self__
        return {target.name: target}
    else:
        return dict()


def _remove_state(x):
    if isinstance(x, brainstate.State):
        return x.value
    return x


def warp_to_no_state_input_output(fn):
    """A decorator to warp a function to a no-state input-output function.

    The decorated function should have the following signature:

        def fn(input1, input2, ..., state1, state2, ...):
            ...

    The decorated function will be transformed to:

        def fn(inputs, states):
            ...

    where `inputs` is a list of all input arguments, and `states` is a list of all state arguments.

    Args:
        fn: The function to be decorated.
    """

    if isinstance(fn, brainstate.typing.Missing):
        return fn

    @wraps(fn)
    def wrapper(*args, **kwargs):
        args, kwargs = jax.tree.map(_remove_state, (args, kwargs), is_leaf=lambda x: isinstance(x, brainstate.State))
        out = fn(*args, **kwargs)
        out = jax.tree.map(_remove_state, out, is_leaf=lambda x: isinstance(x, brainstate.State))
        return out

    return wrapper
