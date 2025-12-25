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
import functools
from typing import Callable

from jax.tree_util import tree_map

from .ndarray import _return, Array


def _as_jax_array_(obj):
    return obj.value if isinstance(obj, Array) else obj


def _is_leaf(a):
    return isinstance(a, Array)


def _compatible_with_brainpy_array(
    fun: Callable,
    module: str = ''
):
    func_to_wrap = fun.__np_wrapped__ if hasattr(fun, '__np_wrapped__') else fun

    @functools.wraps(func_to_wrap)
    def new_fun(*args, **kwargs):
        args = tree_map(_as_jax_array_, args, is_leaf=_is_leaf)
        out = None
        if len(kwargs):
            # compatible with PyTorch syntax
            if 'dim' in kwargs:
                kwargs['axis'] = kwargs.pop('dim')
            if 'keepdim' in kwargs:
                kwargs['keepdims'] = kwargs.pop('keepdim')
            # compatible with TensorFlow syntax
            if 'keep_dims' in kwargs:
                kwargs['keepdims'] = kwargs.pop('keep_dims')
            # compatible with NumPy/PyTorch syntax
            if 'out' in kwargs:
                out = kwargs.pop('out')
                if not isinstance(out, Array):
                    raise TypeError(f'"out" must be an instance of brainpy Array. While we got {type(out)}')
            # format
            kwargs = tree_map(_as_jax_array_, kwargs, is_leaf=_is_leaf)
        r = fun(*args, **kwargs)
        if out is None:
            return tree_map(_return, r)
        else:
            out.value = r

    new_fun.__doc__ = (
        f'Similar to ``jax.numpy.{module + fun.__name__}`` function, '
        f'while it is compatible with brainpy Array/Variable. \n\n'
        f'Note that this function is also compatible with:\n\n'
        f'1. NumPy or PyTorch syntax when receiving ``out`` argument.\n'
        f'2. PyTorch syntax when receiving ``keepdim`` or ``dim`` argument.\n'
        f'3. TensorFlow syntax when receiving ``keep_dims`` argument.'
    )

    return new_fun
