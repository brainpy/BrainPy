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
import inspect
import itertools
from functools import partial
from typing import Dict, Callable, Sequence

import jax

from brainpy._errors import UnsupportedError
from brainpy.version2 import check
from brainpy.version2.math.ndarray import Array as Array

__all__ = [
    'get_default',
    'make_return',
    'vjp_custom',
]


def get_default(x, default):
    if x is None:
        return default, False
    else:
        return x, True


def make_return(r, *args):
    if isinstance(r, (tuple, list)):
        r = tuple(r)
    else:
        r = [r]
    for a in args:
        if a:
            r += [None]
    return tuple(r)


def _get_args(f):
    reduced_args = []
    for name, par in inspect.signature(f).parameters.items():
        if par.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            reduced_args.append(par.name)

        elif par.kind is inspect.Parameter.VAR_POSITIONAL:
            reduced_args.append(f'*{par.name}')

        elif par.kind is inspect.Parameter.KEYWORD_ONLY:
            raise UnsupportedError()
        elif par.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise UnsupportedError()
        elif par.kind is inspect.Parameter.VAR_KEYWORD:  # TODO
            raise UnsupportedError()
        else:
            raise UnsupportedError()
    return reduced_args


class VJPCustom(object):
    def __init__(self,
                 func: Callable,
                 args: Sequence[str],
                 defaults: Dict = None,
                 statics: Dict = None, ):
        if statics is None: statics = dict()
        if defaults is None: defaults = dict()
        assert isinstance(statics, dict)
        assert isinstance(defaults, dict)
        assert callable(func)
        check.is_sequence(args, elem_type=str)

        self.n_args = len(args)
        self.func = func
        self.args = args
        self.defaults = tuple(defaults.items())
        self.statics = statics
        self.all_args = _get_args(func)

        for k in statics:
            if k not in defaults:
                raise KeyError(f'{k} defined as "static_args" should provide its default value in "defaults"')
        self._cached = {}
        if len(statics):
            static_vals = list(statics.values())
            products = list(itertools.product(*static_vals, repeat=1))
            for args in products:
                string = self._str_static_arg(dict(zip(self.statics.keys(), args)))
                self._cached[string] = jax.custom_gradient(partial(self.func,
                                                                   **dict(zip(self.statics.keys(), args))))
        else:
            self._cached[''] = jax.custom_gradient(self.func)

    def _str_static_arg(self, args: Dict):
        r = []
        for k in self.statics:
            r.append(f'{k}={args[k]}')
        return '-'.join(r)

    def __call__(self, *args, **kwargs):
        args = list(args)
        kwargs = dict(kwargs)
        for k in self.args[len(args):]:
            if k not in kwargs:
                raise ValueError(f'Must provide {k} for function {self.func}')
            args.append(kwargs.pop(k))
        for k, v in self.defaults[len(args) - self.n_args:]:
            if k not in kwargs:
                args.append(v)
            else:
                args.append(kwargs.pop(k))
        if len(kwargs):
            raise KeyError(f'Unknown arguments {kwargs} for function {self.func}')
        dynamics = []
        statics = dict()
        for k, v in zip(self.all_args, args):
            if isinstance(v, Array): v = v.value
            if k in self.statics:
                statics[k] = v
            else:
                dynamics.append(v)
        return self._cached[self._str_static_arg(statics)](*dynamics)


def vjp_custom(args: Sequence[str], defaults: Dict, statics: Dict = None):
    """Generalize a customized gradient function as a general Python function.
    """

    def wrapper(fun):
        obj = VJPCustom(fun, args, defaults, statics)
        obj.__doc__ = fun.__doc__
        return obj

    return wrapper
