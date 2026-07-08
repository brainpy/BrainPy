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
import warnings
import weakref
from typing import Any, Dict

from brainpy import _errors as errors

__all__ = [
    'clear_name_cache',
]

# Maps a unique name to a *weak* reference of the object that owns it.
#
# Storing a weak reference (instead of the raw ``id(obj)``) has two benefits:
#   1. The registry no longer keeps objects alive, so it does not grow
#      unboundedly as transient objects are created and discarded.
#   2. Once an owning object is garbage-collected, its name is treated as free
#      again. Keying on ``id(obj)`` was unsafe because CPython readily reuses
#      the integer id of a collected object for a brand-new one, which could
#      trigger spurious ``UniqueNameError`` (or mask a genuine collision).
_name2id: Dict[str, weakref.ref] = dict()  # name -> weakref.ref(obj)
_typed_names: Dict[str, int] = {}


def check_name_uniqueness(name, obj):
    """Check the uniqueness of the name for the object type."""
    if not name.isidentifier():
        raise errors.BrainPyError(f'"{name}" isn\'t a valid identifier '
                                  f'according to Python language definition. '
                                  f'Please choose another name.')
    if name in _name2id:
        existing = _name2id[name]()  # dereference the weak ref
        # ``existing is None``  -> the previous owner has been collected, so the
        #                          name is free and can be re-registered.
        # ``existing is obj``   -> the same object re-registering its own name.
        if existing is not None and existing is not obj:
            raise errors.UniqueNameError(
                f'In BrainPy, each object should have a unique name. '
                f'However, we detect that {obj} has a used name "{name}". \n'
                f'If you try to run multiple trials, you may need \n\n'
                f'>>> brainpy.math.clear_name_cache() \n\n'
                f'to clear all cached names. '
            )

    # (Re)register the name with a weak reference to the current owner. A
    # finalizer drops the entry as soon as the object is collected, keeping the
    # registry bounded.
    def _drop(_ref, _name=name):
        if _name2id.get(_name) is _ref:
            del _name2id[_name]

    _name2id[name] = weakref.ref(obj, _drop)


def get_unique_name(type_: str):
    """Get the unique name for the given object type."""
    if type_ not in _typed_names:
        _typed_names[type_] = 0
    name = f'{type_}{_typed_names[type_]}'
    _typed_names[type_] += 1
    return name


def clear_name_cache(ignore_warn=True):
    """Clear the cached names."""
    _name2id.clear()
    _typed_names.clear()
    if not ignore_warn:
        warnings.warn(f'All named models and their ids are cleared.', UserWarning)


_fun2stack: Dict[Any, Any] = dict()


def cache_stack(func, stack):
    _fun2stack[func] = stack


def clear_stack_cache():
    """Clear the cached stack."""
    for k in tuple(_fun2stack.keys()):
        del _fun2stack[k]


def get_stack_cache(func):
    if func in _fun2stack:
        return _fun2stack[func]
    else:
        return None
