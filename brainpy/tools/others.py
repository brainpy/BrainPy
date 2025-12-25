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
import _thread as thread
import collections.abc
import threading
from typing import Optional, Tuple, Callable, Union, Sequence, TypeVar, Any

import numpy as np

__all__ = [
    'one_of',
    'replicate',
    'not_customized',
    'to_size',
    'size2num',
    'timeout',
]


def one_of(default: Any, *choices, names: Sequence[str] = None):
    names = [f'arg{i}' for i in range(len(choices))] if names is None else names
    res = default
    has_chosen = False
    for c in choices:
        if c is not None:
            if has_chosen:
                raise ValueError(f'Provide one of {names}, but we got {list(zip(choices, names))}')
            else:
                has_chosen = True
                res = c
    return res


T = TypeVar('T')


def replicate(
    element: Union[T, Sequence[T]],
    num_replicate: int,
    name: str,
) -> Tuple[T, ...]:
    """Replicates entry in `element` `num_replicate` if needed."""
    if isinstance(element, (str, bytes)) or not isinstance(element, collections.abc.Sequence):
        return (element,) * num_replicate
    elif len(element) == 1:
        return tuple(element * num_replicate)
    elif len(element) == num_replicate:
        return tuple(element)
    else:
        raise TypeError(f"{name} must be a scalar or sequence of length 1 or "
                        f"sequence of length {num_replicate}.")


def not_customized(fun: Callable) -> Callable:
    """Marks the given module method is not implemented.

    Methods wrapped in @not_customized can define submodules directly within the method.

    For instance::

      @not_customized
      init_fb(self):
        ...

      @not_customized
      def feedback(self):
        ...
    """
    fun.not_customized = True
    return fun


def size2num(size):
    if isinstance(size, (int, np.integer)):
        return size
    elif isinstance(size, (tuple, list)):
        a = 1
        for b in size:
            a *= b
        return a
    else:
        raise ValueError(f'Do not support type {type(size)}: {size}')


def to_size(x) -> Optional[Tuple[int]]:
    if isinstance(x, (tuple, list)):
        return tuple(x)
    if isinstance(x, (int, np.integer)):
        return (x,)
    if x is None:
        return x
    raise ValueError(f'Cannot make a size for {x}')


def timeout(s):
    """Add a timeout parameter to a function and return it.

    Parameters::

    s : float
        Time limit in seconds.

    Returns::

    func : callable
        Functional results. Or, raise an error of KeyboardInterrupt.
    """

    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, thread.interrupt_main)
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result

        return inner

    return outer
