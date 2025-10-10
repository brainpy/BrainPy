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
from typing import Union, Sequence, Dict, Callable

from .base import FunAsObject, BrainPyObject
from .variables import Variable

__all__ = [
    'Partial',
    'to_object',
    'function',
]


class Partial(FunAsObject):
    def __init__(
        self,
        fun: Callable,
        *args,
        child_objs: Union[Callable, BrainPyObject, Sequence[BrainPyObject], Dict[str, BrainPyObject]] = None,
        dyn_vars: Union[Variable, Sequence[Variable], Dict[str, Variable]] = None,
        **keywords
    ):
        super().__init__(target=fun, child_objs=child_objs, dyn_vars=dyn_vars)

        self.fun = fun
        self.args = args
        self.keywords = keywords

    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.fun(*self.args, *args, **keywords)


def to_object(
    f: Callable = None,
    child_objs: Union[Callable, BrainPyObject, Sequence[BrainPyObject], Dict[str, BrainPyObject]] = None,
    dyn_vars: Union[Variable, Sequence[Variable], Dict[str, Variable]] = None,
    name: str = None
):
    """Transform a Python function to :py:class:`~.BrainPyObject`.

    Parameters::

    f: function, callable
      The python function.
    child_objs: Callable, BrainPyObject, sequence of BrainPyObject, dict of BrainPyObject
      The children objects used in this Python function.
    dyn_vars: Variable, sequence of Variable, dict of Variable
      The `Variable` instance used in the Python function.
    name: str
      The name of the created ``BrainPyObject``.

    Returns::

    func: FunAsObject
      The instance of ``BrainPyObject``.
    """

    if f is None:
        def wrap(func) -> FunAsObject:
            return FunAsObject(target=func, child_objs=child_objs, dyn_vars=dyn_vars, name=name)

        return wrap

    else:
        if child_objs is None:
            raise ValueError(f'"child_objs" cannot be None when "f" is provided.')
        return FunAsObject(target=f, child_objs=child_objs, dyn_vars=dyn_vars, name=name)


def function(
    f: Callable = None,
    nodes: Union[Callable, BrainPyObject, Sequence[BrainPyObject], Dict[str, BrainPyObject]] = None,
    dyn_vars: Union[Variable, Sequence[Variable], Dict[str, Variable]] = None,
    name: str = None
):
    """Transform a Python function into a :py:class:`~.BrainPyObject`.

    .. deprecated:: 2.3.0
       Using :py:func:`~.to_object` instead.

    Parameters::

    f: function, callable
      The python function.
    nodes: Callable, BrainPyObject, sequence of BrainPyObject, dict of BrainPyObject
      The children objects used in this Python function.
    dyn_vars: Variable, sequence of Variable, dict of Variable
      The `Variable` instance used in the Python function.
    name: str
      The name of the created ``BrainPyObject``.

    Returns::

    func: FunAsObject
      The instance of ``BrainPyObject``.
    """
    warnings.warn('Using `brainpy.math.to_object()` instead. Will be removed after version 2.4.0.',
                  UserWarning)
    return to_object(f, nodes, dyn_vars, name)
