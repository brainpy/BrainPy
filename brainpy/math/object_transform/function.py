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
    """A picklable, object-aware partial application of a function.

    ``Partial`` behaves like :py:func:`functools.partial`: it binds positional and
    keyword arguments to ``fun`` so that the remaining arguments can be supplied
    later when the instance is called. Unlike :py:func:`functools.partial`, it is a
    :py:class:`~.BrainPyObject`, so any :py:class:`~.Variable` instances and child
    :py:class:`~.BrainPyObject` objects used by ``fun`` are registered and tracked.

    Parameters
    ----------
    fun : callable
        The function to be partially applied.
    *args : Any
        Positional arguments bound ahead of the call-time positional arguments.
    child_objs : callable, BrainPyObject, sequence of BrainPyObject, dict of BrainPyObject, optional
        The children objects used in ``fun``.
    dyn_vars : Variable, sequence of Variable, dict of Variable, optional
        The :py:class:`~.Variable` instances used in ``fun``.
    **keywords : Any
        Keyword arguments bound to ``fun``. Keywords supplied at call time take
        precedence over those bound here.

    See Also
    --------
    to_object : Transform a Python function into a :py:class:`~.BrainPyObject`.

    Examples
    --------
    .. code-block:: python

        >>> import brainpy.math as bm
        >>> add = bm.Partial(lambda x, y: x + y, 1)
        >>> add(2)
        3
    """

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
    warnings.warn('`brainpy.math.function()` is deprecated; use `brainpy.math.to_object()` instead. '
                  'It will be removed in a future release.',
                  DeprecationWarning)
    return to_object(f, nodes, dyn_vars, name)
