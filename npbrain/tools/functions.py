# -*- coding: utf-8 -*-

import functools
import inspect
import types

from .codes import deindent
from .. import numpy as np
from .. import profile

try:
    import numba as nb
    from numba.core.dispatcher import Dispatcher
except ImportError as e:
    nb = None
    Dispatcher = None

__all__ = [
    'jit',
    'func_copy',
    'numba_func',
    'get_func_name',
    'get_func_scope',
]


def get_func_name(func, replace=False):
    func_name = func.__name__
    if replace:
        func_name = func_name.replace('_npbrain_delay_push_', '')
        func_name = func_name.replace('_npbrain_delay_pull_', '')
    return func_name


def jit(func=None):
    """JIT user defined functions.

    Parameters
    ----------
    func : callable, a_list, str
        The function to be jit.

    Returns
    -------
    jit_func : callable
        function.
    """
    if nb is None:
        raise ImportError('Please install numba.')
    if not isinstance(func, nb.core.dispatcher.Dispatcher):
        if not callable(func):
            raise ValueError(f'"func" must be a callable function, but got "{type(func)}".')
        op = profile.get_numba_profile()
        func = nb.jit(func, **op)
    return func


def func_copy(f):
    """Make a deepcopy of a python function.

    This method is adopted from http://stackoverflow.com/a/6528148/190597 (Glenn Maynard).

    Parameters
    ----------
    f : callable
        Function to copy.

    Returns
    -------
    g : callable
        Copied function.
    """
    g = types.FunctionType(code=f.__code__,
                           globals=f.__globals__,
                           name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def numba_func(func, params={}):
    if func == np.func_by_name(func.__name__):
        return func
    if isinstance(func, Dispatcher):
        return func

    vars = inspect.getclosurevars(func)
    code_scope = dict(vars.nonlocals)
    code_scope.update(vars.globals)

    modified = False
    # check scope variables
    for k, v in code_scope.items():
        # function
        if callable(v):
            code_scope[k] = numba_func(v, params)
            modified = True
    # check scope changed parameters
    for p, v in params.items():
        if p in code_scope:
            code_scope[p] = v
            modified = True

    if modified:
        func_code = deindent(inspect.getsource(func))
        exec(compile(func_code, '', "exec"), code_scope)
        return jit(code_scope[func.__name__])
    else:
        return jit(func)


def get_func_scope(func):
    vars = inspect.getclosurevars(func)
    scope = dict(vars.nonlocals)
    scope.update(vars.globals)

    for k, v in list(scope.items()):
        if callable(v) and (Dispatcher is not None and not isinstance(v, Dispatcher)):
            v_scope = get_func_scope(v)
            scope.update(v_scope)

    return scope
