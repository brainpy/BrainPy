# -*- coding: utf-8 -*-

import functools
import inspect
import types

import numba as nb
import numpy as np

from .codes import deindent
from .codes import get_func_source
from .. import backend
from .. import profile
from ..integration.integrator import Integrator

if hasattr(nb, 'dispatcher'):
    from numba.dispatcher import Dispatcher
else:
    from numba.core.dispatcher import Dispatcher



__all__ = [
    'jit',
    'func_copy',
    'numba_func',
    'get_func_name',
    'get_func_scope',
    'find_integrators',
]


def get_func_name(func, replace=False):
    func_name = func.__name__
    if replace:
        func_name = func_name.replace('_npbrain_delayed_', '')
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
    if not isinstance(func, Dispatcher):
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
    if func == backend.func_by_name(func.__name__):
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
            if v != np.func_by_name(v.__name__) and (not isinstance(v, Dispatcher)):
                code_scope[k] = numba_func(v, params)
                modified = True
    # check scope changed parameters
    for p, v in params.items():
        if p in code_scope:
            code_scope[p] = v
            modified = True

    if modified:
        func_code = deindent(get_func_source(func))
        exec(compile(func_code, '', "exec"), code_scope)
        return jit(code_scope[func.__name__])
    else:
        return jit(func)


def _update_scope(k, v, scope):
    if type(v).__name__ in ['module', 'function']:
        return
    if isinstance(v, Integrator):
        return
    if k in scope:
        if v != scope[k]:
            raise ValueError(f'Find scope variable {k} have different values: \n'
                             f'{k} = {v} and {k} = {scope[k]}. \n'
                             f'This maybe cause a grievous mistake in the future. Please change!')
    scope[k] = v


def get_func_scope(func, include_dispatcher=False):
    """Get function scope variables.

    Parameters
    ----------
    func : callable, Integrator
    include_dispatcher

    Returns
    -------

    """
    # get function scope
    if isinstance(func, Integrator):
        func_name = func.py_func_name
        variables = inspect.getclosurevars(func.diff_eq.func)
    elif type(func).__name__ == 'function':
        func_name = get_func_name(func, replace=True)
        variables = inspect.getclosurevars(func)
    else:
        if type(func).__name__ == 'ufunc':
            return {}
        raise ValueError(f'Unknown type: {type(func)}')
    scope = dict(variables.nonlocals)
    scope.update(variables.globals)

    for k, v in list(scope.items()):
        # get the scope of the function item
        if callable(v):
            if isinstance(v, Dispatcher):
                if include_dispatcher:
                    for k2, v2 in get_func_scope(v.py_func).items():
                        try:
                            _update_scope(k2, v2, scope)
                        except ValueError:
                            raise ValueError(f'Definition error in function "{func_name}".')
            else:
                for k2, v2 in get_func_scope(v).items():
                    try:
                        _update_scope(k2, v2, scope)
                    except ValueError:
                        raise ValueError(f'Definition error in function "{func_name}".')

    for k in list(scope.keys()):
        v = scope[k]
        if type(v).__name__ in ['module', 'function']:
            scope.pop(k)
        if isinstance(v, Integrator):
            scope.pop(k)

    return scope


def find_integrators(func):
    """Find integrators in a given function.

    Parameters
    ----------
    func : callable
        The function.

    Returns
    -------
    integrators : list
        A list of integrators.
    """
    if not callable(func) or type(func).__name__ != 'function':
        return []

    ints = []
    variables = inspect.getclosurevars(func)
    scope = dict(variables.nonlocals)
    scope.update(variables.globals)
    for val in scope.values():
        if isinstance(val, Integrator):
            ints.append(val)
        elif callable(val):
            ints.extend(find_integrators(val))
    return ints

