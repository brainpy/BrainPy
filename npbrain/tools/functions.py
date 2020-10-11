# -*- coding: utf-8 -*-

import functools
import inspect
import types

import numpy as onp

from .codes import deindent
from .. import _numpy as np
from .. import profile

try:
    import numba as nb
    from numba.core.dispatcher import Dispatcher
except ImportError as e:
    nb = None



__all__ = [
    # function helpers
    'jit_function',
    'jit',
    'func_copy',
    'numba_func',

    # 'others'
    'is_struct_array',
    'init_struct_array',

]


##############################
# function helpers
##############################


def jit_function(f):
    """Generate ``numba`` JIT functions.

    Parameters
    ----------
    f : callable
        The function.

    Returns
    -------
    callable
        JIT function.
    """
    if nb is None:
        raise ImportError('Please install numba.')
    op = profile.get_numba_profile()
    return nb.jit(f, **op)


def jit(func=None):
    """Format user defined functions.

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
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
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


def is_struct_array(arr):
    if profile.is_numba_bk() or profile.is_numba_bk():
        if isinstance(arr, onp.ndarray) and (arr.dtype.names is not None):
            return True
        else:
            return False

    if profile.is_jax_bk():
        raise NotImplementedError


def init_struct_array(num, variables):
    if isinstance(variables, (list, tuple)):
        variables = {v: np.float_ for v in variables}
    elif isinstance(variables, dict):
        pass
    else:
        raise ValueError(f'Unknown type: {type(variables)}.')

    if profile.is_numba_bk() and profile.is_numpy_bk():
        dtype = np.dtype(list(variables.items()), align=True)
        arr = np.zeros(num, dtype)
        return arr

    if profile.is_jax_bk():
        arr = {k: np.zeros(num, dtype=d) for k, d in variables.items()}
        return arr
