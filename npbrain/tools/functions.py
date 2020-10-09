# -*- coding: utf-8 -*-

import functools
import inspect
import types

import numpy as onp

from .codes import deindent

try:
    import numba as nb
    from numba.core.dispatcher import Dispatcher
except ImportError as e:
    nb = None

from npbrain import _numpy as np
from npbrain import profile

__all__ = [
    # function helpers
    'jit_function',
    'autojit',
    'func_copy',
    'is_lambda_function',
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


def autojit(signature_or_func=None):
    """Format user defined functions.

    Parameters
    ----------
    signature_or_func : callable, a_list, str

    Returns
    -------
    callable
        function.
    """
    if callable(signature_or_func):  # function
        if profile.is_numba_bk():
            if nb is None:
                raise ImportError('Please install numba.')
            if not isinstance(signature_or_func, nb.core.dispatcher.Dispatcher):
                op = profile.get_numba_profile()
                signature_or_func = nb.jit(signature_or_func, **op)
        return signature_or_func

    else:  # signature

        if signature_or_func is None:
            pass
        else:
            if isinstance(signature_or_func, str):
                signature_or_func = [signature_or_func]
            else:
                assert isinstance(signature_or_func, (list, tuple))
                signature_or_func = list(signature_or_func)

        def wrapper(f):
            if profile.is_numba_bk():
                if nb is None:
                    raise ImportError('Please install numba.')
                if not isinstance(f, nb.core.dispatcher.Dispatcher):
                    op = profile.get_numba_profile()
                    if profile.define_signature:
                        j = nb.jit(signature_or_func, **op)
                    else:
                        j = nb.jit(**op)
                    f = j(f)
            return f

        return wrapper


def is_lambda_function(func):
    """Check whether the function is a ``lambda`` function. Comes from
    https://stackoverflow.com/questions/23852423/how-to-check-that-variable-is-a-lambda-function

    Parameters
    ----------
    func : callable function
        The function.

    Returns
    -------
    bool
        True of False.
    """
    return isinstance(func, types.LambdaType) and func.__name__ == "<lambda>"


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
        return autojit(code_scope[func.__name__])
    else:
        return autojit(func)


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
