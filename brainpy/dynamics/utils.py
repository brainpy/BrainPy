# -*- coding: utf-8 -*-

from numba import njit
import inspect
import _thread as thread
import threading
from .. import tools
from .. import backend
from numba.core.dispatcher import Dispatcher
import numpy as np

__all__ = [
    'stability_analysis',
    'rescale',
    'timeout',
    'jit_func',
]

_SADDLE_NODE = 'saddle-node'
_1D_STABLE_POINT = 'stable-point'
_1D_UNSTABLE_POINT = 'unstable-point'
_2D_CENTER = 'center'
_2D_STABLE_NODE = 'stable-node'
_2D_STABLE_FOCUS = 'stable-focus'
_2D_STABLE_STAR = 'stable-star'
_2D_STABLE_LINE = 'stable-line'
_2D_UNSTABLE_NODE = 'unstable-node'
_2D_UNSTABLE_FOCUS = 'unstable-focus'
_2D_UNSTABLE_STAR = 'star'
_2D_UNSTABLE_LINE = 'unstable-line'

plot_scheme = {
    _1D_STABLE_POINT: {"color": 'tab:red'},
    _2D_STABLE_NODE: {"color": 'tab:red'},

    _1D_UNSTABLE_POINT: {"color": 'tab:olive'},
    _2D_UNSTABLE_NODE: {"color": 'tab:olive'},

    _2D_STABLE_FOCUS: {"color": 'tab:purple'},
    _2D_UNSTABLE_FOCUS: {"color": 'tab:cyan'},

    _SADDLE_NODE: {"color": 'tab:blue'},

    _2D_STABLE_LINE: {'color': 'orangered'},
    _2D_UNSTABLE_LINE: {'color': 'dodgerblue'},
    _2D_CENTER: {'color': 'lime'},
    _2D_UNSTABLE_STAR: {'color': 'green'},
    _2D_STABLE_STAR: {'color': 'orange'},
}


def get_1d_classification():
    return [_SADDLE_NODE, _1D_STABLE_POINT, _1D_UNSTABLE_POINT]


def get_2d_classification():
    return [_SADDLE_NODE, _2D_CENTER, _2D_STABLE_NODE, _2D_STABLE_FOCUS,
            _2D_STABLE_STAR, _2D_STABLE_LINE, _2D_UNSTABLE_NODE,
            _2D_UNSTABLE_FOCUS, _2D_UNSTABLE_STAR, _2D_UNSTABLE_LINE]


def stability_analysis(derivative):
    """Stability analysis for fixed point.

    Parameters
    ----------
    derivative : float, tuple, list, np.ndarray
        The derivative of the f.

    Returns
    -------
    fp_type : str
        The type of the fixed point.
    """
    if np.size(derivative) == 1:
        if derivative == 0:
            return _SADDLE_NODE
        elif derivative > 0:
            return _1D_STABLE_POINT
        else:
            return _1D_UNSTABLE_POINT
    elif np.size(derivative) == 4:
        a = derivative[0][0]
        b = derivative[0][1]
        c = derivative[1][0]
        d = derivative[1][1]

        # trace
        p = a + d
        # det
        q = a * d - b * c
        # parabola
        e = p * p - 4 * q

        # judgement
        if q < 0:
            return _SADDLE_NODE
        elif q == 0:
            if p < 0:
                return _2D_STABLE_LINE
            else:
                return _2D_UNSTABLE_LINE
        else:
            if p == 0:
                return _2D_CENTER
            elif p > 0:
                if e < 0:
                    return _2D_UNSTABLE_FOCUS
                elif e == 0:
                    return _2D_UNSTABLE_STAR
                else:
                    return _2D_UNSTABLE_NODE
            else:
                if e < 0:
                    return _2D_STABLE_FOCUS
                elif e == 0:
                    return _2D_STABLE_STAR
                else:
                    return _2D_STABLE_NODE
    else:
        raise ValueError('Unknown derivatives.')


def rescale(min_max, scale=0.01):
    """Rescale lim."""
    min_, max_ = min_max
    length = max_ - min_
    min_ -= scale * length
    max_ += scale * length
    return min_, max_


def timeout(s):
    """Add a timeout parameter to a function and return it.

    Parameters
    ----------
    s : float
        Time limit in seconds.

    Returns
    -------
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


def jit(func):
    if backend.func_in_numpy_or_math(func):
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
            if (not backend.func_in_numpy_or_math(v)) and (not isinstance(v, Dispatcher)):
                code_scope[k] = jit(v)
                modified = True

    if modified:
        func_code = tools.deindent(tools.get_func_source(func))
        exec(compile(func_code, '', "exec"), code_scope)
        func = code_scope[func.__name__]
        return njit(func)
    else:
        njit(func)


def jit_func(scope, func_code, func_name):
    # get function scope
    func_scope = dict()
    for key, val in scope.items():
        if callable(val):
            if backend.func_in_numpy_or_math(val):
                pass
            elif isinstance(val, Dispatcher):
                pass
            else:
                val = jit(val)
        func_scope[key] = val

    # compile function
    exec(compile(func_code, '', 'exec'), func_scope)
    return njit(func_scope[func_name])
