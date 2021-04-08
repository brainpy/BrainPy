# -*- coding: utf-8 -*-

import inspect
from copy import deepcopy

from brainpy import backend
from brainpy import errors

__all__ = [
    'get_args',
]


def get_args(f):
    """Get the function arguments.

    >>> def f1(a, b, t, *args, c=1): pass
    >>> get_args(f1)
    (['a', 'b'], ['t', '*args', 'c'], ['a', 'b', 't', '*args', 'c=1'])

    >>> def f2(a, b, *args, c=1, **kwargs): pass
    >>> get_args(f2)
    ValueError: Don not support dict of keyword arguments: **kwargs

    >>> def f3(a, b, t, c=1, d=2): pass
    >>> get_args(f4)
    (['a', 'b'], ['t', 'c', 'd'], ['a', 'b', 't', 'c=1', 'd=2'])

    >>> def f4(a, b, t, *args): pass
    >>> get_args(f4)
    (['a', 'b'], ['t', '*args'], ['a', 'b', 't', '*args'])

    >>> scope = {}
    >>> exec(compile('def f5(a, b, t, *args): pass', '', 'exec'), scope)
    >>> get_args(scope['f5'])
    (['a', 'b'], ['t', '*args'], ['a', 'b', 't', '*args'])

    Parameters
    ----------
    f : callable
        The function.

    Returns
    -------
    args : tuple
        The variable names, the other arguments, and the original args.
    """

    # 1. get the function arguments
    reduced_args = []
    original_args = []

    for name, par in inspect.signature(f).parameters.items():
        if par.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            reduced_args.append(par.name)

        elif par.kind is inspect.Parameter.VAR_POSITIONAL:
            reduced_args.append(f'*{par.name}')

        elif par.kind is inspect.Parameter.KEYWORD_ONLY:
            reduced_args.append(par.name)

        elif par.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise errors.DiffEqError('Don not support positional only parameters, e.g., /')
        elif par.kind is inspect.Parameter.VAR_KEYWORD:
            raise errors.DiffEqError(f'Don not support dict of keyword arguments: {str(par)}')
        else:
            raise errors.DiffEqError(f'Unknown argument type: {par.kind}')

        original_args.append(str(par))

    # 2. analyze the function arguments
    #   2.1 class keywords
    class_kw = []
    if reduced_args[0] in backend.CLASS_KEYWORDS:
        class_kw.append(reduced_args[0])
        reduced_args = reduced_args[1:]
    for a in reduced_args:
        if a in backend.CLASS_KEYWORDS:
            raise errors.DiffEqError(f'Class keywords "{a}" must be defined '
                                     f'as the first argument.')
    #  2.2 variable names
    var_names = []
    for a in reduced_args:
        if a == 't':
            break
        var_names.append(a)
    else:
        raise ValueError('Do not find time variable "t".')
    other_args = reduced_args[len(var_names):]
    return class_kw, var_names, other_args, original_args
