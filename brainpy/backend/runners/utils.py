# -*- coding: utf-8 -*-


import inspect

from brainpy import errors
from brainpy import profile

__all__ = [
    'get_args'
]


def get_args(f):
    """Get the function arguments.

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
    parameters = inspect.signature(f).parameters

    arguments = []
    for name, par in parameters.items():
        if par.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            arguments.append(par.name)

        elif par.kind is inspect.Parameter.KEYWORD_ONLY:
            arguments.append(par.name)

        elif par.kind is inspect.Parameter.VAR_POSITIONAL:
            raise errors.ModelDefError('Step function do not support positional parameters, e.g., *args')
        elif par.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise errors.ModelDefError('Step function do not support positional only parameters, e.g., /')
        elif par.kind is inspect.Parameter.VAR_KEYWORD:
            raise errors.ModelDefError(f'Step function do not support dict of keyword arguments: {str(par)}')
        else:
            raise errors.ModelDefError(f'Unknown argument type: {par.kind}')

    # 2. check the function arguments
    class_kw = None
    if arguments[0] in profile.CLASS_KEYWORDS:
        class_kw = arguments[0]
        arguments = arguments[1:]
    for a in arguments:
        if a in profile.CLASS_KEYWORDS:
            raise errors.DiffEqError(f'Class keywords "{a}" must be defined '
                                     f'as the first argument.')
    return class_kw, arguments
