# -*- coding: utf-8 -*-

import inspect
from pprint import pprint

from numba.core.dispatcher import Dispatcher

from .. import backend
from .. import errors
from .. import integration
from .. import tools

__all__ = [
    'show_code_str',
    'show_code_scope',
    'find_integrators',
    'get_func_scope',
    'check_slice',
]


def check_slice(start, end, length):
    if start >= end:
        raise errors.ModelUseError(f'Illegal start/end values for subgroup, {start}>={end}')
    if start >= length:
        raise errors.ModelUseError(f'Illegal start value for subgroup, {start}>={length}')
    if end > length:
        raise errors.ModelUseError(f'Illegal stop value for subgroup, {end}>{length}')
    if start < 0:
        raise errors.ModelUseError('Indices have to be positive.')


def show_code_str(func_code):
    print(func_code)
    print()


def show_code_scope(code_scope, ignores=()):
    scope = {}
    for k, v in code_scope.items():
        if k in ignores:
            continue
        if k in integration.CONSTANT_MAPPING:
            continue
        if k in integration.FUNCTION_MAPPING:
            continue
        scope[k] = v
    pprint(scope)
    print()


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

    integrals = []
    variables = inspect.getclosurevars(func)
    scope = dict(variables.nonlocals)
    scope.update(variables.globals)
    for val in scope.values():
        if isinstance(val, integration.Integrator):
            integrals.append(val)
        elif callable(val):
            integrals.extend(find_integrators(val))
    return integrals


def _update_scope(k, v, scope):
    if type(v).__name__ in ['module', 'function']:
        return
    if isinstance(v, integration.Integrator):
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
    if isinstance(func, integration.Integrator):
        func_name = func.py_func_name
        variables = inspect.getclosurevars(func.diff_eq.func)
        scope = dict(variables.nonlocals)
        scope.update(variables.globals)
    elif type(func).__name__ == 'function':
        func_name = tools.get_func_name(func, replace=True)
        variables = inspect.getclosurevars(func)
        if func_name.startswith('xoroshiro128p_'):
            return {}
        scope = dict(variables.nonlocals)
        scope.update(variables.globals)
    else:
        if backend.func_in_numpy_or_math(func):
            return {}
        elif isinstance(func, Dispatcher) and include_dispatcher:
            scope = get_func_scope(func.py_func)
        else:
            raise ValueError(f'Unknown type: {type(func)}')

    # update scope
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
        if isinstance(v, integration.Integrator):
            scope.pop(k)

    return scope
