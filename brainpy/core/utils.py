# -*- coding: utf-8 -*-

import inspect
from .. import integration


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

