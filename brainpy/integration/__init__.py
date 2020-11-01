# -*- coding: utf-8 -*-

from . import diff_equation
from . import integrator
from . import sympy_tools
from . import methods
from .diff_equation import *
from .integrator import *
from .. import profile

_SUPPORT_METHODS = [
    'euler',
    'midpoint',
    'heun',
    'rk2',
    'rk3',
    'rk4',
    'rk4_alternative',
    'exponential',
    'milstein',
    'milstein_ito',
    'milstein_stra',
]


def integrate(func=None, method=None):
    """Generate the one-step integrator function for differential equations.

    Using this method, the users only need to define the right side of the equation.
    For example, for the `m` channel in the Hodgkin–Huxley neuron model

    .. math::

        \\alpha = {0.1 * (V + 40 \\over 1 - \\exp(-(V + 40) / 10)}

        \\beta = 4.0 * \\exp(-(V + 65) / 18)

        {dm \\over dt} = \\alpha * (1 - m) - \\beta * m

    Using ``BrainPy``, this ODE function can be written as

    >>> import brainpy.numpy as np
    >>> from brainpy import integrate
    >>>
    >>> @integrate(method='rk4')
    >>> def int_m(m, t, V):
    >>>     alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    >>>     beta = 4.0 * np.exp(-(V + 65) / 18)
    >>>     return alpha * (1 - m) - beta * m

    Parameters
    ----------
    func : callable
        The function at the right hand of the differential equation.
        If a stochastic equation (SDE) is defined, then `func` is the drift coefficient
        (the deterministic part) of the SDE.
    method : None, str, callable
        The method of numerical integrator.

    Returns
    -------
    integrator : Integrator
        If `f` is provided, then the one-step numerical integrator will be returned.
        if not, the wrapper will be provided.
    """

    method = method if method is not None else profile.get_method()
    _integrator_ = get_integrator(method)

    if func is None:
        return lambda f: _integrator_(DiffEquation(func=f))

    else:
        return _integrator_(DiffEquation(func=func))
