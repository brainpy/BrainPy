# -*- coding: utf-8 -*-

from . import diff_equation
from . import integrator
from .diff_equation import *
from .integrator import *
from .. import profile


def integrate(func=None, noise=None, method=None):
    """Generate the one-step integrator function for differential equations.

    Using this method, the users only need to define the right side of the equation.
    For example, for the `m` channel in the Hodgkin–Huxley neuron model

    .. math::

        \\alpha = {0.1 * (V + 40 \\over 1 - \\exp(-(V + 40) / 10)}

        \\beta = 4.0 * \\exp(-(V + 65) / 18)

        {dm \\over dt} = \\alpha * (1 - m) - \\beta * m

    Using ``NumpyBrain``, this ODE function can be written as

    >>> import npbrain._numpy as bnp
    >>> from npbrain import integrate
    >>>
    >>> @integrate(method='rk4')
    >>> def int_m(m, t, V):
    >>>     alpha = 0.1 * (V + 40) / (1 - bnp.exp(-(V + 40) / 10))
    >>>     beta = 4.0 * bnp.exp(-(V + 65) / 18)
    >>>     return alpha * (1 - m) - beta * m

    Parameters
    ----------
    func : callable
        The function at the right hand of the differential equation.
        If a stochastic equation (SDE) is defined, then `func` is the drift coefficient
        (the deterministic part) of the SDE.
    noise : callable, float
        The diffusion coefficient (the stochastic part) of the SDE. `noise` can be a float
        number, or a function.
    method : None, str, callable
        The method of numerical integrator.

    Returns
    -------
    func : callable
        If `f` is provided, then the one-step numerical integrator will be returned.
        if not, the wrapper will be provided.
    """

    has_noise = not (noise is None or noise == 0.)
    method = method if method is not None else profile.get_method()
    _integrator_ = get_integrator(method)

    if func is None:
        if not has_noise:  # ODE
            def wrapper(f):
                return _integrator_(DiffEquation(f=f))
            return wrapper
        else:  # SDE
            def wrapper(f):
                return _integrator_(DiffEquation(f=f, g=noise))
            return wrapper

    else:
        if not has_noise:  # ODE
            return _integrator_(DiffEquation(f=func))
        else:  # SDE
            return _integrator_(DiffEquation(f=func, g=noise))
