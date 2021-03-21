# -*- coding: utf-8 -*-

from . import ode
from . import sde

__all__ = [
    'odeint',
    'sdeint',
    'ddeint',
    'fdeint',

    'set_default_odeint',
    'get_default_odeint',
    'set_default_sdeint',
    'get_default_sdeint',
]

_DEFAULT_ODE_METHOD = 'euler'
_DEFAULT_SDE_METHOD = 'euler'
SUPPORTED_ODE = [m for m in dir(ode) if not m.startswith('__')]
SUPPORTED_SDE = [m for m in dir(sde) if not m.startswith('__')]


def _wrapper(f, method, module, **kwargs):
    integrator = getattr(module, method)
    return integrator(f, **kwargs)


def odeint(f=None, method=None, **kwargs):
    if method is None:
        method = _DEFAULT_ODE_METHOD
    if method not in SUPPORTED_ODE:
        raise ValueError(f'Unknown ODE numerical method "{method}". Currently '
                         f'BrainPy only support: {SUPPORTED_ODE}')

    if f is None:
        return lambda f: _wrapper(f, method=method, module=ode, **kwargs)
    else:
        return _wrapper(f, method=method, module=ode, **kwargs)


def sdeint(f=None, method=None, **kwargs):
    if method is None:
        method = _DEFAULT_SDE_METHOD
    if method not in SUPPORTED_SDE:
        raise ValueError(f'Unknown SDE numerical method "{method}". Currently '
                         f'BrainPy only support: {SUPPORTED_SDE}')

    if f is None:
        return lambda f: _wrapper(f, method=method, module=sde, **kwargs)
    else:
        return _wrapper(f, method=method, module=sde, **kwargs)


def ddeint():
    raise NotImplementedError


def fdeint():
    raise NotImplementedError


def set_default_odeint(method):
    """Set the default ODE numerical integrator method for differential equations.

    Parameters
    ----------
    method : str, callable
        Numerical integrator method.
    """
    if not isinstance(method, str):
        raise ValueError(f'Only support string, not {type(method)}.')
    if method not in SUPPORTED_ODE:
        raise ValueError(f'Unsupported ODE numerical method: {method}.')

    global _DEFAULT_ODE_METHOD
    _DEFAULT_ODE_METHOD = method


def get_default_odeint():
    """Get the default ODE numerical integrator method.

    Returns
    -------
    method : str
        The default numerical integrator method.
    """
    return _DEFAULT_ODE_METHOD


def set_default_sdeint(method):
    """Set the default SDE numerical integrator method for differential equations.

    Parameters
    ----------
    method : str, callable
        Numerical integrator method.
    """
    if not isinstance(method, str):
        raise ValueError(f'Only support string, not {type(method)}.')
    if method not in SUPPORTED_SDE:
        raise ValueError(f'Unsupported SDE numerical method: {method}.')

    global _DEFAULT_SDE_METHOD
    _DEFAULT_SDE_METHOD = method


def get_default_sdeint():
    """Get the default ODE numerical integrator method.

    Returns
    -------
    method : str
        The default numerical integrator method.
    """
    return _DEFAULT_SDE_METHOD
