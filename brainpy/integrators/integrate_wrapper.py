# -*- coding: utf-8 -*-

from . import ode
from . import sde

__all__ = [
    'odeint',
    'sdeint',
    'ddeint',
    'fdeint',
]

supported_ode = [m for m in dir(ode) if not m.startswith('__')]
supported_sde = [m for m in dir(sde) if not m.startswith('__')]


def odeint(f=None, method=None, **kwargs):
    def wrapper(f, ode_type, **kwargs):
        integrator = getattr(ode, ode_type)
        return integrator(f, **kwargs)

    if method is None:
        method = 'euler'
    if method not in supported_ode:
        raise ValueError(f'Unknown ODE numerical method "{method}". Currently '
                         f'BrainPy only support: {supported_ode}')

    if f is None:
        return lambda f: wrapper(f, method, **kwargs)
    else:
        return wrapper(f, method, **kwargs)


def sdeint(f=None, method=None, **kwargs):
    def wrapper(f, ode_type, **kwargs):
        integrator = getattr(sde, ode_type)
        return integrator(f, **kwargs)

    if method is None:
        method = 'euler'
    if method not in supported_sde:
        raise ValueError(f'Unknown SDE numerical method "{method}". Currently '
                         f'BrainPy only support: {supported_sde}')

    if f is None:
        return lambda f: wrapper(f, method, **kwargs)
    else:
        return wrapper(f, method, **kwargs)


def ddeint():
    pass


def fdeint():
    pass
