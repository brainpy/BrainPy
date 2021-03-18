# -*- coding: utf-8 -*-

"""
The setting of the overall framework by ``profile.py`` API.
"""

__all__ = [
    'set_class_keywords',

    'set_dt',
    'get_dt',

    'set_numerical_method',
    'get_numerical_method',
]

_dt = 0.1
_method = 'euler'
CLASS_KEYWORDS = ['self', 'cls']


def set_class_keywords(*args):
    global CLASS_KEYWORDS
    CLASS_KEYWORDS = list(args)


def set_dt(dt):
    """Set the numerical integrator precision.

    Parameters
    ----------
    dt : float
        Numerical integration precision.
    """
    assert isinstance(dt, float)
    global _dt
    _dt = dt


def get_dt():
    """Get the numerical integrator precision.

    Returns
    -------
    dt : float
        Numerical integration precision.
    """
    return _dt


def set_numerical_method(method):
    """Set the default numerical integrator method for differential equations.

    Parameters
    ----------
    method : str, callable
        Numerical integrator method.
    """
    from brainpy.integrators import _SUPPORT_METHODS

    if not isinstance(method, str):
        raise ValueError(f'Only support string, not {type(method)}.')
    if method not in _SUPPORT_METHODS:
        raise ValueError(f'Unsupported numerical method: {method}.')

    global _method
    _method = method


def get_numerical_method():
    """Get the default numerical integrator method.

    Returns
    -------
    method : str
        The default numerical integrator method.
    """
    return _method
