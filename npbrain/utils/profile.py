# -*- coding: utf-8 -*-

"""
The setting of the overall framework.

Using the API in ``profile.py``, you can set

- the backend of numerical algorithm, ``numpy`` or ``numba``,
- the compilation options of JIT function in ``numba``,
- the precision of the numerical integration,
- the method of the numerical integration.


Example
-------

Set the default numerical integration precision.

>>> from npbrain.utils import profile
>>> profile.set_dt(0.01)

Set the default numerical integration alorithm.

>>> from npbrain.utils import profile
>>> profile.set_method('euler')
>>> # Or, you can use
>>> from npbrain.core import forward_Euler
>>> profile.set_method(forward_Euler)

Set the default backend to ``numba`` and change the default JIT options.

>>> from npbrain.utils import profile
>>> profile.set_backend('numba')
>>> profile.set_numba(nopython=True, fastmath=True, parallel=True, cache=True)

"""

__all__ = [
    'set_backend',
    'get_backend',
    'is_numba_bk',
    'set_numba',
    'get_numba_profile',

    'set_dt',
    'get_dt',
    'set_method',
    'get_method',
]

# -------------------
# backend
# -------------------

_backend = 'numpy'
_nopython = True
_fastmath = True
_parallel = False
_cache = False


def set_backend(bk):
    """Set the backend.

    Parameters
    ----------
    bk : str
        The backend name.
    """
    global _backend

    if bk in ['numpy', 'np', 'Numpy']:
        _backend = 'numpy'
    elif bk in ['numba', 'numba_cpu', 'numba-cpu', 'nb', 'nb_cpu', 'nb-cpu']:
        _backend = 'numba_cpu'
    elif bk in ['numba_gpu', 'numba-gpu', 'nb_gpu', 'nb-gpu']:
        raise NotImplementedError
        _backend = 'numba_gpu'
    else:
        raise ValueError('Unknown backend: {}'.format(bk))


def get_backend():
    """Get the backend.

    Returns
    -------
    backend: str
        Backend name.

    """
    return _backend


def is_numba_bk():
    """Check whether the backend is ``numba``.

    Returns
    -------
    numba_backend : bool
        True or False.
    """
    return _backend.startswith('numba')


def set_numba(**kwargs):
    """Set the compilation options of Numba JIT function.

    :param kwargs: The arguements, including ``cache``, ``fastmath``,
                    ``parallel``, ``nopython``.
    :type kwargs: dict
    """
    if 'cache' in kwargs:
        global _cache
        _cache = kwargs.pop('cache')
    if 'fastmath' in kwargs:
        global _fastmath
        _fastmath = kwargs.pop('fastmath')
    if 'parallel' in kwargs:
        global _parallel
        _parallel = kwargs.pop('parallel')
    if 'nopython' in kwargs:
        global _nopython
        _nopython = kwargs.pop('nopython')


def get_numba_profile():
    """Get the compilation setting of numba JIT function.

    :return: Settings.
    :rtype: dict
    """
    return {'cache': _cache,
            'fastmath': _fastmath,
            'parallel': _parallel,
            'nopython': _nopython}


# ----------------------
# Numerical integration
# ----------------------

_dt = 0.1
_method = 'euler'


def set_dt(dt):
    """Set the numerical integration precision.

    Parameters
    ----------
    dt : float
        precision.
    """
    assert isinstance(dt, float)
    global _dt
    _dt = dt


def get_dt():
    """Get the numerical integration precision.

    :return: Precision.
    :rtype: float
    """
    return _dt


def set_method(method):
    """Set the default numerical integration method for
     differential equations (DE).

    Parameters
    ----------
    method : str, callable
        DE numerical integration method.
    """
    global _method

    ODE = ['euler', 'forward_Euler', 'explicit_Euler',
           'rk2', 'RK2', 'modified_Euler', 'explicit_midpoint_Euler',
           'rk3', 'RK3', 'rk4', 'RK4', 'RK4_alternative', 'rk4_alternative',
           'backward_Euler', 'implicit_Euler', 'trapezoidal_rule']
    SDE = ['Euler_method', 'EM', 'Euler', 'Euler_Maruyama_method', 'EM_method',
           'Milstein_dfree_Ito',
           'Heun_method_2',
           'Heun', 'Euler_Heun', 'Euler_Heun_method', 'Heun_method',
           'Milstein_dfree_Stra', ]

    if isinstance(method, str):
        if method not in SDE + ODE:
            raise ValueError('Unknown ODE method: ', method)
    elif not callable(method):
        raise ValueError('Unknown method type.')

    _ode_method = method


def get_method():
    """Get the default DE numerical integration method.

    Returns
    -------
    method : str, callable
        The default DE numerical integration method.
    """
    return _method
