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
>>> from npbrain.core import ode_euler
>>> profile.set_method(ode_euler)

Set the default backend to ``numba`` and change the default JIT options.

>>> from npbrain.utils import profile
>>> profile.set_backend('numba')
>>> profile.set_numba(nopython=True, fastmath=True, parallel=True, cache=True)

"""

import numba as nb
import re
import multiprocessing

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
_nogil = False
_parallel = False
define_signature = False

# dtype of float
ftype = 'float64'
# dtype of int
itype = 'int64'

jit_diff_eq = 'cfunc'

debug = False


def set_backend(bk):
    """Set the backend.

    Parameters
    ----------
    bk : str
        The backend name.
    """
    if get_backend() == bk:
        return

    global _backend
    global _parallel
    global _nogil

    if bk in ['numpy', 'np', 'Numpy']:
        _backend = 'numpy'
    elif bk in ['numba', 'numba_cpu', 'numba-cpu']:
        _backend = bk
        _parallel = False
        _nogil = False
    elif bk.startswith('numba-pa') or bk.startswith('numba_pa'):
        _backend = bk
        _parallel = True
        _nogil = True
        splits = re.split(r'[-_]', bk)
        if len(splits) == 3:
            num_thread = int(splits[2])
            nb.set_num_threads(num_thread)
        elif len(splits) == 2:
            num_thread = multiprocessing.cpu_count()
            nb.set_num_threads(num_thread)
        else:
            raise ValueError('Unknown backend: {}'.format(bk))
    elif bk.startswith('numba-gpu') or bk.startswith('numba_gpu'):
        _backend = bk
        raise NotImplementedError
    else:
        raise ValueError('Unknown backend: {}'.format(bk))

    from npbrain import _reload
    _reload()


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
    global _fastmath
    global _nopython
    global _nogil
    global _parallel
    global define_signature

    if 'fastmath' in kwargs:
        _fastmath = kwargs.pop('fastmath')
    if 'nopython' in kwargs:
        _nopython = kwargs.pop('nopython')
    if 'pre_sig' in kwargs:
        define_signature = kwargs.pop('pre_sig')
    if 'parallel' in kwargs:
        _parallel = kwargs.pop('parallel')
        _nogil = _parallel


def get_numba_profile():
    """Get the compilation setting of numba JIT function.

    :return: Settings.
    :rtype: dict
    """
    return {
        'fastmath': _fastmath,
        'nopython': _nopython,
        'nogil': _nogil,
        'parallel': _parallel,
    }


# ----------------------
# Numerical integration
# ----------------------

dt = 0.1
method = 'euler'


def set_dt(dt_):
    """Set the numerical integration precision.

    Parameters
    ----------
    dt_ : float
        precision.
    """
    assert isinstance(dt_, float)
    global dt
    dt = dt_


def get_dt():
    """Get the numerical integration precision.

    :return: Precision.
    :rtype: float
    """
    return dt


def set_method(method_):
    """Set the default numerical integration method for
     differential equations (DE).

    Parameters
    ----------
    method_ : str, callable
        DE numerical integration method.
    """
    global method

    ODE = ['euler', 'forward_Euler', 'explicit_Euler',
           'rk2', 'RK2', 'modified_Euler', 'explicit_midpoint_Euler',
           'rk3', 'RK3', 'rk4', 'RK4', 'RK4_alternative', 'rk4_alternative',
           'backward_Euler', 'implicit_Euler', 'trapezoidal_rule']
    SDE = ['Euler_method', 'EM', 'Euler', 'Euler_Maruyama_method', 'EM_method',
           'Milstein_dfree_Ito',
           'Heun_method_2',
           'Heun', 'Euler_Heun', 'Euler_Heun_method', 'Heun_method',
           'Milstein_dfree_Stra', ]

    if isinstance(method_, str):
        if method_ not in SDE + ODE:
            raise ValueError('Unknown ODE method: ', method_)
    elif not callable(method_):
        raise ValueError('Unknown method type.')

    method = method_


def get_method():
    """Get the default DE numerical integration method.

    Returns
    -------
    method : str, callable
        The default DE numerical integration method.
    """
    return method
