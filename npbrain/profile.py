# -*- coding: utf-8 -*-

"""
The setting of the overall framework.

Using the API in ``profile.py``, you can set

- the _numpy of numerical algorithm, ``numpy`` or ``numba``,
- the compilation options of JIT function in ``numba``,
- the precision of the numerical integrator,
- the method of the numerical integrator.


"""


__all__ = [
    'set_backend',
    'get_backend',

    'set_dt',
    'get_dt',
    'set_method',
    'get_method',
]

# -------------------
# _numpy
# -------------------

_backend = 'numpy'
_device = 'cpu'

_nopython = True
_fastmath = True
_nogil = False
_parallel = False

# dtype of float
ftype = 'float64'
# dtype of int
itype = 'int64'


show_codgen = False
auto_pep8 = True
substitute_eqs = False
merge_integral = True


def set(backend=None, device=None, method=None, ode_method=None, sde_method=None, dt=None,
        ftype=None, itype=None):
    pass




def set_backend(bk):
    """Set the _numpy.

    Parameters
    ----------
    bk : str
        The _numpy name.
    """
    global _backend
    global _device

    splits = bk.split('-')
    bk = splits[0]

    # device
    if len(splits) == 2:
        device = splits[1]
    else:
        device = 'cpu'

    # _numpy
    if bk.lower() in ['numpy', 'bnp']:
        backend = 'numpy'
    elif bk.lower() in ['numba', 'Numba']:
        backend = 'numba'
    elif bk.lower() in ['jax', 'JAX']:
        backend = 'jax'
    else:
        raise ValueError(f'Unknown _numpy: {bk}')

    # switch _numpy and device
    if device != _device:
        _device = device
    if backend != _backend:
        _backend = backend

        # from npbrain import _reload
        # _reload()
        from ._numpy import _reload as r1
        r1(backend)


def get_backend():
    """Get the _numpy.

    Returns
    -------
    _numpy: str
        Backend name.

    """
    return _backend


def is_jax_bk():
    """Check whether the backend is ``JAX``.

    Returns
    -------
    jax_backend : bool
        True or False.
    """
    return _backend.startswith('jax')


def is_numpy_bk():
    """Check whether the backend is ``Numpy``.

    Returns
    -------
    numpy_backend : bool
        True or False.
    """
    return _backend.startswith('numpy')


def is_numba_bk():
    """Check whether the _numpy is ``numba``.

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
# Numerical integrator
# ----------------------

dt = 0.1
method = 'euler'


def set_dt(dt_):
    """Set the numerical integrator precision.

    Parameters
    ----------
    dt_ : float
        precision.
    """
    assert isinstance(dt_, float)
    global dt
    dt = dt_


def get_dt():
    """Get the numerical integrator precision.

    :return: Precision.
    :rtype: float
    """
    return dt


def set_method(method_):
    """Set the default numerical integrator method for
     differential equations (DE).

    Parameters
    ----------
    method_ : str, callable
        DE numerical integrator method.
    """
    global method

    ODE = ['euler', 'forward_Euler', 'explicit_Euler', 'exponential',
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
    """Get the default DE numerical integrator method.

    Returns
    -------
    method : str, callable
        The default DE numerical integrator method.
    """
    return method
