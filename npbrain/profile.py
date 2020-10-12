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

_backend = 'numpy'
_device = 'cpu'
_dt = 0.1
_method = 'euler'
_numba_setting = {'nopython': True, 'fastmath': True,
                  'nogil': False, 'parallel': False}

show_formatted_code = False
auto_pep8 = True
substitute_equation = False
merge_integral = False


def set(backend=None, numerical_method=None, dt=None, float_type=None, int_type=None,
        merge_ing=None, substitute=None, show_code=None):
    if backend is not None:
        set_backend(backend)

    if numerical_method is not None:
        set_method(numerical_method)

    if dt is not None:
        set_dt(dt)

    if float_type is not None:
        from .numpy import _set_default_float

        _set_default_float(float_type)

    if int_type is not None:
        from .numpy import _set_default_int

        _set_default_int(int_type)

    if merge_ing is not None:
        global merge_integral
        merge_integral = merge_ing

    if substitute is not None:
        global substitute_equation
        substitute_equation = substitute

    if show_code is not None:
        global show_formatted_code
        show_formatted_code = show_code


def set_backend(bk):
    """Set the backend.

    Parameters
    ----------
    bk : str
        The backend name.
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
    if bk.lower() == 'numpy':
        backend = 'numpy'
    elif bk.lower() == 'numba':
        backend = 'numba'
    elif bk.lower() == 'jax':
        backend = 'jax'
    else:
        raise ValueError(f'Unknown backend: {bk}.')

    # switch backend and device
    if device != _device:
        _device = device

    if backend != _backend:
        _backend = backend
        from .numpy import _reload as r1
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


def set_numba_profile(**kwargs):
    """Set the compilation options of Numba JIT function.

    Parameters
    ----------
    kwargs : dict
        The arguments, including ``cache``, ``fastmath``,
        ``parallel``, ``nopython``.
    """
    global _numba_setting

    if 'fastmath' in kwargs:
        _numba_setting['fastmath'] = kwargs.pop('fastmath')
    if 'nopython' in kwargs:
        _numba_setting['nopython'] = kwargs.pop('nopython')
    if 'nogil' in kwargs:
        _numba_setting['nogil'] = kwargs.pop('nogil')
    if 'parallel' in kwargs:
        _numba_setting['parallel'] = kwargs.pop('parallel')


def get_numba_profile():
    """Get the compilation setting of numba JIT function.

    Returns
    -------
    numba_setting : dict
        numba setting.
    """
    return _numba_setting


def set_dt(dt):
    """Set the numerical integrator precision.

    Parameters
    ----------
    dt : float
        precision.
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


def set_method(method):
    """Set the default numerical integrator method for differential equations.

    Parameters
    ----------
    method : str, callable
        Numerical integrator method.
    """
    from npbrain.integration import _SUPPORT_METHODS

    if not isinstance(method, str):
        raise ValueError(f'Only support string, not {type(method)}.')
    if method not in _SUPPORT_METHODS:
        raise ValueError(f'Unsupported numerical method: {method}.')

    global _method
    _method = method


def get_method():
    """Get the default numerical integrator method.

    Returns
    -------
    method : str
        The default numerical integrator method.
    """
    return _method
