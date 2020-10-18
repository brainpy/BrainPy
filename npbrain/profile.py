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
                  'nogil': True, 'parallel': False}

_show_formatted_code = False
_auto_pep8 = True
_substitute_equation = False
_merge_integral = False


def set(backend=None, device=None, numerical_method=None, dt=None, float_type=None, int_type=None,
        merge_ing=None, substitute=None, show_code=None):
    # backend and device
    if device is not None and backend is None:
        raise ValueError('Please set backend. NumpyBrain now supports "numpy" and "numba" backends.')
    if backend is not None:
        set_backend(backend, device=device)

    # numerical integration method
    if numerical_method is not None:
        set_method(numerical_method)

    # numerical integration precision
    if dt is not None:
        set_dt(dt)

    # default float type
    if float_type is not None:
        from .numpy import _set_default_float
        _set_default_float(float_type)

    # default int type
    if int_type is not None:
        from .numpy import _set_default_int
        _set_default_int(int_type)

    # option of merging integral functions
    if merge_ing is not None:
        global _merge_integral
        _merge_integral = merge_ing

    # option of equation substitution
    if substitute is not None:
        global _substitute_equation
        _substitute_equation = substitute

    # option of formatted code output
    if show_code is not None:
        global _show_formatted_code
        _show_formatted_code = show_code


def set_backend(backend, device=None):
    """Set the backend and the device to deploy the models.

    Parameters
    ----------
    backend : str
        The backend name.
    device : str, optional
        The device name.
    """

    # backend #

    global _backend

    backend = backend.lower()
    if backend not in ['numpy', 'numba', 'jax']:
        raise ValueError(f'Unsupported backend: {backend}.')

    if backend != _backend:
        _backend = backend
        from .numpy import _reload as r1
        r1(_backend)

    # device #

    global _device

    if device:
        device = device.lower()

        if _device != device:
            if backend == 'numpy':
                if device != 'cpu':
                    print(f'NumPy mode only support "cpu" device, not "{device}".')
                else:
                    _device = device
            elif backend == 'numba':
                if device == 'cpu':
                    set_numba_profile(parallel=False)
                elif device == 'multi-core':
                    set_numba_profile(parallel=True)
                elif device == 'gpu':
                    raise NotImplementedError('NumpyBrain currently doesn\'t support GPU.')
                else:
                    raise ValueError(f'Unknown device in Numba mode: {device}.')
                _device = device
            elif backend == 'jax':
                raise NotImplementedError


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
    """Check whether the backend is ``numba``.

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
