# -*- coding: utf-8 -*-

"""
The setting of the overall framework.

Using the API in ``profile.py``, you can set

- the backend of numerical algorithm, ``backend`` or ``numba``,
- the precision of the numerical integrator,
- the method of the numerical integrator.

"""


__all__ = [
    'set',

    'is_jit_backend',
    'is_cpu_device',

    'set_numba_profile',
    'get_numba_profile',

    'set_device',
    'get_device',

    'set_dt',
    'get_dt',

    'set_numerical_method',
    'get_numerical_method',
]


_jit = False
_backend = 'backend'
_device = 'cpu'
_dt = 0.1
_method = 'euler'
_numba_setting = {'nopython': True,
                  'fastmath': True,
                  'nogil': True,
                  'parallel': False}

_show_format_code = False
_show_code_scope = False
_substitute_equation = False
_merge_integrators = True
_merge_steps = False


def set(
        jit=None,
        device=None,
        numerical_method=None,
        dt=None,
        float_type=None,
        int_type=None,
        merge_integrators=None,
        merge_steps=None,
        substitute=None,
        show_code=None,
        show_code_scope=None
):

    # JIT and device
    if device is not None and jit is None:
        assert isinstance(device, str), "'device' must a string."
        set_device(_jit, device=device)
    if jit is not None:
        assert isinstance(jit, bool), "'jit' must be True or False."
        if device is not None:
            assert isinstance(device, str), "'device' must a string."
        set_device(jit, device=device)

    # numerical integration method
    if numerical_method is not None:
        assert isinstance(numerical_method, str), '"numerical_method" must be a string.'
        set_numerical_method(numerical_method)

    # numerical integration precision
    if dt is not None:
        assert isinstance(dt, (float, int)), '"dt" must be float or int.'
        set_dt(dt)

    # default float type
    if float_type is not None:
        from .backend import _set_default_float
        _set_default_float(float_type)

    # default int type
    if int_type is not None:
        from .backend import _set_default_int
        _set_default_int(int_type)

    # option to merge integral functions
    if merge_integrators is not None:
        assert isinstance(merge_integrators, bool), '"merge_integrators" must be True or False.'
        global _merge_integrators
        _merge_integrators = merge_integrators

    # option to merge step functions
    if merge_steps is not None:
        assert isinstance(merge_steps, bool), '"merge_steps" must be True or False.'
        global _merge_steps
        _merge_steps = merge_steps

    # option of the equation substitution
    if substitute is not None:
        assert isinstance(substitute, bool), '"substitute" must be True or False.'
        global _substitute_equation
        _substitute_equation = substitute

    # option of the formatted code output
    if show_code is not None:
        assert isinstance(show_code, bool), '"show_code" must be True or False.'
        global _show_format_code
        _show_format_code = show_code

    # option of the formatted code scope
    if show_code_scope is not None:
        assert isinstance(show_code_scope, bool), '"show_code_scope" must be True or False.'
        global _show_code_scope
        _show_code_scope = show_code_scope


def set_device(jit, device=None):
    """Set the backend and the device to deploy the models.

    Parameters
    ----------
    jit : book
        Whether use the jit acceleration.
    device : str, optional
        The device name.
    """

    # jit
    # ---

    global _jit

    if _jit != jit:
        _jit = jit

    # device
    # ------

    global _device

    if device is None:
        return

    device = device.lower()
    if _device != device:
        if not jit:
            if device != 'cpu':
                print(f'Non-jit mode only support "cpu" device, not "{device}".')
            else:
                _device = device
        else:
            if device == 'cpu':
                set_numba_profile(parallel=False)
            elif device == 'multi-cpu':
                set_numba_profile(parallel=True)
            elif device == 'gpu':
                raise NotImplementedError('BrainPy currently doesn\'t support GPU.')
            else:
                raise ValueError(f'Unknown device in Numba mode: {device}.')
            _device = device


def get_device():
    """Get the device name.

    Returns
    -------
    device: str
        Device name.

    """
    return _device


def is_jit_backend():
    """Check whether the backend is ``numba``.

    Returns
    -------
    jit : bool
        True or False.
    """
    return _jit


def is_cpu_device():
    """Check whether the device is "CPU".

    Returns
    -------
    device : bool
        True or False.
    """
    return _device.endswith('cpu')


def set_numba_profile(**kwargs):
    """Set the compilation options of Numba JIT function.

    Parameters
    ----------
    kwargs : Any
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
        Numba setting.
    """
    return _numba_setting


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
    from brainpy.integration import _SUPPORT_METHODS

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
