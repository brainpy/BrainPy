# -*- coding: utf-8 -*-
import types

from .numpy_ import *

__all__ = [
    'switch_to',
    'set_ops_from_module',
    'set_ops',
    'set_buffer',
    'get_buffer',

    'BUFFER',
    'OPS_FOR_SOLVER',
    'OPS_FOR_SIMULATION',
]

_backend = 'numpy'
BUFFER = {}
OPS_FOR_SOLVER = ['normal', 'sum', 'exp', 'shape', ]
OPS_FOR_SIMULATION = ['as_tensor', 'zeros', 'ones', 'arange',
                      'vstack', 'where', 'unsqueeze', 'squeeze']
OPS_OF_DTYPE = ['bool',
                'int', 'int32', 'int64',
                'float', 'float32', 'float64']


def switch_to(backend):
    if backend == 'numpy':
        from . import numpy_
        set_ops_from_module(numpy_)

    elif backend == 'pytorch':
        from . import pytorch_
        set_ops_from_module(pytorch_)

    elif backend == 'tensorflow':
        from . import tensorflow_
        set_ops_from_module(tensorflow_)

    elif backend == 'numba':
        from . import numba_cpu
        set_ops_from_module(numba_cpu)

    elif backend == 'numba-parallel':
        from . import numba_cpu
        set_ops_from_module(numba_cpu)

    elif backend == 'numba-cuda':
        from . import numba_cuda
        set_ops_from_module(numba_cuda)

    elif backend == 'jax':
        from . import jax_
        set_ops_from_module(jax_)

    else:
        ops_in_buffer = get_buffer(backend)
        for ops in OPS_FOR_SOLVER:
            if ops not in ops_in_buffer:
                raise ValueError(f'Necessary operation "{ops}" is not '
                                 f'defined in "{backend}" backend\'s buffers.')

    # set operations from BUFFER
    ops_in_buffer = get_buffer(backend)
    set_ops(**ops_in_buffer)
    global _backend
    _backend = backend


def set_ops_from_module(module):
    """Set operations from a module.

    Parameters
    ----------
    module :
    """

    ops_in_module = {p: getattr(module, p) for p in dir(module)
                     if (not p.startswith('__')) and
                     (not isinstance(getattr(module, p), types.ModuleType))}
    global_vars = globals()

    for ops in OPS_FOR_SOLVER:
        if ops not in ops_in_module:
            raise ValueError(f'Operation "{ops}" is needed, but is not '
                             f'defined in module "{module}".')
        global_vars[ops] = ops_in_module.pop(ops)
    for ops in OPS_FOR_SIMULATION:
        if ops in ops_in_module:
            global_vars[ops] = ops_in_module.pop(ops)
        else:
            del global_vars[ops]
    for ops in OPS_OF_DTYPE:
        if ops in ops_in_module:
            global_vars[ops] = ops_in_module.pop(ops)
        else:
            del global_vars[ops]

    for ops, val in ops_in_module.items():
        global_vars[ops] = val


def set_ops(**kwargs):
    """Set operations.

    Parameters
    ----------
    kwargs :
        The key=operation setting.
    """
    global_vars = globals()
    for key, value in kwargs.items():
        global_vars[key] = value


def set_buffer(backend, *args, **kwargs):
    global BUFFER
    if backend not in BUFFER:
        BUFFER[backend] = dict()

    # store operations in buffer
    for arg in args:
        assert isinstance(arg, dict), f'Must be a dict with the format of (key, func) when ' \
                                      f'provide *args, but we got {type(arg)}'
        for key, func in arg.items():
            assert callable(func), f'Must be dictionary with the format of (key, func) when ' \
                                   f'provide *args, but we got {key} = {func}.'
            BUFFER[backend][key] = func
    for key, func in kwargs.items():
        assert callable(func), f'Must be dictionary with the format of key=func when provide ' \
                               f'**kwargs, but we got {key} = {func}.'
        BUFFER[backend][key] = func

    # set the operations if the buffer backend
    # is consistent with the global backend.
    if backend == _backend:
        set_ops(**BUFFER[backend])


def get_buffer(backend):
    return BUFFER.get(backend, dict())
