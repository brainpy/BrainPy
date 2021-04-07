# -*- coding: utf-8 -*-

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

BUFFER = {}
OPS_FOR_SOLVER = ['normal', 'sum', 'exp', 'matmul', 'shape', ]
OPS_FOR_SIMULATION = ['as_tensor', 'zeros', 'ones', 'arange',
                      'vstack', 'where', 'unsqueeze', 'squeeze']


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
                raise ValueError(f'Operation "{ops}" is needed, but is not '
                                 f'defined in "{backend}" backend\'s buffers.')

    # set operations from BUFFER
    ops_in_buffer = get_buffer(backend)
    set_ops(**ops_in_buffer)


def set_ops_from_module(module):
    """Set operations from a module.

    Parameters
    ----------
    module :
    """
    global_vars = globals()
    for ops in OPS_FOR_SOLVER:
        if not hasattr(module, ops):
            raise ValueError(f'Operation "{ops}" is needed, but is not '
                             f'defined in module "{module}".')
        global_vars[ops] = getattr(module, ops)
    for ops in OPS_FOR_SIMULATION:
        if hasattr(module, ops):
            global_vars[ops] = getattr(module, ops)
        else:
            del global_vars[ops]


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

    # set the operations
    set_ops(**BUFFER[backend])


def get_buffer(backend):
    return BUFFER.get(backend, dict())

