# -*- coding: utf-8 -*-

import types

from brainpy.backend.ops.necessary_ops.numpy_ import *

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
OPS_FOR_SIMULATION = [
  'as_tensor',  # for array creation
  'zeros',  # for array creation
  'ones',  # for array creation
  'arange',  # for array creation, for example, times
  'concatenate',  # for monitor data concatenation
  'where',  # for connectivity
  'reshape'
]
OPS_OF_DTYPE = ['bool',
                'int', 'int32', 'int64',
                'float', 'float32', 'float64']

_fixed_vars = [
  # packages
  'types',
  # variables
  '_backend', '_fixed_vars',
  'BUFFER', 'OPS_FOR_SOLVER', 'OPS_FOR_SIMULATION', 'OPS_OF_DTYPE',
  # functions
  'switch_to',
  'set_ops_from_module', 'set_ops', 'set_buffer', 'get_buffer',
]


def switch_to(backend):
  global _backend
  if _backend == backend:
    return

  # 1. pop out all operations
  global_vars = globals()
  for key in list(global_vars.keys()):
    if (not key.startswith('__')) and (key not in _fixed_vars):
      global_vars.pop(key)

  # 2. append new operations in the new backend
  if backend == 'numpy':
    from brainpy.backend.ops.necessary_ops import numpy_
    set_ops_from_module(numpy_)

  elif backend == 'pytorch':
    from brainpy.backend.ops.necessary_ops import pytorch_
    set_ops_from_module(pytorch_)

  elif backend == 'tensorflow':
    from brainpy.backend.ops.necessary_ops import tensorflow_
    set_ops_from_module(tensorflow_)

  elif backend == 'numba':
    from brainpy.backend.ops.necessary_ops import numba
    set_ops_from_module(numba)

  elif backend == 'numba-parallel':
    from brainpy.backend.ops.necessary_ops import numba
    set_ops_from_module(numba)

  elif backend == 'jax':
    from brainpy.backend.ops.necessary_ops import jax_
    set_ops_from_module(jax_)

  else:
    if backend not in BUFFER:
      raise ValueError(f'Cannot switch to "{backend}" backend, because '
                       f'"{backend}" is neither a pre-defined backend '
                       f'(support "numpy", "numba", "jax", "pytorch", '
                       f'"tensorflow"), nor a backend in the BUFFER.')
    ops_in_buffer = get_buffer(backend)
    for ops in OPS_FOR_SOLVER:
      if ops not in ops_in_buffer:
        raise ValueError(f'Necessary operation "{ops}" is not '
                         f'defined in "{backend}" backend\'s buffers.')

  # 3. set operations from BUFFER
  ops_in_buffer = get_buffer(backend)
  set_ops(**ops_in_buffer)

  _backend = backend


def set_ops_from_module(module):
  """Set operations from a module.

  Parameters
  ----------
  module :
  """

  ops_in_module = {}
  for p in dir(module):
    val = getattr(module, p)
    if (not p.startswith('__')) and (not isinstance(val, types.ModuleType)):
      ops_in_module[p] = val

  global_vars = globals()

  for ops in OPS_FOR_SOLVER:
    if ops not in ops_in_module:
      raise ValueError(f'Operation "{ops}" is needed, but is not '
                       f'defined in module "{module}".')
    global_vars[ops] = ops_in_module.pop(ops)
  for ops in OPS_FOR_SIMULATION:
    if ops in ops_in_module:
      global_vars[ops] = ops_in_module.pop(ops)
  for ops in OPS_OF_DTYPE:
    if ops in ops_in_module:
      global_vars[ops] = ops_in_module.pop(ops)

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


from brainpy.backend.ops.more_unified_ops import numpy_
