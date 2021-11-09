# -*- coding: utf-8 -*-

"""
The ``math`` module for whole BrainPy ecosystem.
This module provides basic mathematical operations, including:

- numpy-like array operations
- linear algebra functions
- random sampling functions
- discrete fourier transform functions
- compilations of ``jit``, ``vmap``, ``pmap`` for class objects
- automatic differentiation of ``grad``, ``jacocian``, ``hessian``, etc. for class objects
- loss functions
- activation functions
- optimization classes

Details in the following.
"""


from brainpy import errors
from brainpy.math.numpy import *


# 1. backend name
# --------------------------

BACKEND_NAME = 'numpy'


def get_backend_name():
  """Get the current backend name.

  Returns
  -------
  backend : str
      The name of the current backend name.
  """
  return BACKEND_NAME


def is_numpy_backend():
  return get_backend_name() == 'numpy'


def is_jax_backend():
  return get_backend_name() == 'jax'


# 2. numerical precision
# --------------------------

__dt = 0.1


def set_dt(dt):
  """Set the numerical integrator precision.

  Parameters
  ----------
  dt : float
      Numerical integration precision.
  """
  assert isinstance(dt, float), f'"dt" must a float, but we got {dt}'
  global __dt
  __dt = dt


def get_dt():
  """Get the numerical integrator precision.

  Returns
  -------
  dt : float
      Numerical integration precision.
  """
  return __dt


# 3. backend setting
# ------------------


def use_backend(name, module=None):
  # check name
  if not isinstance(name, str):
    raise errors.BrainPyError(f'"name" must be a str, but we got {type(name)}: {name}')

  # check module
  if module is None:
    if name == 'numpy':
      from brainpy.math import numpy as module
    elif name == 'jax':
      try:
        from brainpy.math import jax as module
      except ModuleNotFoundError:
        raise errors.PackageMissingError('"jax" backend need JAX, but is not installed. '
                                         'Please install jax via:\n\n'
                                         '>>> pip install jax\n'
                                         '>>> # or \n'
                                         '>>> conda install jax -c conda-forge')
    else:
      raise errors.BrainPyError(f'Unknown backend "{name}", now we only support: numpy, jax.')
  else:
    from types import ModuleType
    if not isinstance(module, ModuleType):
      raise errors.BrainPyError(f'"module" must be a module, but we got a '
                                f'type of {type(module)}: {module}')

  global_vars = globals()
  if global_vars['BACKEND_NAME'] == name:
    return

  # replace operations
  global_vars['BACKEND_NAME'] = name
  for key, value in module.__dict__.items():
    if key.startswith('_'):
      if key not in ['__name__', '__doc__', '__file__', '__path__']:
        continue
    global_vars[key] = value
