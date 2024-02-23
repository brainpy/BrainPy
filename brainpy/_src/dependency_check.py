import functools
import os
import sys

from jax.lib import xla_client

__all__ = [
  'import_taichi',
  'raise_taichi_not_found',
  'check_taichi_func',
  'check_taichi_class',
  'import_numba',
  'raise_numba_not_found',
  'check_numba_func',
  'check_numba_class',
  'import_brainpylib_cpu_ops',
  'import_brainpylib_gpu_ops',
]

_minimal_brainpylib_version = '0.2.6'
_minimal_taichi_version = (1, 7, 0)

taichi = None
numba = None
brainpylib_cpu_ops = None
brainpylib_gpu_ops = None

taichi_install_info = (f'We need taichi=={_minimal_taichi_version}. '
                       f'Currently you can install taichi=={_minimal_taichi_version} through:\n\n'
                       '> pip install taichi==1.7.0')
numba_install_info = ('We need numba. Please install numba by pip . \n'
                      '> pip install numba'
                      )
os.environ["TI_LOG_LEVEL"] = "error"


def import_taichi(error_if_not_found=True):
  global taichi
  if taichi is None:
    with open(os.devnull, 'w') as devnull:
      old_stdout = sys.stdout
      sys.stdout = devnull
      try:
        import taichi as taichi  # noqa
      except ModuleNotFoundError:
        if error_if_not_found:
          raise raise_taichi_not_found()
      finally:
        sys.stdout = old_stdout

  if taichi is None:
    return None
  if taichi.__version__ != _minimal_taichi_version:
    raise RuntimeError(taichi_install_info)
  return taichi


def raise_taichi_not_found():
  raise ModuleNotFoundError(taichi_install_info)


def check_taichi_func(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    if taichi is None:
      raise_taichi_not_found()
    return func(*args, **kwargs)

  return wrapper


def check_taichi_class(cls):
  class Wrapper(cls):
    def __init__(self, *args, **kwargs):
      if taichi is None:
        raise_taichi_not_found()
      super().__init__(*args, **kwargs)

  return Wrapper


def import_numba(error_if_not_found=True):
  global numba
  if numba is None:
    try:
      import numba as numba
    except ModuleNotFoundError:
      if error_if_not_found:
        raise_numba_not_found()
      else:
        return None
  return numba


def raise_numba_not_found():
  raise ModuleNotFoundError(numba_install_info)


def check_numba_func(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    if numba is None:
      raise_numba_not_found()
    return func(*args, **kwargs)

  return wrapper


def check_numba_class(cls):
  class Wrapper(cls):
    def __init__(self, *args, **kwargs):
      if numba is None:
        raise_numba_not_found()
      super().__init__(*args, **kwargs)

  return Wrapper


def is_brainpylib_gpu_installed():
  return False if brainpylib_gpu_ops is None else True


def import_brainpylib_cpu_ops():
  global brainpylib_cpu_ops
  if brainpylib_cpu_ops is None:
    try:
      from brainpylib import cpu_ops as brainpylib_cpu_ops

      for _name, _value in brainpylib_cpu_ops.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="cpu")

      import brainpylib
      if brainpylib.__version__ < _minimal_brainpylib_version:
        raise SystemError(f'This version of brainpy needs brainpylib >= {_minimal_brainpylib_version}.')
      if hasattr(brainpylib, 'check_brainpy_version'):
        brainpylib.check_brainpy_version()

    except ImportError:
      raise ImportError('Please install brainpylib. \n'
                        'See https://brainpy.readthedocs.io for installation instructions.')

  return brainpylib_cpu_ops


def import_brainpylib_gpu_ops():
  global brainpylib_gpu_ops
  if brainpylib_gpu_ops is None:
    try:
      from brainpylib import gpu_ops as brainpylib_gpu_ops

      for _name, _value in brainpylib_gpu_ops.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

      import brainpylib
      if brainpylib.__version__ < _minimal_brainpylib_version:
        raise SystemError(f'This version of brainpy needs brainpylib >= {_minimal_brainpylib_version}.')
      if hasattr(brainpylib, 'check_brainpy_version'):
        brainpylib.check_brainpy_version()

    except ImportError:
      raise ImportError('Please install GPU version of brainpylib. \n'
                        'See https://brainpy.readthedocs.io for installation instructions.')

  return brainpylib_gpu_ops
