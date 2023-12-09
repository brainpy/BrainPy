import os
import sys
from jax.lib import xla_client

__all__ = [
  'import_taichi',
  'import_brainpylib_cpu_ops',
  'import_brainpylib_gpu_ops',
]

_minimal_brainpylib_version = '0.1.10'
_minimal_taichi_version = (1, 7, 0)

taichi = None
brainpylib_cpu_ops = None
brainpylib_gpu_ops = None

taichi_install_info = (f'We need taichi=={_minimal_taichi_version}. '
                       f'Currently you can install taichi=={_minimal_taichi_version} through:\n\n'
                       '> pip install taichi==1.7.0')
os.environ["TI_LOG_LEVEL"] = "error"


def import_taichi():
  global taichi
  if taichi is None:
    with open(os.devnull, 'w') as devnull:
      old_stdout = sys.stdout
      sys.stdout = devnull
      try:
        import taichi as taichi  # noqa
      except ModuleNotFoundError:
        raise ModuleNotFoundError(taichi_install_info)
      finally:
        sys.stdout = old_stdout

  if taichi.__version__ != _minimal_taichi_version:
    raise RuntimeError(taichi_install_info)
  return taichi


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
