# -*- coding: utf-8 -*-

import numpy as np

try:
  import numba
  from numba import njit
except (ImportError, ModuleNotFoundError):
  njit = numba = None

try:
  import brainpylib
except (ImportError, ModuleNotFoundError):
  brainpylib = None


__all__ = [
  'import_numba',
  'import_brainpylib',
  'numba_jit',
  'numba_seed',
  'numba_range',
  'SUPPORT_NUMBA',
]


_minimal_brainpylib_version = '0.1.9'


def import_numba():
  if numba is None:
    raise ModuleNotFoundError('Numba is needed. Please install numba through:\n\n'
                              '> pip install numba')
  return numba


def import_brainpylib():
  if brainpylib is None:
    raise ModuleNotFoundError('brainpylib is needed. Please install brainpylib through:\n'
                              '> pip install brainpylib\n\n')
  if brainpylib.__version__ < _minimal_brainpylib_version:
    raise SystemError(f'This version of brainpy needs brainpylib >= {_minimal_brainpylib_version}.')
  return brainpylib


SUPPORT_NUMBA = njit is not None


def numba_jit(f=None, **kwargs):
  if f is None:
    return lambda f: (f if (njit is None) else njit(f, **kwargs))
  else:
    if njit is None:
      return f
    else:
      return njit(f)


@numba_jit
def _seed(seed):
  np.random.seed(seed)


def numba_seed(seed):
  if njit is not None and seed is not None:
    _seed(seed)


numba_range = numba.prange if SUPPORT_NUMBA else range
