# -*- coding: utf-8 -*-

import numpy as np

try:
  import numba
except (ImportError, ModuleNotFoundError):
  numba = None

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


def import_numba():
  if numba is None:
    raise ModuleNotFoundError('Numba is needed. Please install numba through:\n\n'
                              '> pip install numba')
  return numba


def import_brainpylib():
  if brainpylib is None:
    raise ModuleNotFoundError('brainpylib is needed. Please install brainpylib through:\n'
                              '> pip install brainpylib\n\n')
  return brainpylib


SUPPORT_NUMBA = numba is not None


def numba_jit(f=None, **kwargs):
  if f is None:
    return lambda f: (f if (numba is None) else numba.njit(f, **kwargs))
  else:
    if numba is None:
      return f
    else:
      return numba.njit(f)


@numba_jit
def _seed(seed):
  np.random.seed(seed)


def numba_seed(seed):
  if numba is not None and seed is not None:
    _seed(seed)


numba_range = numba.prange if SUPPORT_NUMBA else range
