# -*- coding: utf-8 -*-
import numba
import numpy as np
try:
  from numba import njit
except (ImportError, ModuleNotFoundError):
  njit = None


__all__ = [
  'numba_jit',
  'numba_seed',
  'numba_range',
  'SUPPORT_NUMBA',
]


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
