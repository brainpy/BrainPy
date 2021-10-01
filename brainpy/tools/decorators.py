# -*- coding: utf-8 -*-

import numpy

try:
  import numba
except ModuleNotFoundError:
  numba = None

__all__ = [
  'numba_jit',
  'numba_seed',
]


def numba_jit(f=None, **setting):
  if numba is None:
    if f is None: return lambda f: f
    else: return f
  else:
    if f is None: return lambda f: numba.njit(f, **setting)
    else: return numba.njit(f, **setting)


@numba_jit
def _numba_seed(seed=None):
  numpy.random.seed(seed)


def numba_seed(seed):
  if numba is None:
    pass
  else:
    assert isinstance(seed, int)
    _numba_seed(seed)

