# -*- coding: utf-8 -*-

import numpy

try:
  import numba
except ModuleNotFoundError:
  numba = None

__all__ = [
  'numba_range',
  'numba_njit',
  'numba_pjit',
  'numba_jit',
  'numba_seed',
]

if numba is None:
  numba_range = range
else:
  numba_range = numba.prange


def numba_jit(f=None, **setting):
  if numba is None:
    if f is None: return lambda f: f
    else: return f
  else:
    if f is None: return lambda f: numba.njit(f, **setting)
    else: return numba.njit(f, **setting)


def numba_njit(f=None):
  if numba is None:
    if f is None: return lambda f: f
    else: return f
  else:
    if f is None: return lambda f: numba.njit(f)
    else: return numba.njit(f)


def numba_pjit(f=None):
  if numba is None:
    if f is None: return lambda f: f
    else: return f
  else:
    if f is None: return lambda f: numba.njit(f, parallel=True, nogil=True)
    else: return numba.njit(f, parallel=True, nogil=True)


@numba_jit
def _numba_seed(seed=None):
  numpy.random.seed(seed)


def numba_seed(seed):
  if numba is None:
    pass
  else:
    if seed is None: seed = numpy.random.randint(0, 1000000)
    assert isinstance(seed, int)
    _numba_seed(seed)




