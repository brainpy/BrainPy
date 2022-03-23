# -*- coding: utf-8 -*-

try:
  from numba import njit
except (ImportError, ModuleNotFoundError):
  njit = None


__all__ = [
  'numba_jit'
]


def numba_jit(f=None, **kwargs):
  if f is None:
    return lambda f: (f if (njit is None) else njit(f, **kwargs))
  else:
    if njit is None:
      return f
    else:
      return njit(f)

