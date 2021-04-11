# -*- coding: utf-8 -*-


try:
    import numba
except ModuleNotFoundError:
    numba = None


__all__ = [
    'numba_jit'
]


def numba_jit(f):
    if numba is None:
        return f
    else:
        return numba.njit(f)
