# -*- coding: utf-8 -*-

import warnings

from .numba_approach import register_op_with_numba

__all__ = [
  'register_op'
]


def register_op(*args, **kwargs):
  warnings.warn('"brainpylib.register_op()" has been deprecated since version 0.1.0. '
                'Please use "brainpylib.register_op_with_numba()" instead.',
                UserWarning)
  return register_op_with_numba(*args, multiple_results=True, **kwargs)
