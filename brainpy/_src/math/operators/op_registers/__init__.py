# -*- coding: utf-8 -*-

"""
This module aims to provide a convenient way to register customized operators
based on Python syntax.
"""

from . import (
  numba_approach,
  taichi_approach,
  triton_approach,
  compat,
  utils
)
from .compat import *
from .numba_approach import *
from .taichi_approach import *
from .triton_approach import *
from .op_register import *
from .utils import *

__all__ = (
    numba_approach.__all__ +
    taichi_approach.__all__ +
    triton_approach.__all__ +
    op_register.__all__ +
    compat.__all__ +
    utils.__all__
)
