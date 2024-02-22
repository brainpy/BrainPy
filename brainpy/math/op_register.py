# -*- coding: utf-8 -*-
from brainpy._src.dependency_check import import_taichi_else_None, import_numba_else_None

if import_taichi_else_None() is not None and import_numba_else_None() is not None:
  from brainpy._src.math.op_register import (
    CustomOpByNumba,
    compile_cpu_signature_with_numba,
    clean_caches,
    check_kernels_count,
  )

  from brainpy._src.math.op_register.base import XLACustomOp
  from brainpy._src.math.op_register.ad_support import defjvp


