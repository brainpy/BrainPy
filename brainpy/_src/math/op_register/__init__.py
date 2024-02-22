from brainpy._src.dependency_check import import_numba_else_None, import_taichi_else_None

numba = import_numba_else_None()
taichi = import_taichi_else_None()

if numba is not None:
  from .numba_approach import (CustomOpByNumba,
                               register_op_with_numba,
                               compile_cpu_signature_with_numba)
  from .base import XLACustomOp
  from .utils import register_general_batching
if taichi is not None:
  from .taichi_aot_based import clean_caches, check_kernels_count
  from .base import XLACustomOp
  from .utils import register_general_batching
