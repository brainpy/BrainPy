from brainpy._src.dependency_check import import_taichi_else_None

from brainpy._src.math.sparse import (
  seg_matmul,
)
if import_taichi_else_None() is not None:
  from brainpy._src.math.sparse import (
    csrmv,

    csr_to_dense as csr_to_dense,
    csr_to_coo as csr_to_coo,
    coo_to_csr as coo_to_csr,
  )

