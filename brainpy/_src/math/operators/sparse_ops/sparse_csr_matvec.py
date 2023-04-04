# -*- coding: utf-8 -*-


from typing import Union, Tuple

import numba
import numpy as np
from jax import numpy as jnp, ensure_compile_time_eval
from jax.core import ShapedArray
from jax.lib import xla_client

from brainpylib._src.errors import GPUOperatorNotFound
from brainpylib._src.op_register import register_op_with_numba
from brainpylib._src.tools import transform_brainpy_array

try:
  from brainpylib import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  'csr_matvec'
]


def csr_matvec(
    data: Union[float, jnp.ndarray],
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    vector: jnp.ndarray,
    *,
    shape: Tuple[int, int],
    method: str = 'vector'
) -> jnp.ndarray:
  """CSR sparse matrix product with a dense vector, which outperforms the cuSPARSE algorithm.

  This function supports JAX transformations, including `jit()`,
  `vmap()` and `pmap()`.

  Parameters
  ----------
  data: ndarray, float
    An array of shape ``(nse,)``.
  indices: ndarray
    An array of shape ``(nse,)``.
  indptr: ndarray
    An array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``.
  vector: ndarray
    An array of shape ``(shape[0] if transpose else shape[1],)``
    and dtype ``data.dtype``.
  shape: tuple of int
    A length-2 tuple representing the matrix shape.
  transpose: bool
    A boolean specifying whether to transpose the sparse matrix
    before computing.
  method: str
    The computing method used in GPU backend. Currently, we support
    `scalar`, `vector` and `adaptive`.

  Returns
  -------
  y : ndarry
    The array of shape ``(shape[1] if transpose else shape[0],)`` representing
    the matrix vector product.
  """

  data = transform_brainpy_array(data)
  indices = transform_brainpy_array(indices)
  indptr = transform_brainpy_array(indptr)
  vector = transform_brainpy_array(vector)
  if method not in ['scalar', 'vector', 'adaptive']:
    raise ValueError('Only support methods: scalar, vector, and adaptive. '
                     f'But we got {method}.')

  data = jnp.atleast_1d(data)
  if not isinstance(data, jnp.ndarray):
    raise TypeError(f'data must a ndarray. But we got {type(data)}')
  if data.dtype not in [jnp.float32, jnp.float64]:
    raise TypeError(f'Only support float32 and float64. But we got {data.dtype}.')
  if data.dtype != vector.dtype:
    raise TypeError('The types of data and vector should be the same. '
                    f'But we got {data.dtype} != {vector.dtype}.')
  assert data.ndim == indices.ndim == indptr.ndim == vector.ndim == 1

  if method == 'adaptive':
    with ensure_compile_time_eval():
      pass
    return csr_matvec_adaptive_p.bind(data, indices, indptr, vector, shape=shape)

  elif method == 'scalar':
    return csr_matvec_scalar_p.bind(data, indices, indptr, vector, shape=shape)

  elif method == 'vector':
    return csr_matvec_vector_p.bind(data, indices, indptr, vector, shape=shape)

  else:
    raise ValueError('Only support methods: scalar, vector, and adaptive. '
                     f'But we got {method}.')


def _csr_matvec_abstract(*args, **kwargs):
  data = args[0]
  assert len(kwargs) == 1
  shape = kwargs['shape']
  return ShapedArray(dtype=data.dtype, shape=(shape[0],))


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _csr_matvec_numba_imp(outs, ins):
  data, indices, indptr, vector, shape = ins
  outs.fill(0)

  if len(data) == 1:
    data = data[0]
    for row_i in numba.prange(shape[0]):
      res = 0.
      for j in range(indptr[row_i], indptr[row_i + 1]):
        res += vector[indices[j]] * data
      outs[row_i] = res

  else:
    for row_i in numba.prange(shape[0]):
      res = 0.
      for j in range(indptr[row_i], indptr[row_i + 1]):
        res += vector[indices[j]] * data[j]
      outs[row_i] = res


def _csr_matvec_scalar_gpu_translation(c, data, indices, indptr, vector, *, shape):
  if gpu_ops is None:
    raise GPUOperatorNotFound(csr_matvec_scalar_p.name)

  data_shape = c.get_shape(data)
  type_name = b'float' if data_shape.element_type() == np.float32 else b'double'
  data_name = b'homo' if data_shape.dimensions() == (1,) else b'heter'
  opaque = gpu_ops.build_double_size_descriptor(shape[0], shape[1])
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'csr_matvec_' + data_name + b'_scalar_' + type_name,
    operands=(data, indices, indptr, vector),
    operand_shapes_with_layout=(c.get_shape(data),
                                c.get_shape(indices),
                                c.get_shape(indptr),
                                c.get_shape(vector)),
    shape_with_layout=xla_client.Shape.array_shape(data_shape.element_type(), (shape[0],), (0,)),
    opaque=opaque,
  )


csr_matvec_scalar_p = register_op_with_numba(
  'csr_matvec_scalar',
  cpu_func=_csr_matvec_numba_imp,
  out_shapes=_csr_matvec_abstract,
  gpu_func_translation=_csr_matvec_scalar_gpu_translation,
)


def _csr_matvec_vector_gpu_translation(c, data, indices, indptr, vector, *, shape):
  if gpu_ops is None:
    raise GPUOperatorNotFound(csr_matvec_vector_p.name)

  data_shape = c.get_shape(data)
  type_name = b'float' if data_shape.element_type() == jnp.float32 else b'double'
  data_name = b'homo' if data_shape.dimensions() == (1,) else b'heter'
  opaque = gpu_ops.build_double_size_descriptor(shape[0], shape[1])
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'csr_matvec_' + data_name + b'_vector_' + type_name,
    operands=(data, indices, indptr, vector),
    operand_shapes_with_layout=(c.get_shape(data),
                                c.get_shape(indices),
                                c.get_shape(indptr),
                                c.get_shape(vector)),
    shape_with_layout=xla_client.Shape.array_shape(data_shape.element_type(), (shape[0],), (0,)),
    opaque=opaque,
  )


csr_matvec_vector_p = register_op_with_numba(
  'csr_matvec_vector',
  cpu_func=_csr_matvec_numba_imp,
  out_shapes=_csr_matvec_abstract,
  gpu_func_translation=_csr_matvec_vector_gpu_translation,
)


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _csr_matvec_adaptive(outs, ins):
  data, indices, indptr, vector, _, shape = ins
  outs.fill(0)

  if len(data) == 1:
    data = data[0]
    for i in numba.prange(shape[0]):
      res = 0.
      for j in range(indptr[i], indptr[i + 1]):
        res += vector[indices[j]] * data
      outs[i] = res

  else:
    for i in numba.prange(shape[0]):
      res = 0.
      for j in range(indptr[i], indptr[i + 1]):
        res += vector[indices[j]] * data[j]
      outs[i] = res


def _csr_matvec_adaptive_gpu_translation(c, data, indices, indptr, row_blocks, vector, *, shape):
  if gpu_ops is None:
    raise GPUOperatorNotFound(csr_matvec_adaptive_p.name)

  data_shape = c.get_shape(data)
  type_name = b'float' if data_shape.element_type() == np.float32 else b'double'
  data_name = b'homo' if data_shape.dimensions() == (1,) else b'heter'
  opaque = gpu_ops.build_double_size_descriptor(shape[0], shape[1])
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'csr_matvec_' + data_name + b'_adaptive_' + type_name,
    operands=(data, indices, indptr, row_blocks, vector),
    operand_shapes_with_layout=(c.get_shape(data),
                                c.get_shape(indices),
                                c.get_shape(indptr),
                                c.get_shape(row_blocks),
                                c.get_shape(vector)),
    shape_with_layout=xla_client.Shape.array_shape(data_shape.element_type(), (shape[0],), (0,)),
    opaque=opaque,
  )


csr_matvec_adaptive_p = register_op_with_numba(
  'csr_matvec_adaptive',
  out_shapes=_csr_matvec_abstract,
  cpu_func=_csr_matvec_adaptive,
  gpu_func_translation=_csr_matvec_adaptive_gpu_translation,
)
