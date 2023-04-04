# -*- coding: utf-8 -*-

from functools import partial
from typing import Tuple, Optional

import jax
import numpy as np
from jax import numpy as jnp, dtypes
from jax.core import ShapedArray, Primitive
from jax.interpreters import xla, ad
from jax.lib import xla_client

from brainpylib._src.errors import GPUOperatorNotFound
from brainpylib._src.op_register import (register_general_batching)
from brainpylib._src.tools import transform_brainpy_array

try:
  from brainpylib import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  # 'matmat_prob_conn_homo_weight',
  'matmat_prob_conn_uniform_weight',
  'matmat_prob_conn_normal_weight',
]


def matmat_prob_conn_homo_weight(
    vector: jnp.ndarray,
    weight: float,
    *,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jnp.ndarray:
  r"""Perform the :math:`Y=X@M` operation, where :math:`X`, :math:`Y` and :math:`M` are matrices,
  and :math:`M` is just-in-time randomly generated with a scalar `weight` at each position.

  This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
  on CPU and GPU devices.

  .. warning::

     This API may change in the future.

  In this operation, :math:`M` is the random matrix with a connection probability
  `conn_prob`, and at each connection the value is the same scalar `weight`.

  When ``transpose=True``, we perform an operation of :math:`Y=X@M^T`.

  .. note::

     Note that the just-in-time generated :math:`M` (`transpose=False`) is
     different from the generated :math:`M^T` (`transpose=True`).

     If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
     matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
     the speed compared with ``outdim_parallel=False``.

  Parameters
  ----------
  vector: Array, ndarray
    The vector.
  weight: float
    The value of the random matrix.
  conn_prob: float
    The connection probability.
  shape: tuple of int
    The matrix shape.
  seed: int
    The random number generation seed.
  transpose: bool
    Transpose the random matrix or not.
  outdim_parallel: bool
    Perform the parallel random generations along the out dimension or not.
    It can be used to set the just-in-time generated :math:M^T: is the same
    as the just-in-time generated :math:`M` when ``transpose=True``.

  Returns
  -------
  out: Array, ndarray
    The output of :math:`Y = X @ M`.
  """
  vector = transform_brainpy_array(vector)
  weight = transform_brainpy_array(weight)
  if np.ndim(vector) != 1:
    raise ValueError('vector should be a 1D vector.')
  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if transpose:
    if vector.shape[0] != shape[0]:
      raise ValueError(f'Shape mismatch, vec ({vector.shape[0]},) @ mat {shape}.')
  else:
    if vector.shape[0] != shape[1]:
      raise ValueError(f'Shape mismatch, mat {shape} @ vec ({vector.shape[0]},).')
  if seed is None:
    seed = int(np.random.randint(0, int(1e8)))
  r = matvec_prob_homo_p.bind(vector,
                              conn_prob=conn_prob,
                              shape=shape,
                              seed=seed,
                              transpose=transpose,
                              outdim_parallel=outdim_parallel)[0]
  weight = jnp.asarray(weight, dtype=r.dtype)
  return r * weight


def matmat_prob_conn_uniform_weight(
    matrix: jax.Array,
    *,
    w_low: float,
    w_high: float,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    version: str = 'v1'
) -> jnp.ndarray:
  r"""Perform the :math:`y=M@v` operation,
  where :math:`M` is just-in-time randomly generated with a uniform distribution for its value.

  This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
  on CPU and GPU devices.

  .. warning::

     This API may change in the future.

  In this operation, :math:`M` is the random matrix with a connection probability
  `conn_prob`, and at each connection the value is the same scalar `weight`.

  When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

  .. note::

     Note that the just-in-time generated :math:`M` (`transpose=False`) is
     different from the generated :math:`M^T` (`transpose=True`).

     If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
     matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
     the speed compared with ``outdim_parallel=False``.

  Parameters
  ----------
  matrix: Array
    The matrix :math:`X`.
  w_low: float
    Lower boundary of the output interval.
  w_high: float
    Upper boundary of the output interval.
  conn_prob: float
    The connection probability.
  shape: tuple of int
    The matrix shape.
  seed: int
    The random number generation seed.

  Returns
  -------
  out: Array, ndarray
    The output of :math:`y = M @ v`.
  """
  matrix = transform_brainpy_array(matrix)
  assert w_high > w_low
  if np.ndim(matrix) != 2:
    raise ValueError('matrix should be a 2D vector.')
  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if matrix.shape[1] != shape[0]:
    raise ValueError(f'Shape mismatch, X {matrix.shape} @ M ({shape},).')
  if seed is None:
    raise ValueError('Seed cannot be none.')
  return matmat_prob_uniform_p.bind(matrix,
                                    w_low=w_low,
                                    w_high=w_high,
                                    conn_prob=conn_prob,
                                    shape=shape,
                                    seed=seed,
                                    version=version)[0]


def matmat_prob_conn_normal_weight(
    matrix: jax.Array,
    *,
    w_mu: float,
    w_sigma: float,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    version: str = 'v1'
) -> jax.Array:
  r"""Perform the :math:`Y=X@M` operation, where :math:`X`, :math:`Y` and :math:`M` are matrices,
  and :math:`M` is just-in-time randomly generated with a normal distribution for its value.

  This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
  on CPU and GPU devices.

  .. warning::

     This API may change in the future.

  In this operation, :math:`M` is the random matrix with a connection probability
  `conn_prob`, and at each connection the value is the same scalar `weight`.

  Parameters
  ----------
  matrix: Array
    The matrix :math:`X`.
  w_mu: float
    Mean (centre) of the distribution.
  w_sigma: float
    Standard deviation (spread or “width”) of the distribution. Must be non-negative.
  conn_prob: float
    The connection probability.
  shape: tuple of int
    The matrix shape.
  seed: int
    The random number generation seed.

  Returns
  -------
  out: Array, ndarray
    The output of :math:`y = M @ v`.
  """
  matrix = transform_brainpy_array(matrix)
  if np.ndim(matrix) != 2:
    raise ValueError('vector should be a 2D matrix.')
  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if matrix.shape[1] != shape[0]:
    raise ValueError(f'Shape mismatch, X {matrix.shape} @ M ({shape},).')
  if seed is None:
    raise ValueError('Seed cannot be none.')
  return matmat_prob_normal_p.bind(matrix,
                                   w_mu=w_mu,
                                   w_sigma=w_sigma,
                                   conn_prob=conn_prob,
                                   shape=shape,
                                   seed=seed,
                                   version=version)[0]


def _matvec_prob_homo_abstract(
    vector, *, conn_prob, shape, seed, transpose, outdim_parallel, version='v2'
):
  out = ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                    shape=(shape[1] if transpose else shape[0],))
  return [out]


def _matvec_prob_homo_cpu_translation(
    c, vector, *, conn_prob, shape, seed, transpose, outdim_parallel, version
):
  log_p = float(np.log((1 - conn_prob) if (conn_prob < 1) else 1e-40))
  n_row, n_col = (shape[1], shape[0]) if transpose else shape

  vec_shape = c.get_shape(vector)
  out_dtype = vec_shape.element_type()
  out_type = b'_float' if out_dtype == jnp.float32 else b'_double'

  if outdim_parallel:
    fn = b'cpu_matvec_prob_homo' + out_type
  else:
    fn = b'cpu_matvec_atomic_prob_homo' + out_type
  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(vector,
              xla_client.ops.ConstantLiteral(c, log_p),
              xla_client.ops.ConstantLiteral(c, seed),
              xla_client.ops.ConstantLiteral(c, n_row),
              xla_client.ops.ConstantLiteral(c, n_col)),
    operand_shapes_with_layout=(c.get_shape(vector),
                                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ())),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
  )


def _matvec_prob_homo_gpu_translation(
    c, vector, *, conn_prob, shape, seed, transpose, outdim_parallel, version
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(matvec_prob_homo_p.name)

  out_dtype = dtypes.canonicalize_dtype(float)
  type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  if version == 'v1':
    version_name = b'_v1'
    opaque = gpu_ops.build_jitconn_prob_homo_descriptor(shape[1] if transpose else shape[0],
                                                        shape[0] if transpose else shape[1],
                                                        seed,
                                                        conn_prob)
  elif version == 'v2':
    version_name = b'_v2'
    opaque = gpu_ops.build_jitconn_prob_homo_descriptor(shape[1] if transpose else shape[0],
                                                        shape[0] if transpose else shape[1],
                                                        seed,
                                                        float(np.log((1 - conn_prob) if conn_prob < 1 else 1e-40)))
  else:
    raise ValueError

  if outdim_parallel:
    fn = b'gpu_matvec_prob_homo' + version_name + type_name
  else:
    fn = b'gpu_matvec_atomic_prob_homo' + version_name + type_name
  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(vector,),
    operand_shapes_with_layout=(c.get_shape(vector),),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
    opaque=opaque,
  )


def _matvec_prob_homo_jvp(
    primals, tangents, *, conn_prob, shape, seed, transpose, outdim_parallel, version
):
  vector, = primals
  vector_dot, = tangents
  r = matvec_prob_homo_p.bind(vector,
                              conn_prob=conn_prob,
                              shape=shape,
                              seed=seed,
                              transpose=transpose,
                              outdim_parallel=outdim_parallel,
                              version=version)
  r_dot = matvec_prob_homo_p.bind(vector_dot,
                                  conn_prob=conn_prob,
                                  shape=shape,
                                  seed=seed,
                                  transpose=transpose,
                                  outdim_parallel=outdim_parallel,
                                  version=version)
  return r, r_dot


def _matvec_prob_homo_transpose(
    ct, vector, *, conn_prob, shape, seed, transpose, outdim_parallel, version
):
  return matvec_prob_homo_p.bind(ct[0],
                                 conn_prob=conn_prob,
                                 seed=seed,
                                 shape=shape,
                                 transpose=not transpose,
                                 outdim_parallel=not outdim_parallel,
                                 version=version)


matvec_prob_homo_p = Primitive('matvec_prob_homo')
matvec_prob_homo_p.multiple_results = True
matvec_prob_homo_p.def_abstract_eval(_matvec_prob_homo_abstract)
matvec_prob_homo_p.def_impl(partial(xla.apply_primitive, matvec_prob_homo_p))
xla.backend_specific_translations['cpu'][matvec_prob_homo_p] = _matvec_prob_homo_cpu_translation
xla.backend_specific_translations['gpu'][matvec_prob_homo_p] = _matvec_prob_homo_gpu_translation
register_general_batching(matvec_prob_homo_p)
ad.primitive_jvps[matvec_prob_homo_p] = _matvec_prob_homo_jvp
ad.primitive_transposes[matvec_prob_homo_p] = _matvec_prob_homo_transpose


def _matvec_prob_uniform_abstract(
    matrix, *, w_low, w_high, conn_prob, shape, seed, version
):
  out = ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                    shape=(matrix.shape[0], shape[1]))
  return [out]


def _matmat_prob_uniform_gpu_translation(
    c, matrix, *, w_low, w_high, conn_prob, shape, seed, version
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(matmat_prob_uniform_p.name)

  matrix_info = c.get_shape(matrix)
  dtype = matrix_info.element_type()
  matrix_shape = matrix_info.dimensions()
  type_name = b'_float' if dtype == jnp.float32 else b'_double'
  if version == 'v1':
    version_name = b'_v1'
  else:
    raise ValueError
  opaque = gpu_ops.build_matmat_jit_prob_descriptor1(
    matrix_shape[0],
    shape[0],
    shape[1],
    seed,
    float(np.log((1 - conn_prob) if conn_prob < 1 else 1e-40)),
    w_low,
    w_high - w_low
  )
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'gpu_matmat_prob_prob_uniform' + type_name + version_name,
    operands=(matrix,),
    operand_shapes_with_layout=(c.get_shape(matrix),),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (xla_client.Shape.array_shape(dtype, (matrix_shape[0], shape[1]), (1, 0)),)
    ),
    opaque=opaque,
  )


matmat_prob_uniform_p = Primitive('matmat_prob_uniform')
matmat_prob_uniform_p.multiple_results = True
matmat_prob_uniform_p.def_abstract_eval(_matvec_prob_uniform_abstract)
matmat_prob_uniform_p.def_impl(partial(xla.apply_primitive, matmat_prob_uniform_p))
xla.backend_specific_translations['gpu'][matmat_prob_uniform_p] = _matmat_prob_uniform_gpu_translation
register_general_batching(matmat_prob_uniform_p)


def _matmat_prob_normal_abstract(
    matrix, *, w_mu, w_sigma, conn_prob, shape, seed, version
):
  out = ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                    shape=(matrix.shape[0], shape[1]))
  return [out]


def _matvec_prob_normal_gpu_translation(
    c, matrix, *, w_mu, w_sigma, conn_prob, shape, seed, version
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(matmat_prob_normal_p.name)

  matrix_info = c.get_shape(matrix)
  dtype = matrix_info.element_type()
  matrix_shape = matrix_info.dimensions()
  type_name = b'_float' if dtype == jnp.float32 else b'_double'
  if version == 'v1':
    version_name = b'_v1'
  elif version == 'v2':
    version_name = b'_v2'
  elif version == 'v3':
    version_name = b'_v3'
  else:
    raise
  opaque = gpu_ops.build_matmat_jit_prob_descriptor1(
    matrix_shape[0],
    shape[0],
    shape[1],
    seed,
    float(np.log((1 - conn_prob) if conn_prob < 1 else 1e-40)),
    w_mu,
    w_sigma
  )
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'gpu_matmat_prob_prob_normal' + type_name + version_name,
    operands=(matrix,),
    operand_shapes_with_layout=(c.get_shape(matrix),),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (xla_client.Shape.array_shape(dtype, (matrix_shape[0], shape[1]), (1, 0)),)
    ),
    opaque=opaque,
  )


matmat_prob_normal_p = Primitive('matmat_prob_normal')
matmat_prob_normal_p.multiple_results = True
matmat_prob_normal_p.def_abstract_eval(_matmat_prob_normal_abstract)
matmat_prob_normal_p.def_impl(partial(xla.apply_primitive, matmat_prob_normal_p))
xla.backend_specific_translations['gpu'][matmat_prob_normal_p] = _matvec_prob_normal_gpu_translation
register_general_batching(matmat_prob_normal_p)
