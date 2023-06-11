# -*- coding: utf-8 -*-


import math
from functools import partial
from typing import Tuple, Optional, Union

import jax
import numpy as np
from jax import numpy as jnp, dtypes
from jax.core import ShapedArray, Primitive
from jax.interpreters import xla, ad
from jax.lib import xla_client

from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array, _get_dtype
from brainpy._src.math.op_registers import register_general_batching
from brainpy.errors import GPUOperatorNotFound, MathError

try:
  from brainpylib import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  'mv_prob_homo',
  'mv_prob_uniform',
  'mv_prob_normal',
]


def mv_prob_homo(
    vector: Union[Array, jax.Array],
    weight: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  r"""Perform the :math:`y=M@v` operation,
  where :math:`M` is just-in-time randomly generated with a scalar `weight` at each position.

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
    The output of :math:`y = M @ v`.
  """
  vector = as_jax(vector)
  weight = jnp.atleast_1d(as_jax(weight))
  conn_prob = jnp.atleast_1d(as_jax(conn_prob))
  clen = jnp.asarray(jnp.ceil(1 / conn_prob) * 2 - 1, dtype=jnp.int32)
  with jax.ensure_compile_time_eval():
    if seed is None:
      seed = int(np.random.randint(0, int(1e8)))
  seed = jnp.atleast_1d(as_jax(seed))
  return mv_prob_homo_p.bind(vector,
                             weight,
                             clen,
                             seed,
                             shape=shape,
                             transpose=transpose,
                             outdim_parallel=outdim_parallel,
                             )[0]


def mv_prob_uniform(
    vector: jax.Array,
    w_low: float,
    w_high: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
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
  vector: Array, ndarray
    The vector.
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
  transpose: bool
    Transpose the random matrix or not.
  outdim_parallel: bool
    Perform the parallel random generations along the out dimension or not.
    It can be used to set the just-in-time generated :math:M^T: is the same
    as the just-in-time generated :math:`M` when ``transpose=True``.

  Returns
  -------
  out: Array, ndarray
    The output of :math:`y = M @ v`.
  """
  vector = as_jax(vector)
  w_low = jnp.atleast_1d(as_jax(w_low))
  w_high = jnp.atleast_1d(as_jax(w_high))
  conn_prob = jnp.atleast_1d(as_jax(conn_prob))
  clen = jnp.asarray(jnp.ceil(1 / conn_prob) * 2 - 1, dtype=jnp.int32)
  with jax.ensure_compile_time_eval():
    if seed is None:
      seed = int(np.random.randint(0, int(1e8)))
  seed = jnp.atleast_1d(as_jax(seed))
  return mv_prob_uniform_p.bind(vector,
                                w_low,
                                w_high,
                                clen,
                                seed,
                                shape=shape,
                                transpose=transpose,
                                outdim_parallel=outdim_parallel)[0]


def mv_prob_normal(
    vector: jax.Array,
    w_mu: float,
    w_sigma: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  r"""Perform the :math:`y=M@v` operation,
  where :math:`M` is just-in-time randomly generated with a normal distribution for its value.

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
  vector: Array, ndarray
    The vector.
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
  transpose: bool
    Transpose the random matrix or not.
  outdim_parallel: bool
    Perform the parallel random generations along the out dimension or not.
    It can be used to set the just-in-time generated :math:M^T: is the same
    as the just-in-time generated :math:`M` when ``transpose=True``.

  Returns
  -------
  out: Array, ndarray
    The output of :math:`y = M @ v`.
  """
  vector = as_jax(vector)
  w_mu = jnp.atleast_1d(as_jax(w_mu))
  w_sigma = jnp.atleast_1d(as_jax(w_sigma))
  conn_prob = jnp.atleast_1d(as_jax(conn_prob))
  clen = jnp.asarray(jnp.ceil(1 / conn_prob) * 2 - 1, dtype=jnp.int32)
  with jax.ensure_compile_time_eval():
    if seed is None:
      seed = int(np.random.randint(0, int(1e8)))
  seed = jnp.atleast_1d(as_jax(seed))
  return mv_prob_normal_p.bind(vector,
                               w_mu,
                               w_sigma,
                               clen,
                               seed,
                               shape=shape,
                               transpose=transpose,
                               outdim_parallel=outdim_parallel)[0]


def _matvec_prob_homo_abstract(
    vector, weight, clen, seed, *, shape, transpose, outdim_parallel
):
  assert _get_dtype(vector) in [jnp.float32, jnp.float64]
  assert _get_dtype(weight) in [jnp.float32, jnp.float64], '"weight" must be float valued.'
  assert _get_dtype(clen) in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]
  assert _get_dtype(seed) in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]

  if vector.ndim != 1:
    raise ValueError('vector should be a 1D vector.')
  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if seed.ndim != 1:
    raise ValueError('seed must be a 1D scalar.')
  if clen.ndim != 1:
    raise ValueError('conn_prob must be a 1D scalar.')
  if weight.ndim != 1:
    raise ValueError('weight must be a 1D scalar.')

  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if not isinstance(outdim_parallel, bool):
    raise ValueError('outdim_parallel must be boolean value.')
  if not isinstance(transpose, bool):
    raise ValueError('transpose must be boolean value.')
  if transpose:
    if vector.shape[0] != shape[0]:
      raise ValueError(f'Shape mismatch, vec ({vector.shape[0]},) @ mat {shape}.')
  else:
    if vector.shape[0] != shape[1]:
      raise ValueError(f'Shape mismatch, mat {shape} @ vec ({vector.shape[0]},).')
  out = ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                    shape=(shape[1] if transpose else shape[0],))
  return [out]


def _matvec_prob_homo_cpu_translation(
    c, vector, weight, clen, seed, *, shape, transpose, outdim_parallel
):
  n_row, n_col = (shape[1], shape[0]) if transpose else shape

  vec_shape = c.get_shape(vector)
  out_dtype = vec_shape.element_type()
  if out_dtype == jnp.float32:
    out_type = b'_float'
  elif out_dtype == jnp.float64:
    out_type = b'_double'
  else:
    raise TypeError

  if outdim_parallel:
    fn = b'cpu_matvec_prob_homo' + out_type
  else:
    fn = b'cpu_matvec_atomic_prob_homo' + out_type
  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(vector,
              weight,
              clen,
              seed,
              xla_client.ops.ConstantLiteral(c, n_row),
              xla_client.ops.ConstantLiteral(c, n_col)),
    operand_shapes_with_layout=(c.get_shape(vector),
                                c.get_shape(weight),
                                c.get_shape(clen),
                                c.get_shape(seed),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ())),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
  )


def _matvec_prob_homo_gpu_translation(
    c, vector, weight, clen, seed, *, shape, transpose, outdim_parallel
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(mv_prob_homo_p.name)

  vec_shape = c.get_shape(vector)
  out_dtype = vec_shape.element_type()
  if out_dtype == jnp.float32:
    type_name = b'_float'
  elif out_dtype == jnp.float64:
    type_name = b'_double'
  else:
    raise TypeError

  opaque = gpu_ops.build_double_size_descriptor(shape[1] if transpose else shape[0],
                                                shape[0] if transpose else shape[1])

  if outdim_parallel:
    fn = b'gpu_matvec_prob_homo_v2' + type_name
  else:
    fn = b'gpu_matvec_atomic_prob_homo_v2' + type_name
  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(vector, weight, clen, seed),
    operand_shapes_with_layout=(c.get_shape(vector),
                                c.get_shape(weight),
                                c.get_shape(clen),
                                c.get_shape(seed)),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
    opaque=opaque,
  )


def _matvec_prob_homo_jvp(
    primals, tangents, *, shape, transpose, outdim_parallel
):
  vector, weight, clen, seed = primals
  vector_dot, weight_dot, clen_dot, seed_dot = tangents
  r = mv_prob_homo_p.bind(vector,
                          weight,
                          clen,
                          seed,
                          shape=shape,
                          transpose=transpose,
                          outdim_parallel=outdim_parallel)
  assert type(clen_dot) is ad.Zero
  assert type(seed_dot) is ad.Zero
  if type(weight_dot) is ad.Zero:
    if type(vector_dot) is ad.Zero:
      raise ValueError
    r_dot = mv_prob_homo_p.bind(vector_dot,
                                weight,
                                clen,
                                seed,
                                shape=shape,
                                transpose=transpose,
                                outdim_parallel=outdim_parallel)
  elif type(vector_dot) is ad.Zero:
    r_dot = mv_prob_homo_p.bind(vector,
                                weight_dot,
                                clen,
                                seed,
                                shape=shape,
                                transpose=transpose,
                                outdim_parallel=outdim_parallel)
  else:
    r_dot = mv_prob_homo_p.bind(vector_dot,
                                weight_dot,
                                clen,
                                seed,
                                shape=shape,
                                transpose=transpose,
                                outdim_parallel=outdim_parallel)

  return r, r_dot


def _matvec_prob_homo_transpose(
    ct, vector, weight, clen, seed, *, shape, transpose, outdim_parallel
):
  assert type(weight) is not ad.UndefinedPrimal
  assert type(clen) is not ad.UndefinedPrimal
  assert type(seed) is not ad.UndefinedPrimal
  assert type(vector) is ad.UndefinedPrimal
  r = mv_prob_homo_p.bind(ct[0],
                          weight,
                          clen,
                          seed,
                          shape=shape,
                          transpose=not transpose,
                          outdim_parallel=not outdim_parallel)[0]
  return r, weight, clen, seed


mv_prob_homo_p = Primitive('matvec_prob_homo')
mv_prob_homo_p.multiple_results = True
mv_prob_homo_p.def_abstract_eval(_matvec_prob_homo_abstract)
mv_prob_homo_p.def_impl(partial(xla.apply_primitive, mv_prob_homo_p))
xla.backend_specific_translations['cpu'][mv_prob_homo_p] = _matvec_prob_homo_cpu_translation
xla.backend_specific_translations['gpu'][mv_prob_homo_p] = _matvec_prob_homo_gpu_translation
register_general_batching(mv_prob_homo_p)
ad.primitive_jvps[mv_prob_homo_p] = _matvec_prob_homo_jvp
ad.primitive_transposes[mv_prob_homo_p] = _matvec_prob_homo_transpose


def _matvec_prob_uniform_abstract(
    vector, w_low, w_high, clen, seed, *, shape, transpose, outdim_parallel
):
  assert _get_dtype(vector) in [jnp.float32, jnp.float64]
  _w_low_dtype = _get_dtype(w_low)
  _w_high_dtype = _get_dtype(w_low)
  assert _w_low_dtype == _w_high_dtype, '"w_low" and "w_high" must be same typed.'
  assert _w_low_dtype in [jnp.float32, jnp.float64], '"w_low" must be float valued.'
  assert _w_high_dtype in [jnp.float32, jnp.float64], '"w_high" must be float valued.'
  assert _get_dtype(clen) in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]
  assert _get_dtype(seed) in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]

  if vector.ndim != 1:
    raise ValueError('vector should be a 1D vector.')
  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if w_low.ndim != 1:
    raise ValueError('w_low must be a 1D scalar.')
  if w_high.ndim != 1:
    raise ValueError('w_high must be a 1D scalar.')
  if clen.ndim != 1:
    raise ValueError('clen must be a 1D scalar.')
  if seed.ndim != 1:
    raise ValueError('seed must be a 1D scalar.')

  if not isinstance(transpose, bool):
    raise ValueError('transpose must be a boolean value.')
  if not isinstance(outdim_parallel, bool):
    raise ValueError('outdim_parallel must be a boolean value.')
  assert w_low.dtype == w_high.dtype == vector.dtype

  out = ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                    shape=(shape[1] if transpose else shape[0],))
  return [out]


def _matvec_prob_uniform_cpu_translation(
    c, vector, w_low, w_high, clen, seed, *, shape, transpose, outdim_parallel
):
  n_row, n_col = (shape[1], shape[0]) if transpose else shape

  vec_shape = c.get_shape(vector)
  out_dtype = vec_shape.element_type()

  if out_dtype == jnp.float32:
    type_name = b'_float'
  elif out_dtype == jnp.float64:
    type_name = b'_double'
  else:
    raise TypeError

  if outdim_parallel:
    fn = b'cpu_matvec_prob_uniform' + type_name
  else:
    fn = b'cpu_matvec_atomic_prob_uniform' + type_name

  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(vector,
              w_low,
              w_high,
              clen,
              seed,
              xla_client.ops.ConstantLiteral(c, n_row),
              xla_client.ops.ConstantLiteral(c, n_col)),
    operand_shapes_with_layout=(c.get_shape(vector),
                                c.get_shape(w_low),
                                c.get_shape(w_high),
                                c.get_shape(clen),
                                c.get_shape(seed),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ())),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
  )


def _matvec_prob_uniform_gpu_translation(
    c, vector, w_low, w_high, clen, seed, *, shape, transpose, outdim_parallel
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(mv_prob_homo_p.name)

  vec_shape = c.get_shape(vector)
  out_dtype = vec_shape.element_type()
  if out_dtype == jnp.float32:
    type_name = b'_float'
  elif out_dtype == jnp.float64:
    type_name = b'_double'
  else:
    raise TypeError(f'Only support float or double, while got {out_dtype}')

  opaque = gpu_ops.build_double_size_descriptor(shape[1] if transpose else shape[0],
                                                shape[0] if transpose else shape[1])

  if outdim_parallel:
    fn = b'gpu_matvec_prob_uniform_v2' + type_name
  else:
    fn = b'gpu_matvec_atomic_prob_uniform_v2' + type_name

  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(vector, w_low, w_high, clen, seed),
    operand_shapes_with_layout=(c.get_shape(vector),
                                c.get_shape(w_low),
                                c.get_shape(w_high),
                                c.get_shape(clen),
                                c.get_shape(seed),),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
    opaque=opaque,
  )


def _matvec_prob_uniform_jvp(
    primals, tangents, *, shape, transpose, outdim_parallel
):
  vector, w_low, w_high, clen, seed = primals
  vector_dot, w_low_dot, w_high_dot, clen_dot, seed_dot = tangents
  r = mv_prob_uniform_p.bind(vector,
                             w_low,
                             w_high,
                             clen,
                             seed,
                             shape=shape,
                             transpose=transpose,
                             outdim_parallel=outdim_parallel)
  assert type(w_low_dot) is ad.Zero
  assert type(w_high_dot) is ad.Zero
  assert type(clen_dot) is ad.Zero
  assert type(seed_dot) is ad.Zero
  r_dot = mv_prob_uniform_p.bind(vector_dot,
                                 w_low,
                                 w_high,
                                 clen,
                                 seed,
                                 shape=shape,
                                 transpose=transpose,
                                 outdim_parallel=outdim_parallel)
  return r, r_dot


def _matvec_prob_uniform_transpose(
    ct, vector, w_low, w_high, clen, seed, *, shape, transpose, outdim_parallel
):
  assert type(vector) is ad.UndefinedPrimal
  assert type(w_low) is not ad.UndefinedPrimal
  assert type(w_high) is not ad.UndefinedPrimal
  assert type(clen) is not ad.UndefinedPrimal
  assert type(seed) is not ad.UndefinedPrimal

  r = mv_prob_uniform_p.bind(ct[0],
                             w_low,
                             w_high,
                             clen,
                             seed,
                             shape=shape,
                             transpose=not transpose,
                             outdim_parallel=not outdim_parallel)[0]
  return r, w_low, w_high, clen, seed


mv_prob_uniform_p = Primitive('matvec_prob_uniform')
mv_prob_uniform_p.multiple_results = True
mv_prob_uniform_p.def_abstract_eval(_matvec_prob_uniform_abstract)
mv_prob_uniform_p.def_impl(partial(xla.apply_primitive, mv_prob_uniform_p))
xla.backend_specific_translations['cpu'][mv_prob_uniform_p] = _matvec_prob_uniform_cpu_translation
xla.backend_specific_translations['gpu'][mv_prob_uniform_p] = _matvec_prob_uniform_gpu_translation
register_general_batching(mv_prob_uniform_p)
ad.primitive_jvps[mv_prob_uniform_p] = _matvec_prob_uniform_jvp
ad.primitive_transposes[mv_prob_uniform_p] = _matvec_prob_uniform_transpose


def _matvec_prob_normal_abstract(
    vector, w_mu, w_sigma, clen, seed, *, shape, transpose, outdim_parallel
):
  assert _get_dtype(vector) in [jnp.float32, jnp.float64]
  _w_mu_dtype = _get_dtype(w_mu)
  _w_sigma_dtype = _get_dtype(w_sigma)
  assert _w_mu_dtype == _w_sigma_dtype, '"w_mu" and "w_sigma" must be same typed.'
  assert _w_mu_dtype in [jnp.float32, jnp.float64], '"w_mu" must be float valued.'
  assert _w_sigma_dtype in [jnp.float32, jnp.float64], '"w_sigma" must be float valued.'
  assert _get_dtype(clen) in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]
  assert _get_dtype(seed) in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]

  if w_mu.ndim != 1:
    raise ValueError('w_mu should be a 1D scalar.')
  if w_sigma.ndim != 1:
    raise ValueError('w_sigma should be a 1D scalar.')
  if clen.ndim != 1:
    raise ValueError('clen should be a 1D scalar.')
  if vector.ndim != 1:
    raise ValueError('vector should be a 1D vector.')
  if seed.ndim != 1:
    raise ValueError('seed must be a 1D scalar.')

  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if not isinstance(transpose, bool):
    raise ValueError('transpose must be a boolean value.')
  if not isinstance(outdim_parallel, bool):
    raise ValueError('outdim_parallel must be a boolean value.')

  out = ShapedArray(dtype=dtypes.canonicalize_dtype(float),
                    shape=(shape[1] if transpose else shape[0],))
  return [out]


def _matvec_prob_normal_cpu_translation(
    c, vector, w_mu, w_sigma, clen, seed, *, shape, transpose, outdim_parallel
):
  n_row, n_col = (shape[1], shape[0]) if transpose else shape

  vec_shape = c.get_shape(vector)
  out_dtype = vec_shape.element_type()

  if out_dtype == jnp.float32:
    type_name = b'_float'
  elif out_dtype == jnp.float64:
    type_name = b'_double'
  else:
    raise TypeError

  if outdim_parallel:
    fn = b'cpu_matvec_prob_normal' + type_name
  else:
    fn = b'cpu_matvec_atomic_prob_normal' + type_name

  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(vector,
              w_mu,
              w_sigma,
              clen,
              seed,
              xla_client.ops.ConstantLiteral(c, n_row),
              xla_client.ops.ConstantLiteral(c, n_col)),
    operand_shapes_with_layout=(c.get_shape(vector),
                                c.get_shape(w_mu),
                                c.get_shape(w_sigma),
                                c.get_shape(clen),
                                c.get_shape(seed),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ())),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
  )


def _matvec_prob_normal_gpu_translation(
    c, vector, w_mu, w_sigma, clen, seed, *, shape, transpose, outdim_parallel
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(mv_prob_homo_p.name)

  event_shape = c.get_shape(vector)
  out_dtype = event_shape.element_type()

  if out_dtype == jnp.float32:
    type_name = b'_float'
  elif out_dtype == jnp.float64:
    type_name = b'_double'
  else:
    raise TypeError(f'Only support float or double, while got {out_dtype}')
  opaque = gpu_ops.build_double_size_descriptor(shape[1] if transpose else shape[0],
                                                shape[0] if transpose else shape[1])

  if outdim_parallel:
    fn = b'gpu_matvec_prob_normal_v2' + type_name
  else:
    fn = b'gpu_matvec_atomic_prob_normal_v2' + type_name

  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(vector,
              w_mu,
              w_sigma,
              clen,
              seed,),
    operand_shapes_with_layout=(c.get_shape(vector),
                                c.get_shape(w_mu),
                                c.get_shape(w_sigma),
                                c.get_shape(clen),
                                c.get_shape(seed),),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
    opaque=opaque,
  )


def _matvec_prob_normal_jvp(
    primals, tangents, *, shape, transpose, outdim_parallel
):
  vector, w_mu, w_sigma, clen, seed = primals
  vector_dot, w_mu_dot, w_sigma_dot, clen_dot, seed_dot = tangents
  r = mv_prob_normal_p.bind(vector,
                            w_mu,
                            w_sigma,
                            clen,
                            seed,
                            shape=shape,
                            transpose=transpose,
                            outdim_parallel=outdim_parallel)
  assert type(w_mu_dot) is ad.Zero
  assert type(w_sigma_dot) is ad.Zero
  assert type(clen_dot) is ad.Zero
  assert type(seed_dot) is ad.Zero
  r_dot = mv_prob_normal_p.bind(vector_dot,
                                w_mu,
                                w_sigma,
                                clen,
                                seed,
                                shape=shape,
                                transpose=transpose,
                                outdim_parallel=outdim_parallel)
  return r, r_dot


def _matvec_prob_normal_transpose(
    ct, vector, w_mu, w_sigma, clen, seed, *, shape, transpose, outdim_parallel
):
  assert type(vector) is ad.UndefinedPrimal
  assert type(w_mu) is not ad.UndefinedPrimal
  assert type(w_sigma) is not ad.UndefinedPrimal
  assert type(clen) is not ad.UndefinedPrimal
  assert type(seed) is not ad.UndefinedPrimal

  r = mv_prob_normal_p.bind(ct[0],
                            w_mu,
                            w_sigma,
                            clen,
                            seed,
                            shape=shape,
                            transpose=not transpose,
                            outdim_parallel=not outdim_parallel)[0]
  return r, w_mu, w_sigma, clen, seed


mv_prob_normal_p = Primitive('matvec_prob_normal')
mv_prob_normal_p.multiple_results = True
mv_prob_normal_p.def_abstract_eval(_matvec_prob_normal_abstract)
mv_prob_normal_p.def_impl(partial(xla.apply_primitive, mv_prob_normal_p))
xla.backend_specific_translations['cpu'][mv_prob_normal_p] = _matvec_prob_normal_cpu_translation
xla.backend_specific_translations['gpu'][mv_prob_normal_p] = _matvec_prob_normal_gpu_translation
register_general_batching(mv_prob_normal_p)
ad.primitive_jvps[mv_prob_normal_p] = _matvec_prob_normal_jvp
ad.primitive_transposes[mv_prob_normal_p] = _matvec_prob_normal_transpose
