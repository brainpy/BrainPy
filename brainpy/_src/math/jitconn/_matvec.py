# -*- coding: utf-8 -*-


from functools import partial
from typing import Tuple, Optional, Union

import jax
import numpy as np
from jax import numpy as jnp, dtypes
from jax.core import ShapedArray, Primitive
from jax.interpreters import xla, ad
from jax.lib import xla_client

from brainpy._src.dependency_check import import_brainpylib_gpu_ops, import_brainpylib_cpu_ops, import_taichi
from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array, _get_dtype
from brainpy._src.math.op_register import register_general_batching, XLACustomOp
from brainpy._src.math.tifunc import (lfsr88_key, lfsr88_random_integers, lfsr88_uniform, lfsr88_normal)
from brainpy.errors import GPUOperatorNotFound

ti = import_taichi()

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
  return mv_prob_homo_taichi(vector, weight, conn_prob, seed, shape=shape, transpose=transpose,
                             outdim_parallel=outdim_parallel)


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
  return mv_prob_uniform_taichi(vector, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose,
                                outdim_parallel=outdim_parallel)


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
  return mv_prob_uniform_taichi(vector, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose,
                                outdim_parallel=outdim_parallel)


### BRAINYPLIB ###

def mv_prob_homo_brainpylib(
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
  seed = jnp.atleast_1d(as_jax(seed, dtype=jnp.int32))
  return mv_prob_homo_p.bind(vector,
                             weight,
                             clen,
                             seed,
                             shape=shape,
                             transpose=transpose,
                             outdim_parallel=outdim_parallel,
                             )[0]


def mv_prob_uniform_brainpylib(
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
  seed = jnp.atleast_1d(as_jax(seed, dtype=jnp.int32))
  return mv_prob_uniform_p.bind(vector,
                                w_low,
                                w_high,
                                clen,
                                seed,
                                shape=shape,
                                transpose=transpose,
                                outdim_parallel=outdim_parallel)[0]


def mv_prob_normal_brainpylib(
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
  seed = jnp.atleast_1d(as_jax(seed, dtype=jnp.int32))
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
  import_brainpylib_cpu_ops()
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
  gpu_ops = import_brainpylib_gpu_ops()
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
    fn = b'gpu_jit_csrmv_prob_homo_v2' + type_name
  else:
    fn = b'gpu_jit_csrmv_atomic_prob_homo_v2' + type_name
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
  import_brainpylib_cpu_ops()
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
  gpu_ops = import_brainpylib_gpu_ops()
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
    fn = b'gpu_jit_csrmv_prob_uniform_v2' + type_name
  else:
    fn = b'gpu_jit_csrmv_atomic_prob_uniform_v2' + type_name

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
  import_brainpylib_cpu_ops()
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
  gpu_ops = import_brainpylib_gpu_ops()
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
    fn = b'gpu_jit_csrmv_prob_normal_v2' + type_name
  else:
    fn = b'gpu_jit_csrmv_atomic_prob_normal_v2' + type_name

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


### TAICHI ###
def mv_prob_homo_taichi(
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

      Generally, the :math:`M` in ``f(outdim_parallel=True, transpose=False)`` is the same of
      the :math:`M^T` used in ``f(outdim_parallel=False, transpose=True)``.

      Similarly, the :math:`M^T` in ``f(outdim_parallel=True, transpose=True)`` is the same
      of the :math:`M` used in ``f(outdim_parallel=False, transpose=False)``.

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
  if isinstance(weight, float):
    weight = as_jax(weight, dtype=vector.dtype)
  weight = jnp.atleast_1d(as_jax(weight))
  conn_len = jnp.ceil(1 / conn_prob) * 2 - 1
  clen = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
  if seed is None:
    with jax.ensure_compile_time_eval():
      seed = np.random.randint(0, int(1e8), 1)
  seed = jnp.asarray(seed, dtype=jnp.uint32)
  seed = jnp.atleast_1d(seed)
  return raw_mv_prob_homo(vector, weight, clen, seed, shape=shape,
                          transpose=transpose, outdim_parallel=outdim_parallel)[0]


def mv_prob_uniform_taichi(
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
  if isinstance(w_low, float): w_low = as_jax(w_low, dtype=vector.dtype)
  if isinstance(w_high, float): w_high = as_jax(w_high, dtype=vector.dtype)
  w_low = jnp.atleast_1d(as_jax(w_low))
  w_high = jnp.atleast_1d(as_jax(w_high))
  conn_len = jnp.ceil(1 / conn_prob) * 2 - 1
  conn_len = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
  if seed is None:
    with jax.ensure_compile_time_eval():
      seed = np.random.randint(0, int(1e8), 1)
  seed = jnp.atleast_1d(jnp.asarray(seed, dtype=jnp.uint32))
  return raw_mv_prob_uniform(vector, w_low, w_high, conn_len, seed, shape=shape,
                             transpose=transpose, outdim_parallel=outdim_parallel)[0]


def mv_prob_normal_taichi(
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
  if isinstance(w_mu, float): w_mu = as_jax(w_mu, dtype=vector.dtype)
  if isinstance(w_sigma, float): w_sigma = as_jax(w_sigma, dtype=vector.dtype)
  w_mu = jnp.atleast_1d(as_jax(w_mu))
  w_sigma = jnp.atleast_1d(as_jax(w_sigma))
  conn_len = jnp.ceil(1 / conn_prob) * 2 - 1
  conn_len = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
  if seed is None:
    with jax.ensure_compile_time_eval():
      seed = np.random.randint(0, int(1e8), 1)
  seed = jnp.atleast_1d(jnp.asarray(seed, dtype=jnp.uint32))
  return raw_mv_prob_normal(vector, w_mu, w_sigma, conn_len, seed, shape=shape,
                            transpose=transpose, outdim_parallel=outdim_parallel)[0]


def _reverse(shape):
  return shape[::-1]


@ti.kernel
def _mv_prob_homo_cpu(
    vector: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
  num_row = out.shape[0]
  num_col = vector.shape[0]
  weight0 = weight[0]
  clen0 = clen[0]
  seed0 = seed[0]

  for i_col in range(num_col):
    key = lfsr88_key(seed0 + i_col)
    key, i_row = lfsr88_random_integers(key, 0, clen0 - 1)
    v = vector[i_col] * weight0
    while i_row < num_row:
      out[i_row] += v
      key, inc = lfsr88_random_integers(key, 1, clen0)
      i_row += inc


@ti.kernel
def _mv_prob_homo_outdim_parallel_cpu(
    vector: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
  num_row = out.shape[0]
  num_col = vector.shape[0]
  weight0 = weight[0]
  clen0 = clen[0]
  seed0 = seed[0]

  for i_row in range(num_row):
    r = 0.
    key = lfsr88_key(seed0 + i_row)
    key, i_col = lfsr88_random_integers(key, 0, clen0 - 1)
    while i_col < num_col:
      r += vector[i_col]
      key, inc = lfsr88_random_integers(key, 1, clen0)
      i_col += inc
    out[i_row] = r * weight0


@ti.kernel
def _mv_prob_homo_gpu(
    vector: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
  num_row = out.shape[0]
  num_col = vector.shape[0]
  weight0 = weight[0]
  clen0 = clen[0]
  seed0 = seed[0]
  step = ti.uint32(ti.max((num_row + 1) >> 5, 1))

  for i in range(num_col * 32):
    i_col = i >> 5
    index = i & 31
    col_v = vector[i_col]
    i_row = step * index - 1
    end = ti.min(i_row + step, num_row)
    key = lfsr88_key(seed0 + i)
    key, inc = lfsr88_random_integers(key, 1, clen0)
    i_row += inc
    while i_row < end:
      out[i_row] += weight0 * col_v
      key, inc = lfsr88_random_integers(key, 1, clen0)
      i_row += inc


@ti.kernel
def _mv_prob_homo_outdim_parallel_gpu(
    vector: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
  num_row = out.shape[0]
  num_col = vector.shape[0]
  weight0 = weight[0]
  clen0 = clen[0]
  seed0 = seed[0]
  step = ti.u32(ti.max((num_row + 1) >> 5, 1))

  for i in range(num_row * 32):
    i_row = i >> 5
    i_thread = i & 31
    i_col = step * i_thread - 1
    end_col = ti.min(i_col + step, num_col)
    r = 0.
    key = lfsr88_key(seed0 + i)
    key, inc = lfsr88_random_integers(key, 1, clen0)
    i_col += inc
    while i_col < end_col:
      r += vector[i_col]
      key, inc = lfsr88_random_integers(key, 1, clen0)
      i_col += inc
    out[i_row] += weight0 * r  # TODO: warp-level reduction


def _mv_prob_homo_jvp_vector(v_dot, vector, weight, clen, seed, *, outs, shape, transpose, outdim_parallel):
  shape = _reverse(shape) if transpose else shape
  return raw_mv_prob_homo(v_dot, weight, clen, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)


def _mv_prob_homo_jvp_weight(w_dot, vector, weight, clen, seed, *, outs, shape, transpose, outdim_parallel):
  shape = _reverse(shape) if transpose else shape
  return raw_mv_prob_homo(vector, w_dot, clen, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)


def _mv_prob_homo_transpose(
    ct, vector, weight, clen, seed, *, outs, shape, transpose, outdim_parallel
):
  shape = _reverse(shape) if transpose else shape
  if ad.is_undefined_primal(vector):
    if type(ct) is ad.Zero:
      return ad.Zero(vector), weight, clen, seed
    else:
      dv = raw_mv_prob_homo(ct[0], weight, clen, seed, shape=shape,
                            transpose=not transpose, outdim_parallel=not outdim_parallel)[0]
      return dv, weight, clen, seed
  elif ad.is_undefined_primal(weight):
    if type(ct) is ad.Zero:
      return vector, ad.Zero(weight), clen, seed
    else:
      row = raw_mv_prob_homo(ct[0], jnp.ones(1, dtype=ct[0].dtype), clen, seed,
                             shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)[0]
      dw = jnp.sum(row * vector, keepdims=True)
      return vector, dw, clen, seed
  else:
    assert type(clen) is not ad.UndefinedPrimal, 'Cannot differentiate through clen.'
    assert type(seed) is not ad.UndefinedPrimal, 'Cannot differentiate through seed.'


def _general_checking(vector, clen, seed, shape, outdim_parallel, transpose, *weights):
  if vector.ndim != 1:
    raise ValueError('vector should be a 1D vector.')
  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if seed.ndim != 1:
    raise ValueError('seed must be a 1D scalar.')
  if clen.ndim != 1:
    raise ValueError('conn_prob must be a 1D scalar.')

  assert _get_dtype(clen) in [jnp.int16, jnp.int32, jnp.int64, jnp.uint16, jnp.uint32, jnp.uint64]
  assert _get_dtype(seed) in [jnp.int16, jnp.int32, jnp.int64, jnp.uint16, jnp.uint32, jnp.uint64]

  for weight in weights:
    if weight.ndim != 1:
      raise ValueError('weight must be a 1D scalar.')
    assert _get_dtype(weight) in [jnp.float16, jnp.float32, jnp.float64], '"weight" must be float valued.'

  if not isinstance(outdim_parallel, bool):
    raise ValueError('outdim_parallel must be boolean value.')
  if not isinstance(transpose, bool):
    raise ValueError('transpose must be boolean value.')

  if transpose:
    out_shape = (shape[1],)
    if vector.shape[0] != shape[0]:
      raise ValueError(f'Shape mismatch, vec {vector.shape} @ mat {shape}.')
    shape = _reverse(shape)
  else:
    if vector.shape[0] != shape[1]:
      raise ValueError(f'Shape mismatch, mat {shape} @ vec ({vector.shape[0]},).')
    out_shape = (shape[0],)

  return shape, out_shape


def _non_event_checking(vector, clen, seed, shape, outdim_parallel, transpose, *weights):
  assert _get_dtype(vector) in [jnp.float16, jnp.float32, jnp.float64]
  return _general_checking(vector, clen, seed, shape, outdim_parallel, transpose, *weights)


def raw_mv_prob_homo(
    vector: jax.Array,
    weight: jax.Array,  # vector with size 1
    clen: jax.Array,  # vector with size 1
    seed: jax.Array,  # vector with size 1
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  mat_shape, out_shape = _non_event_checking(vector, clen, seed, shape, outdim_parallel, transpose, weight)

  if outdim_parallel:
    prim = _mv_prob_homo_outdim_parallel_p
  else:
    prim = _mv_prob_homo_p

  return prim(vector,
              weight,
              clen,
              seed,
              outs=[jax.ShapeDtypeStruct(shape=out_shape, dtype=vector.dtype)],
              shape=mat_shape,
              transpose=transpose,
              outdim_parallel=outdim_parallel)


def _define_mv_prob_homo_prim(cpu_kernel, gpu_kernel):
  prim = XLACustomOp(cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel)
  prim.defjvp(_mv_prob_homo_jvp_vector, _mv_prob_homo_jvp_weight, None, None)
  prim.def_transpose_rule(_mv_prob_homo_transpose)
  return prim


# outdim_parallel = True
_mv_prob_homo_outdim_parallel_p = _define_mv_prob_homo_prim(cpu_kernel=_mv_prob_homo_outdim_parallel_cpu,
                                                            gpu_kernel=_mv_prob_homo_outdim_parallel_gpu)

# outdim_parallel = False
_mv_prob_homo_p = _define_mv_prob_homo_prim(cpu_kernel=_mv_prob_homo_cpu,
                                            gpu_kernel=_mv_prob_homo_gpu)


@ti.kernel
def _mv_prob_uniform_cpu(
    vector: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
  num_row = out.shape[0]
  num_col = vector.shape[0]
  w_min0 = w_min[0]
  w_max0 = w_max[0]
  clen0 = clen[0]
  seed0 = seed[0]

  for i_col in range(num_col):
    col_v = vector[i_col]
    key = lfsr88_key(seed0 + i_col)
    key, i_row = lfsr88_random_integers(key, 0, clen0 - 1)
    while i_row < num_row:
      key, raw_v = lfsr88_uniform(key, w_min0, w_max0)
      out[i_row] += col_v * raw_v
      key, inc = lfsr88_random_integers(key, 1, clen0)
      i_row += inc


@ti.kernel
def _mv_prob_uniform_outdim_parallel_cpu(
    vector: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
  num_row = out.shape[0]
  num_col = vector.shape[0]
  w_min0 = w_min[0]
  w_max0 = w_max[0]
  clen0 = clen[0]
  seed0 = seed[0]

  for i_row in range(num_row):
    r = 0.
    key = lfsr88_key(seed0 + i_row)
    key, i_col = lfsr88_random_integers(key, 0, clen0 - 1)
    while i_col < num_col:
      key, raw_v = lfsr88_uniform(key, w_min0, w_max0)
      r += vector[i_col] * raw_v
      key, inc = lfsr88_random_integers(key, 1, clen0)
      i_col += inc
    out[i_row] = r


@ti.kernel
def _mv_prob_uniform_gpu(
    vector: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
  num_row = out.shape[0]
  num_col = vector.shape[0]
  w_min0 = w_min[0]
  w_max0 = w_max[0]
  clen0 = clen[0]
  seed0 = seed[0]
  step = ti.uint32(ti.max((num_row + 1) >> 5, 1))

  for i in range(num_col * 32):
    i_col = i >> 5
    index = i & 31
    col_v = vector[i_col]
    i_row = step * index - 1
    end = ti.min(i_row + step, num_row)
    key = lfsr88_key(seed0 + i)
    key, inc = lfsr88_random_integers(key, 1, clen0)
    i_row += inc
    while i_row < end:
      key, row_v = lfsr88_uniform(key, w_min0, w_max0)
      out[i_row] += row_v * col_v
      key, inc = lfsr88_random_integers(key, 1, clen0)
      i_row += inc


@ti.kernel
def _mv_prob_uniform_outdim_parallel_gpu(
    vector: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
  num_row = out.shape[0]
  num_col = vector.shape[0]
  w_min0 = w_min[0]
  w_max0 = w_max[0]
  clen0 = clen[0]
  seed0 = seed[0]
  step = ti.u32(ti.max((num_row + 1) >> 5, 1))

  for i in range(num_row * 32):
    i_row = i >> 5
    i_thread = i & 31
    i_col = step * i_thread - 1
    end_col = ti.min(i_col + step, num_col)
    r = 0.
    key = lfsr88_key(seed0 + i)
    key, inc = lfsr88_random_integers(key, 1, clen0)
    i_col += inc
    while i_col < end_col:
      key, row_v = lfsr88_uniform(key, w_min0, w_max0)
      r += vector[i_col] * row_v
      key, inc = lfsr88_random_integers(key, 1, clen0)
      i_col += inc
    out[i_row] += r  # TODO: warp-level reduction


def _mv_prob_uniform_jvp_vector(v_dot, vector, w_low, w_high, clen, seed, *,
                                outs, shape, transpose, outdim_parallel):
  shape = _reverse(shape) if transpose else shape
  return raw_mv_prob_uniform(v_dot, w_low, w_high, clen, seed, shape=shape,
                             transpose=transpose, outdim_parallel=outdim_parallel)


def _mv_prob_uniform_jvp_wlow(w_dot, vector, w_low, w_high, clen, seed, *,
                              outs, shape, transpose, outdim_parallel):
  shape = _reverse(shape) if transpose else shape
  return raw_mv_prob_uniform(vector, w_dot, w_high, clen, seed, shape=shape,
                             transpose=transpose, outdim_parallel=outdim_parallel)


def _mv_prob_uniform_jvp_whigh(w_dot, vector, w_low, w_high, clen, seed, *,
                               outs, shape, transpose, outdim_parallel):
  shape = _reverse(shape) if transpose else shape
  return raw_mv_prob_uniform(vector, w_low, w_dot, clen, seed, shape=shape,
                             transpose=transpose, outdim_parallel=outdim_parallel)


def _mv_prob_uniform_transpose(
    ct, vector, w_low, w_high, clen, seed, *, outs, shape, transpose, outdim_parallel
):
  shape = _reverse(shape) if transpose else shape
  if ad.is_undefined_primal(vector):
    if type(ct) is ad.Zero:
      return ad.Zero(vector), w_low, w_high, clen, seed
    else:
      dv = raw_mv_prob_uniform(ct[0], w_low, w_high, clen, seed, shape=shape,
                               transpose=not transpose, outdim_parallel=not outdim_parallel)[0]
      return dv, w_low, w_high, clen, seed
  else:
    assert type(w_low) is not ad.UndefinedPrimal, 'Cannot differentiate through w_low.'
    assert type(w_high) is not ad.UndefinedPrimal, 'Cannot differentiate through w_high.'
    assert type(clen) is not ad.UndefinedPrimal, 'Cannot differentiate through clen.'
    assert type(seed) is not ad.UndefinedPrimal, 'Cannot differentiate through seed.'


def raw_mv_prob_uniform(
    vector: jax.Array,
    w_low: jax.Array,
    w_high: jax.Array,
    conn_len: jax.Array,
    seed: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  mat_shape, out_shape = _non_event_checking(vector, conn_len, seed, shape, outdim_parallel, transpose, w_low, w_high)

  if outdim_parallel:
    prim = _mv_prob_uniform_outdim_parallel_p
  else:
    prim = _mv_prob_uniform_p

  return prim(vector,
              w_low,
              w_high,
              conn_len,
              seed,
              outs=[jax.ShapeDtypeStruct(shape=out_shape, dtype=vector.dtype)],
              shape=mat_shape,
              transpose=transpose,
              outdim_parallel=outdim_parallel)


def _define_mv_prob_uniform_prim(cpu_kernel, gpu_kernel):
  prim = XLACustomOp(cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel)
  prim.defjvp(_mv_prob_uniform_jvp_vector,
              _mv_prob_uniform_jvp_wlow,
              _mv_prob_uniform_jvp_whigh,
              None,
              None)
  prim.def_transpose_rule(_mv_prob_uniform_transpose)
  return prim


# outdim_parallel = True
_mv_prob_uniform_outdim_parallel_p = _define_mv_prob_uniform_prim(
  cpu_kernel=_mv_prob_uniform_outdim_parallel_cpu,
  gpu_kernel=_mv_prob_uniform_outdim_parallel_gpu
)

# outdim_parallel = False
_mv_prob_uniform_p = _define_mv_prob_uniform_prim(
  cpu_kernel=_mv_prob_uniform_cpu,
  gpu_kernel=_mv_prob_uniform_gpu
)


@ti.kernel
def _mv_prob_normal_cpu(
    vector: ti.types.ndarray(ndim=1),
    w_mu: ti.types.ndarray(ndim=1),
    w_sigma: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
  num_row = out.shape[0]
  num_col = vector.shape[0]
  w_mu0 = w_mu[0]
  w_sigma0 = w_sigma[0]
  clen0 = clen[0]
  seed0 = seed[0]

  for i_col in range(num_col):
    col_v = vector[i_col]
    key = lfsr88_key(seed0 + i_col)
    key, i_row = lfsr88_random_integers(key, 0, clen0 - 1)
    while i_row < num_row:
      key, raw_v = lfsr88_normal(key, w_mu0, w_sigma0)
      out[i_row] += col_v * raw_v
      key, inc = lfsr88_random_integers(key, 1, clen0)
      i_row += inc


@ti.kernel
def _mv_prob_normal_outdim_parallel_cpu(
    vector: ti.types.ndarray(ndim=1),
    w_mu: ti.types.ndarray(ndim=1),
    w_sigma: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
  num_row = out.shape[0]
  num_col = vector.shape[0]
  w_mu0 = w_mu[0]
  w_sigma0 = w_sigma[0]
  clen0 = clen[0]
  seed0 = seed[0]

  for i_row in range(num_row):
    r = 0.
    key = lfsr88_key(seed0 + i_row)
    key, i_col = lfsr88_random_integers(key, 0, clen0 - 1)
    while i_col < num_col:
      key, raw_v = lfsr88_normal(key, w_mu0, w_sigma0)
      r += vector[i_col] * raw_v
      key, inc = lfsr88_random_integers(key, 1, clen0)
      i_col += inc
    out[i_row] = r


@ti.kernel
def _mv_prob_normal_gpu(
    vector: ti.types.ndarray(ndim=1),
    w_mu: ti.types.ndarray(ndim=1),
    w_sigma: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
  num_row = out.shape[0]
  num_col = vector.shape[0]
  w_mu0 = w_mu[0]
  w_sigma0 = w_sigma[0]
  clen0 = clen[0]
  seed0 = seed[0]
  step = ti.uint32(ti.max((num_row + 1) >> 5, 1))

  for i in range(num_col * 32):
    i_col = i >> 5
    index = i & 31
    col_v = vector[i_col]
    i_row = step * index - 1
    end = ti.min(i_row + step, num_row)
    key = lfsr88_key(seed0 + i)
    key, inc = lfsr88_random_integers(key, 1, clen0)
    i_row += inc
    while i_row < end:
      key, row_v = lfsr88_normal(key, w_mu0, w_sigma0)
      out[i_row] += row_v * col_v
      key, inc = lfsr88_random_integers(key, 1, clen0)
      i_row += inc


@ti.kernel
def _mv_prob_normal_outdim_parallel_gpu(
    vector: ti.types.ndarray(ndim=1),
    w_mu: ti.types.ndarray(ndim=1),
    w_sigma: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
  num_row = out.shape[0]
  num_col = vector.shape[0]
  w_mu0 = w_mu[0]
  w_sigma0 = w_sigma[0]
  clen0 = clen[0]
  seed0 = seed[0]
  step = ti.u32(ti.max((num_row + 1) >> 5, 1))

  for i in range(num_row * 32):
    i_row = i >> 5
    i_thread = i & 31
    i_col = step * i_thread - 1
    end_col = ti.min(i_col + step, num_col)
    r = 0.
    key = lfsr88_key(seed0 + i)
    key, inc = lfsr88_random_integers(key, 1, clen0)
    i_col += inc
    while i_col < end_col:
      key, row_v = lfsr88_normal(key, w_mu0, w_sigma0)
      r += vector[i_col] * row_v
      key, inc = lfsr88_random_integers(key, 1, clen0)
      i_col += inc
    out[i_row] += r  # TODO: warp-level reduction


def _mv_prob_normal_jvp_vector(v_dot, vector, w_mu, w_sigma, clen, seed, *, outs, shape, transpose, outdim_parallel):
  shape = _reverse(shape) if transpose else shape
  return raw_mv_prob_normal(v_dot, w_mu, w_sigma, clen, seed, shape=shape,
                            transpose=transpose, outdim_parallel=outdim_parallel)


def _mv_prob_normal_jvp_w_mu(w_dot, vector, w_mu, w_sigma, clen, seed, *, outs, shape, transpose, outdim_parallel):
  shape = _reverse(shape) if transpose else shape
  return raw_mv_prob_normal(vector, w_dot, w_sigma, clen, seed, shape=shape,
                            transpose=transpose, outdim_parallel=outdim_parallel)


def _mv_prob_normal_jvp_w_sigma(w_dot, vector, w_mu, w_sigma, clen, seed, *, outs, shape, transpose, outdim_parallel):
  shape = _reverse(shape) if transpose else shape
  return raw_mv_prob_normal(vector, w_mu, w_dot, clen, seed, shape=shape,
                            transpose=transpose, outdim_parallel=outdim_parallel)


def _mv_prob_normal_transpose(
    ct, vector, w_mu, w_sigma, clen, seed, *, outs, shape, transpose, outdim_parallel
):
  shape = _reverse(shape) if transpose else shape
  if ad.is_undefined_primal(vector):
    if type(ct) is ad.Zero:
      return ad.Zero(vector), w_mu, w_sigma, clen, seed
    else:
      dv = raw_mv_prob_normal(ct[0], w_mu, w_sigma, clen, seed, shape=shape,
                              transpose=not transpose, outdim_parallel=not outdim_parallel)[0]
      return dv, w_mu, w_sigma, clen, seed
  else:
    assert type(w_mu) is not ad.UndefinedPrimal, 'Cannot differentiate through w_mu.'
    assert type(w_sigma) is not ad.UndefinedPrimal, 'Cannot differentiate through w_sigma.'
    assert type(clen) is not ad.UndefinedPrimal, 'Cannot differentiate through clen.'
    assert type(seed) is not ad.UndefinedPrimal, 'Cannot differentiate through seed.'


def raw_mv_prob_normal(
    vector: jax.Array,
    w_mu: jax.Array,
    w_sigma: jax.Array,
    conn_len: jax.Array,
    seed: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  mat_shape, out_shape = _non_event_checking(vector, conn_len, seed, shape, outdim_parallel, transpose, w_mu, w_sigma)

  if outdim_parallel:
    prim = _mv_prob_normal_outdim_parallel_p
  else:
    prim = _mv_prob_normal_p

  return prim(vector,
              w_mu,
              w_sigma,
              conn_len,
              seed,
              outs=[jax.ShapeDtypeStruct(shape=out_shape, dtype=vector.dtype)],
              shape=mat_shape,
              transpose=transpose,
              outdim_parallel=outdim_parallel)


def _define_mv_prob_normal_prim(cpu_kernel, gpu_kernel):
  prim = XLACustomOp(cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel)
  prim.defjvp(_mv_prob_normal_jvp_vector,
              _mv_prob_normal_jvp_w_mu,
              _mv_prob_normal_jvp_w_sigma,
              None,
              None)
  prim.def_transpose_rule(_mv_prob_normal_transpose)
  return prim


# outdim_parallel = True
_mv_prob_normal_outdim_parallel_p = _define_mv_prob_normal_prim(
  cpu_kernel=_mv_prob_normal_outdim_parallel_cpu,
  gpu_kernel=_mv_prob_normal_outdim_parallel_gpu
)

# outdim_parallel = False
_mv_prob_normal_p = _define_mv_prob_normal_prim(
  cpu_kernel=_mv_prob_normal_cpu,
  gpu_kernel=_mv_prob_normal_gpu
)
