# -*- coding: utf-8 -*-

import warnings
from functools import partial
from typing import Tuple


import jax
import numpy as np
from brainpy._src.math.interoperability import as_jax
from jax import core, numpy as jnp
from jax import lax
from jax.interpreters import batching
from jax.interpreters import mlir, ad
from jax.tree_util import tree_flatten, tree_unflatten
from jaxlib import gpu_sparse
if jax.__version__ >= '0.5.0':
  from jax.extend.core import Primitive
else:
  from jax.core import Primitive

__all__ = [
  'coo_to_csr',
  'csr_to_coo',
  'csr_to_dense'
]

def _general_batching_rule(prim, args, axes, **kwargs):
  batch_axes, batch_args, non_batch_args = [], {}, {}
  for ax_i, ax in enumerate(axes):
    if ax is None:
      non_batch_args[f'ax{ax_i}'] = args[ax_i]
    else:
      batch_args[f'ax{ax_i}'] = args[ax_i] if ax == 0 else jnp.moveaxis(args[ax_i], ax, 0)
      batch_axes.append(ax_i)

  def f(_, x):
    pars = tuple([(x[f'ax{i}'] if i in batch_axes else non_batch_args[f'ax{i}'])
                  for i in range(len(axes))])
    return 0, prim.bind(*pars, **kwargs)

  _, outs = lax.scan(f, 0, batch_args)
  out_vals, out_tree = tree_flatten(outs)
  out_dim = tree_unflatten(out_tree, (0,) * len(out_vals))
  return outs, out_dim

def _register_general_batching(prim):
  batching.primitive_batchers[prim] = partial(_general_batching_rule, prim)


def coo_to_csr(
    pre_ids: jnp.ndarray,
    post_ids: jnp.ndarray,
    *,
    num_row: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """convert pre_ids, post_ids to (indices, indptr)."""
  pre_ids = as_jax(pre_ids)
  post_ids = as_jax(post_ids)

  # sorting
  sort_ids = jnp.argsort(pre_ids, kind='stable')
  post_ids = post_ids[sort_ids]

  indices = post_ids
  unique_pre_ids, pre_count = jnp.unique(pre_ids, return_counts=True)
  final_pre_count = jnp.zeros(num_row)
  final_pre_count[unique_pre_ids] = pre_count
  indptr = final_pre_count.cumsum()
  indptr = jnp.insert(indptr, 0, 0)
  return indices, indptr


def csr_to_coo(
    indices: jnp.ndarray,
    indptr: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Given CSR (indices, indptr) return COO (row, col)"""
  indices = as_jax(indices)
  indptr = as_jax(indptr)
  return jnp.cumsum(jnp.zeros_like(indices).at[indptr].add(1)) - 1, indices


def csr_to_csc():
  pass


def coo_to_dense(
    data: jnp.ndarray,
    rows: jnp.ndarray,
    cols: jnp.ndarray,
    *,
    shape: Tuple[int, int]
) -> jnp.ndarray:
  pass


def csr_to_dense(
    data: jnp.ndarray,
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    *,
    shape: Tuple[int, int]
) -> jnp.ndarray:
  data = as_jax(data)
  indices = as_jax(indices)
  indptr = as_jax(indptr)
  return csr_to_dense_p.bind(data, indices, indptr, shape=shape)


def _coo_extract(row, col, mat):
  """Extract values of dense matrix mat at given COO indices."""
  return mat[row, col]


def _csr_extract(indices, indptr, mat):
  """Extract values of dense matrix mat at given CSR indices."""
  return _coo_extract(*csr_to_coo(indices, indptr), mat)


def _coo_todense(data, row, col, *, shape):
  """Convert CSR-format sparse matrix to a dense matrix.

  Args:
    data : array of shape ``(nse,)``.
    row : array of shape ``(nse,)``
    col : array of shape ``(nse,)`` and dtype ``row.dtype``
    shape : COOInfo object containing matrix metadata

  Returns:
    mat : array with specified shape and dtype matching ``data``
  """
  return jnp.zeros(shape, data.dtype).at[row, col].add(data)


def _csr_to_dense_impl(data, indices, indptr, *, shape):
  return _coo_todense(data, *csr_to_coo(indices, indptr), shape=shape)


def _csr_to_dense_abstract_eval(data, indices, indptr, *, shape):
  assert data.ndim == indices.ndim == indptr.ndim == 1
  assert indices.dtype == indptr.dtype
  assert data.shape == indices.shape
  assert indptr.shape[0] == shape[0] + 1
  return core.ShapedArray(shape, data.dtype)


_csr_to_dense_lowering = mlir.lower_fun(_csr_to_dense_impl, multiple_results=False)


def _csr_to_dense_gpu_lowering(ctx, data, indices, indptr, *, shape):
  data_aval, indices_aval, _ = ctx.avals_in
  dtype = data_aval.dtype
  if not (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating)):
    warnings.warn(f"csr_todense cusparse/hipsparse lowering not available for dtype={dtype}. "
                  "Falling back to default implementation.",
                  UserWarning)
    return _csr_to_dense_lowering(ctx, data, indices, indptr, shape=shape)
  return [gpu_sparse.cuda_csr_todense(data, indices, indptr,
                                      shape=shape, data_dtype=dtype,
                                      index_dtype=indices_aval.dtype)]


def _csr_to_dense_jvp(data_dot, data, indices, indptr, *, shape):
  return csr_to_dense(data_dot, indices, indptr, shape=shape)


def _csr_to_dense_transpose(ct, data, indices, indptr, *, shape):
  # Note: we assume that transpose has the same sparsity pattern.
  # Can we check this?
  assert ad.is_undefined_primal(data)
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ct.shape == shape
  assert indices.aval.dtype == indptr.aval.dtype
  assert ct.dtype == data.aval.dtype
  return _csr_extract(indices, indptr, ct), indices, indptr


csr_to_dense_p = Primitive('csr_to_dense')
csr_to_dense_p.def_impl(_csr_to_dense_impl)
csr_to_dense_p.def_abstract_eval(_csr_to_dense_abstract_eval)
ad.defjvp(csr_to_dense_p, _csr_to_dense_jvp, None, None)
ad.primitive_transposes[csr_to_dense_p] = _csr_to_dense_transpose
mlir.register_lowering(csr_to_dense_p, _csr_to_dense_lowering)
_register_general_batching(csr_to_dense_p)
if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(csr_to_dense_p, _csr_to_dense_gpu_lowering, platform='cuda')
