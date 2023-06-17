# -*- coding: utf-8 -*-

from functools import partial
from typing import Union, Tuple

import jax.lax
import numba
import numpy as np
from jax import numpy as jnp
from jax.core import Primitive, ShapedArray
from jax.interpreters import ad, xla
from jax.lib import xla_client

from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.op_registers import (compile_cpu_signature_with_numba,
                                            register_general_batching)
from brainpy.errors import GPUOperatorNotFound

try:
  from brainpylib import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  'bcsrmm',
]


def get_mask(dense_b, blockshape, blockcount):
  mask = jnp.zeros(blockcount[0] * blockcount[1], dtype=jnp.bool_)

  for i in range(blockcount[1]):
    for j in range(blockcount[0]):
      if jnp.abs(dense_b[i * blockshape[1]: (i + 1) * blockshape[1],
                 j * blockshape[0]: (j + 1) * blockshape[0]]).sum() != 0:
        mask = mask.at[i * blockcount[0] + j].set(True)
  mask = mask.reshape(blockcount[1], blockcount[0])
  return mask


def get_mask_from_ptr_indices(ptr, indices, blockcount):
  mask = jnp.zeros((blockcount[1], blockcount[0]), dtype=jnp.bool_)
  for idx, indice in enumerate(indices):
    row_index = 0
    for ptr_ in ptr[1:]:
      if idx < ptr_:
        break
      row_index += 1
    mask = mask.at[row_index, indice].set(True)
  return mask


def get_data(dense_b, mask, blockshape, blockcount, n_blocks):
  data = jnp.zeros(
    shape=(n_blocks * blockshape[1], blockshape[0]),
    dtype=jnp.float32
  )

  assignment_count = 0
  for i in range(blockcount[1]):
    for j in range(blockcount[0]):
      if mask[i, j]:
        data = data.at[assignment_count * blockshape[1]: (assignment_count + 1) * blockshape[1],
               :].set(dense_b[i * blockshape[1]: (i + 1) * blockshape[1],
                      j * blockshape[0]: (j + 1) * blockshape[0]])
        assignment_count += 1
  return data


def get_ptr_indices(mask, blockcount, n_blocks, block_ptr=None):
  nnz = jnp.nonzero(mask)

  if block_ptr is None:
    block_ptr = jnp.arange(0, len(nnz[0]))

  indices = jnp.argsort(block_ptr)
  _ = jnp.take(block_ptr, indices)

  blocks = nnz[0][jnp.array(indices)], nnz[1][jnp.array(indices)]
  blocks = jnp.stack([nnz[0][jnp.array(indices)], nnz[1][jnp.array(indices)]], axis=-1).astype(
    dtype=jnp.int32
  )
  blocks = jnp.flip(blocks, axis=-1).flatten()

  X = blockcount[1]
  Y = blockcount[0]

  rows = nnz[0][:]
  cols = nnz[1][:]

  block_indices = jnp.zeros(X * Y, dtype=jnp.int32)
  positions = rows * Y + cols
  block_indices = block_indices.at[positions].set(block_ptr + 1)
  block_indices = block_indices.reshape(X, Y).transpose().reshape(X * Y)

  block_ptr = block_indices[jnp.nonzero(block_indices)[0]] - 1

  X, Y = Y, X
  rows = cols
  nnztt = jnp.nonzero(mask.transpose())
  cols = nnztt[:][1]

  rows.astype(jnp.int32)

  ptr_b = jnp.zeros((X + 1,), dtype=jnp.int32)
  for row in rows:
    ptr_b = ptr_b.at[row + 1].set(ptr_b[row + 1] + 1)
  ptr_b = ptr_b.cumsum(0).astype(dtype=jnp.int32)

  indices_b = jnp.stack([cols, block_ptr], axis=1).astype(dtype=jnp.int32)

  return ptr_b, indices_b


def get_dense(ptr, indices, data, shape, blockshape):
  mask = get_mask_from_ptr_indices(ptr, indices, blockshape)
  dense_data = jnp.zeros(shape, dtype=jnp.float32)
  mask_count = 0
  for i in range(mask.shape[1]):
    for j in range(mask.shape[0]):
      if mask[i, j]:
        dense_data = dense_data.at[
                     i * blockshape[0]: (i + 1) * blockshape[0],
                     j * blockshape[1]: (j + 1) * blockshape[1],
                     ].set(data[mask_count * blockshape[0]: (mask_count + 1) * blockshape[0], :])
        mask_count += 1
  return dense_data


def blocksparse_matmat_multiply(dense_a,
                                ptr_b=None,
                                indices_b=None,
                                data_b=None,
                                shape_b=None,
                                dense_b=None,
                                blockshape=(32, 32),
                                device='cpu'):
  if dense_b is not None:
    # m, n, k
    m = dense_a.shape[0]
    k = dense_a.shape[1]
    n = dense_b.shape[1]

    # blockcount
    blockcount = (n // blockshape[0], k // blockshape[1])

    # mask
    mask = get_mask(dense_b, blockshape, blockcount)

    # n_blocks
    n_blocks = mask.sum()

    # data_b
    data_b = get_data(dense_b, mask, blockshape, blockcount, n_blocks)

    # ptr_b, indices_b
    ptr_b, indices_b = get_ptr_indices(mask, blockcount, n_blocks)
  else:
    # m, n, k
    m = dense_a.shape[0]
    n = shape_b[1]
    k = dense_a.shape[1]

    # blockcount
    blockcount = (n // blockshape[0], k // blockshape[1])

    mask = get_mask_from_ptr_indices(ptr_b, indices_b, blockcount)

    n_blocks = mask.sum()

    ptr_b, indices_b = get_ptr_indices(mask, blockcount, n_blocks)

  # out
  # out = jnp.zeros((n, m))

  # verbose
  print('data_b: ', data_b)
  print('ptr:', ptr_b)
  print('indices:', indices_b)

  '''out = blocksparse_matmat_cpu_test(dense_a,
          ptr_b,
          indices_b,
          data_b,
          out,
          m=m,
          n=n,
          k=k,
          block_size_k=blockshape[0],
          block_size_n=blockshape[1])
  return out'''

  if device == 'cpu':
    out = bcsrmm(
      dense_a,
      ptr_b,
      indices_b,
      data_b,
      m=m,
      n=n,
      k=k,
      block_size_k=blockshape[0],
      block_size_n=blockshape[1],
    )
    return out
  elif device == 'gpu':
    out = bcsrmm(
      dense_a,
      ptr_b,
      indices_b,
      data_b,
      m=m,
      n=n,
      k=k,
      block_size_k=blockshape[0],
      block_size_n=blockshape[1],
    )
    return out.transpose()
  else:
    raise Exception('Invalid device: ', device)


def bcsrmm(
    A_data: jax.Array,
    B_data: jax.Array,
    B_indices: jax.Array,
    B_ptr: jax.Array,
    *,
    shape: Tuple[int, int],
    block_size: Tuple[int, int],
    transpose: bool = False,
    method: str = 'cutlass'
) -> jax.Array:
  """Perform the matrix multiplication :math:`C = A @ B` with BSR data structure.

  Args:
    A_data: The dense matrix :math:`A`.
    B_data: The data at each block of :math:`B`.
    B_indices: The sparse indices of :math:`B`.
    B_ptr: The connection pointer of :math:`B`.
    shape: a tuple of int, indicating the array shape of :math:`B`.
    block_size: a tuple of int, indicating the block size for portioning :math:`B`.
    transpose: boolean. If True, perform :math:`A @ B^T`; otherwise, perform :math:`A @ B`.
    method: a sting for denoting the BSR sparse computing method.

  Returns:
    The dense array :math:`C`.
  """
  A_data = as_jax(A_data)
  B_data = as_jax(B_data)
  B_indices = as_jax(B_indices)
  B_ptr = as_jax(B_ptr)
  assert A_data.shape[1] == shape[0]

  if method == 'cutlass':
    C = _bcsrmm_cutlass_p.bind(A_data,
                               B_data,
                               B_indices,
                               B_ptr,
                               m=A_data.shape[0],
                               k=shape[0],
                               n=shape[1],
                               transpose=transpose,
                               block_size_k=block_size[0],
                               block_size_n=block_size[1])[0]
    return C.T
  else:
    raise ValueError


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _bcsrmm_cutlass_imp_transpose(outs, ins):  # dense(m, k) @ bcsr(n, k) -> dense(n, m)
  res_val = outs[0]
  # B_data: (num_block, block_size_k, block_size_n)
  A_data, B_data, B_indices, B_inptr, m, k, n, block_size_k, block_size_n = ins
  block_size_k = block_size_k[()]
  block_size_n = block_size_n[()]
  n_block = n // block_size_n

  for ni in numba.prange(n_block):
    C_tmp = np.zeros((block_size_n, m), dtype=A_data.dtype)
    start, end = B_inptr[ni], B_inptr[ni + 1]
    ns = ni * block_size_n
    ne = ns + block_size_n
    for i in range(start, end):
      ki = B_indices[i, 0]
      ks = ki * block_size_k
      ke = ki + block_size_k
      bi = B_indices[i, 1]
      C_tmp += np.matmul(B_data[bi], A_data[:, ks: ke].T)
    res_val[ns: ne] = C_tmp
  return res_val


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _bcsrmm_cutlass_imp2(outs, ins):  # dense(m, k) @ bcsr(k, n) -> dense(n, m)
  res_val = outs[0]
  # B_data: (num_block, block_size_n, block_size_k)
  A_data, B_data, B_indices, B_inptr, m, k, n, block_size_k, block_size_n = ins
  block_size_k = block_size_k[()]
  block_size_n = block_size_n[()]
  n_block = n // block_size_n

  for ni in numba.prange(n_block):
    C_tmp = np.zeros((block_size_n, m), dtype=A_data.dtype)
    start, end = B_inptr[ni], B_inptr[ni + 1]
    ns = ni * block_size_n
    ne = ns + block_size_n
    for i in range(start, end):
      ki = B_indices[i, 0]
      ks = ki * block_size_k
      ke = ki + block_size_k
      bi = B_indices[i, 1]
      C_tmp += np.matmul(B_data[bi], A_data[:, ks: ke].T)
    res_val[ns: ne] = C_tmp
  return res_val


def _bcsrmm_cutlass_abstract(
    A_data, B_data, B_indices, B_ptr, *, m, k, n, block_size_k, block_size_n
):
  assert block_size_k == 32, 'cutlass based block-sparse mm only support block size (32, 32)'
  assert block_size_n == 32, 'cutlass based block-sparse mm only support block size (32, 32)'
  assert B_indices.shape[0] * block_size_n == B_data.shape[0]
  assert block_size_k == B_data.shape[1]
  assert A_data.shape[0] == m
  assert A_data.shape[1] == k
  assert A_data.dtype == B_data.dtype
  assert n // block_size_n + 1 == B_ptr.shape[0]
  return [ShapedArray(dtype=A_data.dtype, shape=(n, m))]


def _bcsrmm_cutlass_cpu_translation(
    c, A_data, B_data, B_indices, B_ptr, *,
    m, k, n, block_size_k, block_size_n
):
  inputs = (A_data, B_ptr, B_indices, B_data)
  description = dict(m=m,
                     n=n,
                     k=k,
                     block_size_k=block_size_k,
                     block_size_n=block_size_n)
  name, inputs, in_layouts, out_layouts = compile_cpu_signature_with_numba(
    c,
    _bcsrmm_cutlass_imp2,
    abs_eval_fn=_bcsrmm_cutlass_abstract,
    multiple_results=True,
    inputs=inputs,
    description=description
  )
  return xla_client.ops.CustomCallWithLayout(
    c, name,
    operands=inputs,
    operand_shapes_with_layout=in_layouts,
    shape_with_layout=out_layouts,
  )


def _bcsrmm_cutlass_gpu_translation(c, A_data, B_data, B_indices, B_ptr, *, m, k, n, block_size_k, block_size_n):
  if gpu_ops is None:
    raise GPUOperatorNotFound(_bcsrmm_cutlass_p.name)

  matrix_info = c.get_shape(A_data)
  dtype = matrix_info.element_type()

  opaque = gpu_ops.build_blocksparse_format_descriptor(m,
                                                       n,
                                                       k,
                                                       block_size_k,
                                                       block_size_n)

  fn = b'gpu_blocksparse_matmat'

  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(A_data, B_ptr, B_indices, B_data,),
    operand_shapes_with_layout=(c.get_shape(A_data),
                                c.get_shape(B_ptr),
                                c.get_shape(B_indices),
                                c.get_shape(B_data),),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (xla_client.Shape.array_shape(dtype, (n, m), (1, 0)),)
    ),
    opaque=opaque
  )


def _bcsrmm_cutlass_jvp_dense_a(dense_a_dot, A_data, B_ptr, B_indices, B_data, *, m, n, k, block_size_k,
                                block_size_n):
  return bcsrmm(dense_a_dot, B_ptr, B_indices, B_data, m=m, n=n, k=k, block_size_k=block_size_k,
                block_size_n=block_size_n)


def _bcsrmm_cutlass_jvp_data_b(data_b_dot, A_data, B_ptr, B_indices, B_data, *, m, n, k, block_size_k,
                               block_size_n):
  return bcsrmm(A_data, B_ptr, B_indices, data_b_dot, m=m, n=n, k=k, block_size_k=block_size_k,
                block_size_n=block_size_n)


def _bcsrmm_cutlass_jvp_transpose():
  # TODO: implement
  pass


_bcsrmm_cutlass_p = Primitive('bcsrmm_cutlass_pim')
_bcsrmm_cutlass_p.multiple_results = True
_bcsrmm_cutlass_p.def_abstract_eval(_bcsrmm_cutlass_abstract)
_bcsrmm_cutlass_p.def_impl(partial(xla.apply_primitive, _bcsrmm_cutlass_p))
xla.backend_specific_translations['cpu'][_bcsrmm_cutlass_p] = _bcsrmm_cutlass_cpu_translation
xla.backend_specific_translations['gpu'][_bcsrmm_cutlass_p] = _bcsrmm_cutlass_gpu_translation
ad.primitive_jvps[_bcsrmm_cutlass_p] = _bcsrmm_cutlass_jvp_transpose
ad.primitive_transposes[_bcsrmm_cutlass_p] = _bcsrmm_cutlass_jvp_transpose
register_general_batching(bcsrmm)


def _blocksparse_matmat_back_abstract(
    A_data, B_data, blocks, *, m, n, k, transpose, block_size_k, block_size_n, blocks_len
):
  shape = (n, k)
  dtype = A_data.dtype
  out = ShapedArray(dtype=dtype, shape=shape)
  return [out]


def _blocksparse_matmat_back_gpu_translation(
    c, A_data, B_data, blocks, *, m, n, k, transpose, block_size_k, block_size_n, blocks_len
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(_bcsrmm_cutlass_back_p.name)
  matrix_info = c.get_shape(A_data)
  dtype = matrix_info.element_type()

  opaque = gpu_ops.build_blocksparse_back_format_descriptor(m,
                                                            n,
                                                            k,
                                                            block_size_k,
                                                            block_size_n,
                                                            blocks_len)

  fn = b'gpu_blocksparse_matmat_back'

  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(A_data, B_data, blocks,),
    operand_shape_with_layout=(c.get_shape(A_data),
                               c.get_shape(B_data),
                               c.get_shape(blocks),),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (xla_client.Shape.array_shape(dtype, (k, n), (1, 0)),)
    ),
    opaque=opaque
  )


_bcsrmm_cutlass_back_p = Primitive('bcsrmm_cutlass_back_prim')
_bcsrmm_cutlass_back_p.multiple_results = True
_bcsrmm_cutlass_back_p.def_abstract_eval(_blocksparse_matmat_back_abstract)
_bcsrmm_cutlass_back_p.def_impl(partial(xla.apply_primitive, _bcsrmm_cutlass_back_p))
xla.backend_specific_translations['gpu'][_bcsrmm_cutlass_back_p] = _blocksparse_matmat_back_gpu_translation
register_general_batching(_bcsrmm_cutlass_back_p)
