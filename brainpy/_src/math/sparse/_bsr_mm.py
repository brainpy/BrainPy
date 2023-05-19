# -*- coding: utf-8 -*-

import warnings
from functools import partial
from typing import Union, Tuple

import jax.lax
import numba
import numpy as np
from jax import core, numpy as jnp, dtypes, default_backend, random
from jax.interpreters import ad, mlir, xla
from jax.lib import xla_client
from jax.core import Primitive, ShapedArray
from jaxlib import gpu_sparse

from brainpy._src.math.op_registers import (compile_cpu_signature_with_numba,
                                         register_general_batching)
from brainpy._src.math.sparse.utils import csr_to_coo
from brainpy._src.math.interoperability import as_jax
from brainpy.errors import GPUOperatorNotFound

import brainpylib as bl

try:
    from brainpylib import gpu_ops
except ImportError:
    gpu_ops = None

__all__ = [
    'blocksparse_matmat',
    'blocksparse_matmat_back'
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
    for position in positions:
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
        n = dense_b.shape[1]
        k = dense_a.shape[1]

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
            block_size_rows_b=blockshape[0],
            block_size_cols_b=blockshape[1])
    return out'''

    if device == 'cpu':
        out = blocksparse_matmat(
            dense_a,
            ptr_b,
            indices_b,
            data_b,
            m=m,
            n=n,
            k=k,
            block_size_rows_b=blockshape[0],
            block_size_cols_b=blockshape[1],
        )
        return out
    elif device == 'gpu':
        out = blocksparse_matmat(
            dense_a,
            ptr_b,
            indices_b,
            data_b,
            m=m,
            n=n,
            k=k,
            block_size_rows_b=blockshape[0],
            block_size_cols_b=blockshape[1],
        )
        return out.transpose()
    else:
        raise Exception('Invalid device: ', device)

def blocksparse_matmat(
        A_data: jnp.ndarray,
        B_ptr: jnp.ndarray,
        B_indices: jnp.ndarray,
        B_data: jnp.ndarray,
        *,
        m: int,
        n: int,
        k: int,
        block_size_rows_b: int,
        block_size_cols_b: int,
) -> jax.Array:
    A_data = as_jax(A_data)
    B_ptr = as_jax(B_ptr)
    B_indices = as_jax(B_indices)
    B_data = as_jax(B_data)
    return blocksparse_matmat_p.bind(A_data,
                                     B_ptr,
                                     B_indices,
                                     B_data,
                                     m=m,
                                     n=n,
                                     k=k,
                                     block_size_rows_b=block_size_rows_b,
                                     block_size_cols_b=block_size_cols_b)[0]

'''def blocksparse_matmat_cpu_test(
            A_data,
            B_ptr,
            B_indices,
            B_data,
            m,
            n,
            k,
            block_size_rows_b,
            block_size_cols_b):
    res_val = np.zeros((m, n))
    # index[0]为index, index[1]为该index对应的block的index
    for idx, index in enumerate(B_indices):
        # find the column
        row_index = 0
        for ptr in B_ptr[1:]:
            if ptr > idx:
                break
            row_index += 1
        row_start = row_index * block_size_cols_b
        # find the row
        col_start = index[0] * block_size_rows_b
        # calculate the value and add to the res_val
        for i in range(block_size_rows_b):
            for j in range(block_size_cols_b):
                if B_data[index[1] * block_size_rows_b + i, j] == 0:
                    continue
                row_now = row_start + j
                for c_m in range(m):
                    print('c{c_col}{c_row} = a{a_col}{a_row} * b{b_col}{b_row}'.format(c_col=c_m + 1,c_row=row_now + 1,
                                                                                     a_col=c_m + 1, a_row=col_start + i + 1,
                                                                                     b_col=col_start + i + 1, b_row=row_start + j + 1))
                    res_val[c_m, row_now] += A_data[c_m, col_start + i] * B_data[index[1] * block_size_rows_b + i, j]
                # res_val[:, row_now + j] += A_data[:, row_now + j] * B_data[index[1] * block_size_rows_b + i, j]

    return res_val'''

# CPU implement
@numba.njit(fastmath=True, parallel=True, nogil=True)
def _blocksparse_matmat_numba_imp(outs, ins):
    res_val = outs[0]
    res_val.transpose()
    res_val.fill(0)
    A_data, B_ptr, B_indices, B_data, m, n, k, block_size_rows_b, block_size_cols_b = ins
    m = np.int32(m)
    n = np.int32(n)
    k = np.int32(k)
    block_size_rows_b = np.int32(block_size_rows_b)
    block_size_cols_b = np.int32(block_size_cols_b)

    # index[0]为index, index[1]为该index对应的block的index
    for idx, index in enumerate(B_indices):
        # find the column
        row_index = 0
        for ptr in B_ptr[1:]:
            if ptr > idx:
                break
            row_index += 1
        row_start = row_index * block_size_cols_b
        # find the row
        col_start = index[0] * block_size_rows_b
        # calculate the value and add to the res_val
        for i in range(block_size_rows_b):
            for j in range(block_size_cols_b):
                if B_data[index[1] * block_size_rows_b + i, j] == 0:
                    continue
                row_now = row_start + j
                col_now = col_start + i
                res_val[:, row_now] += A_data[:, col_now] * B_data[index[1] * block_size_rows_b + i, j]
                '''for c_m in range(m):
                    print('c{c_col}{c_row} = a{a_col}{a_row} * b{b_col}{b_row}'.format(c_col=c_m + 1,c_row=row_now + 1,
                                                                                     a_col=c_m + 1, a_row=col_start + i + 1,
                                                                                     b_col=col_start + i + 1, b_row=row_start + j + 1))
                    res_val[c_m, row_now] += A_data[c_m, col_start + i] * B_data[index[1] * block_size_rows_b + i, j]'''
                # res_val[:, row_now + j] += A_data[:, row_now + j] * B_data[index[1] * block_size_rows_b + i, j]

    return res_val


def _blocksparse_matmat_cpu_translation(c, A_data, B_ptr, B_indices, B_data, *, m, n, k, block_size_rows_b,
                                        block_size_cols_b):
    inputs = (A_data, B_ptr, B_indices, B_data)
    description = dict(m=m,
                       n=n,
                       k=k,
                       block_size_rows_b=block_size_rows_b,
                       block_size_cols_b=block_size_cols_b)
    name, inputs, in_layouts, out_layouts = compile_cpu_signature_with_numba(
        c,
        _blocksparse_matmat_numba_imp,
        abs_eval_fn=_blocksparse_matmat_abstract,
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


def _blocksparse_matmat_abstract(
        A_data, B_ptr, B_indices, B_data, *, m, n, k, block_size_rows_b, block_size_cols_b
):
    shape = (n, m)
    dtype = A_data.dtype
    out = ShapedArray(dtype=dtype, shape=shape)
    return [out]


def _blocksparse_matmat_gpu_translation(
        c, A_data, B_ptr, B_indices, B_data, *, m, n, k, block_size_rows_b, block_size_cols_b
):
    if gpu_ops is None:
        raise GPUOperatorNotFound(blocksparse_matmat_p.name)

    matrix_info = c.get_shape(A_data)
    dtype = matrix_info.element_type()

    opaque = gpu_ops.build_blocksparse_format_descriptor(m,
                                                         n,
                                                         k,
                                                         block_size_rows_b,
                                                         block_size_cols_b)

    fn = b'gpu_blocksparse_matmat'

    return xla_client.ops.CustomCallWithLayout(
        c,
        fn,
        operands=(A_data, B_ptr, B_indices, B_data,),
        operand_shapes_with_layout=(c.get_shape(A_data),
                                    c.get_shape(B_ptr),
                                    c.get_shape(B_indices),
                                    c.get_shape(B_data), ),
        shape_with_layout=xla_client.Shape.tuple_shape(
            (xla_client.Shape.array_shape(dtype, (m, n), (1, 0)),)
        ),
        opaque=opaque
    )

def _blocksparse_matmat_jvp_dense_a(dense_a_dot, A_data, B_ptr, B_indices, B_data, *, m, n, k, block_size_rows_b, block_size_cols_b):
    return blocksparse_matmat(dense_a_dot, B_ptr, B_indices, B_data, m=m, n=n, k=k, block_size_rows_b=block_size_rows_b, block_size_cols_b=block_size_cols_b)

def _blocksparse_matmat_jvp_data_b(data_b_dot, A_data, B_ptr, B_indices, B_data, *, m, n, k, block_size_rows_b, block_size_cols_b):
    return blocksparse_matmat(A_data, B_ptr, B_indices, data_b_dot, m=m, n=n, k=k, block_size_rows_b=block_size_rows_b, block_size_cols_b=block_size_cols_b)

def _blocksparse_matmat_jvp_transpose():
    # TODO: implement
    pass

blocksparse_matmat_p = Primitive('gpu_blocksparse_matmat')
blocksparse_matmat_p.multiple_results = True
blocksparse_matmat_p.def_abstract_eval(_blocksparse_matmat_abstract)
blocksparse_matmat_p.def_impl(partial(xla.apply_primitive, blocksparse_matmat_p))
xla.backend_specific_translations['cpu'][blocksparse_matmat_p] = _blocksparse_matmat_cpu_translation
xla.backend_specific_translations['gpu'][blocksparse_matmat_p] = _blocksparse_matmat_gpu_translation
ad.defjvp(blocksparse_matmat_p, _blocksparse_matmat_jvp_dense_a, None, None, _blocksparse_matmat_jvp_data_b)
ad.primitive_jvps[blocksparse_matmat_p] = _blocksparse_matmat_jvp_transpose
register_general_batching(blocksparse_matmat)


def blocksparse_matmat_back(
        A_data: jnp.ndarray,
        B_data: jnp.ndarray,
        blocks: jnp.ndarray,
        *,
        m: int,
        n: int,
        k: int,
        block_size_rows_b: int,
        block_size_cols_b: int,
        blocks_len: int,
) -> jax.Array:
    A_data = as_jax(A_data)
    B_data = as_jax(B_data)
    blocks = as_jax(blocks)
    return blocksparse_matmat_back_p.bind(A_data,
                                          B_data,
                                          blocks,
                                          m = m,
                                          n = n,
                                          k = k,
                                          block_size_rows_b = block_size_rows_b,
                                          block_size_cols_b = block_size_cols_b,
                                          blocks_len = blocks_len)[0]

def _blocksparse_matmat_back_abstract(
        A_data, B_data, blocks, *, m, n, k, block_size_rows_b, block_size_cols_b,blocks_len
):
    shape = (n, k)
    dtype = A_data.dtype
    out = ShapedArray(dtype=dtype, shape=shape)
    return  [out]


def _blocksparse_matmat_back_gpu_translation(
        c, A_data, B_data, blocks, *, m, n, k, block_size_rows_b, block_size_cols_b,blocks_len
):
    if gpu_ops is None:
        raise GPUOperatorNotFound(blocksparse_matmat_back_p.name)
    matrix_info = c.get_shape(A_data)
    dtype = matrix_info.element_type()

    opaque = gpu_ops.build_blocksparse_back_format_descriptor(m,
                                                              n,
                                                              k,
                                                              block_size_rows_b,
                                                              block_size_cols_b,
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

blocksparse_matmat_back_p = Primitive('gpu_blocksparse_matmat_back')
blocksparse_matmat_back_p.multiple_results = True
blocksparse_matmat_back_p.def_abstract_eval(_blocksparse_matmat_back_abstract)
blocksparse_matmat_back_p.def_impl(partial(xla.apply_primitive, blocksparse_matmat_back_p))

xla.backend_specific_translations['gpu'][blocksparse_matmat_back_p] = _blocksparse_matmat_back_gpu_translation


register_general_batching(blocksparse_matmat_back)