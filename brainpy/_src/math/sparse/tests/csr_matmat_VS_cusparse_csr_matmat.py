import os
import time

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import jax
import jax.numpy as jnp
import taichi as ti
from jax.experimental.sparse import csr

import brainpy as bp
import brainpy.math as bm

bm.set_platform('gpu')

size = [
  (100, 100, 100),
  (100, 1000, 100),
  (1000, 1000, 100),
  (1000, 1000, 1000),
  (100, 10000, 100),
  (10000, 100, 1000),
  (1000, 100, 10000),
  (10000, 10000, 1000),
  (10000, 1000, 10000),
  (10000, 10000, 10000),
  (20000, 20000, 20000),
]

values_type = [
  'homo',
  # 'heter'
]
events_type = ['float']
transpose = [
  # True,
  False
]
ITERATION = 10
SPARSITY = 0.05

print(bm.get_platform())


@ti.kernel
def _csr_matmat_transpose_homo_cpu(col_indices: ti.types.ndarray(ndim=1),
                                   row_ptr: ti.types.ndarray(ndim=1),
                                   matrix: ti.types.ndarray(ndim=2),
                                   out: ti.types.ndarray(ndim=2)):
  # matrix: (k, n)
  # sparse matrix: (m, k)
  n = out.shape[1]
  m = row_ptr.shape[0] - 1
  for j in range(n):  # parallize along the n dimension
    for row_i in range(m):  # loop along the m dimension
      for i in range(row_ptr[row_i], row_ptr[row_i + 1]):
        out[col_indices[i], j] += matrix[row_i, j]


@ti.kernel
def _csr_matmat_transpose_homo_gpu(col_indices: ti.types.ndarray(ndim=1),
                                   row_ptr: ti.types.ndarray(ndim=1),
                                   matrix: ti.types.ndarray(ndim=2),
                                   out: ti.types.ndarray(ndim=2)):
  m = row_ptr.shape[0] - 1
  n = matrix.shape[1]
  for j, row_i in ti.ndrange(n, m):  # paralleize along the (n and m) dimensions
    for i in range(row_ptr[row_i], row_ptr[row_i + 1]):
      out[col_indices[i], j] += matrix[row_i, j]


@ti.kernel
def _csr_matmat_homo(col_indices: ti.types.ndarray(ndim=1),
                     row_ptr: ti.types.ndarray(ndim=1),
                     matrix: ti.types.ndarray(ndim=2),
                     out: ti.types.ndarray(ndim=2)):
  # matrix: (k, n)
  # sparse matrix: (m, k)
  m, n = out.shape
  for row_i, col_k in ti.ndrange(m, n):
    r = 0.
    for row_j in range(row_ptr[row_i], row_ptr[row_i + 1]):
      r += matrix[col_indices[row_j], col_k]
    out[row_i, col_k] = r


# transpose homo
_csr_matmat_transpose_homo_p = bm.XLACustomOp(cpu_kernel=_csr_matmat_transpose_homo_cpu,
                                              gpu_kernel=_csr_matmat_transpose_homo_gpu)

# no transpose homo
_csr_matmat_homo_p = bm.XLACustomOp(cpu_kernel=_csr_matmat_homo, gpu_kernel=_csr_matmat_homo)


def taichi_csrmm(weight, indices, indptr, matrix, shape, transpose):
  indices = bm.as_jax(indices)
  indptr = bm.as_jax(indptr)
  matrix = bm.as_jax(matrix)
  weight = jnp.atleast_1d(weight)
  out_shape = shape[1] if transpose else shape[0]
  result_shape = (out_shape, matrix.shape[1])
  if transpose:
    prim = _csr_matmat_transpose_homo_p
  else:
    prim = _csr_matmat_homo_p
  r = prim(indices,
           indptr,
           matrix,
           outs=[jax.ShapeDtypeStruct(result_shape, dtype=matrix.dtype)],
           transpose=transpose,
           shape=shape)
  return r[0]


SHARED_MEM_SIZE = 256


# @ti.kernel
# def _csr_matmat_homo2(col_indices: ti.types.ndarray(ndim=1),
#                       row_ptr: ti.types.ndarray(ndim=1),
#                       matrix: ti.types.ndarray(ndim=2),
#                       out: ti.types.ndarray(ndim=2)):
#   m, n = out.shape
#   l = col_indices.shape[0]
#   ti.loop_config(block_dim=SHARED_MEM_SIZE)
#   # for i_col, i_row in ti.ndrange(n, m):
#   for i in range(m * n):
#     indices_sm = ti.simt.block.SharedArray((SHARED_MEM_SIZE,), ti.int32)
#
#     # one block threads compute will SHARED_MEM_SIZE columns
#     i_row = i // SHARED_MEM_SIZE
#     i_col = i % SHARED_MEM_SIZE
#
#     index_start = row_ptr[i_row]
#     end_border = row_ptr[i_row + 1]
#     n_share = (end_border - index_start) // SHARED_MEM_SIZE
#     n_last = end_border - index_start - n_share * SHARED_MEM_SIZE
#
#     r = 0.
#     for i_share in range(n_share):
#       indices_sm[i_col] = col_indices[i_col + i_share * SHARED_MEM_SIZE]
#       ti.simt.block.sync()
#       # compute
#       for j in range(SHARED_MEM_SIZE):
#         r += matrix[indices_sm[j], i_col]
#     indices_sm[i_col] = col_indices[ti.min(i_col + n_share * SHARED_MEM_SIZE, l)]
#     ti.simt.block.sync()
#     for j in range(n_last):
#       r += matrix[indices_sm[j], i_col]
#     out[i_row, i_col] += r


@ti.kernel
def _csr_matmat_homo2(col_indices: ti.types.ndarray(ndim=1),
                      row_ptr: ti.types.ndarray(ndim=1),
                      matrix: ti.types.ndarray(ndim=2),
                      out: ti.types.ndarray(ndim=2)):
  m, n = out.shape
  l = col_indices.shape[0]
  ti.loop_config(block_dim=SHARED_MEM_SIZE)

  indices_sm = ti.simt.block.SharedArray((SHARED_MEM_SIZE,), ti.int32)
  # for i_col, i_row in ti.ndrange(n, m):
  for i in ti.ndrange(n * m):
    # i_col = ti.global_thread_idx() % n
    # i_row = ti.global_thread_idx() // n
    i_col = i % n
    i_row = i // n
    i_share = i_col % SHARED_MEM_SIZE

    index_start = row_ptr[i_row]
    end_border = row_ptr[i_row + 1]
    n_share = (end_border - index_start) // SHARED_MEM_SIZE
    n_last = end_border - index_start - n_share * SHARED_MEM_SIZE

    r = 0.
    for k in range(n_share):
      indices_sm[i_share] = col_indices[index_start + i_share + k * SHARED_MEM_SIZE]
      ti.simt.block.sync()
      for j in range(SHARED_MEM_SIZE):
        r += matrix[indices_sm[j], i_col]
    indices_sm[i_share] = col_indices[ti.min(index_start + i_share + n_share * SHARED_MEM_SIZE, l)]
    ti.simt.block.sync()
    for j in range(n_last):
      r += matrix[indices_sm[j], i_col]

    # final results
    out[i_row, i_col] += r


# no transpose homo
_csr_matmat_homo2_p = bm.XLACustomOp(gpu_kernel=_csr_matmat_homo2)


def taichi_csrmm2(weight, indices, indptr, matrix, shape, transpose):
  indices = bm.as_jax(indices)
  indptr = bm.as_jax(indptr)
  matrix = bm.as_jax(matrix)
  weight = jnp.atleast_1d(weight)
  result_shape = (shape[1] if transpose else shape[0], matrix.shape[1])
  return _csr_matmat_homo2_p(indices, indptr, matrix, transpose=transpose, shape=shape,
                             outs=[jax.ShapeDtypeStruct(result_shape, dtype=matrix.dtype)])[0]


def jaxlib_csrmm(weight, indices, indptr, matrix, shape, transpose):
  indices = bm.as_jax(indices)
  indptr = bm.as_jax(indptr)
  matrix = bm.as_jax(matrix)
  weight = jnp.atleast_1d(weight)
  return csr.csr_matmat_p.bind(weight, indices, indptr, matrix, shape=shape, transpose=transpose)


def generate_op(op):
  def csrmm(weight, indices, indptr, matrix, shape, transpose):
    r = 0
    for i in range(ITERATION):
      t = op(weight, indices, indptr, matrix, shape=shape, transpose=transpose)
      r += t
    return r

  return jax.jit(csrmm, static_argnames=('shape', 'transpose'))


def run_spmm_homo(op, shape, transpose, use_heter_data=False):
  bm.random.seed(1234)
  matrix1_shape = (shape[1], shape[0]) if transpose else (shape[0], shape[1])
  matrix2_shape = (shape[1], shape[2])
  indices, indptr = bp.conn.FixedProb(SPARSITY, seed=1234, allow_multi_conn=True)(*matrix1_shape).require('pre2post')
  matrix = bm.as_jax(bm.random.random(matrix2_shape))
  weight = 1.
  if use_heter_data:
    weight = bm.ones(indices.shape) * weight

  result = jax.block_until_ready(op(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  times = []
  for i in range(10):
    time0 = time.time()
    result = jax.block_until_ready(op(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
    time1 = time.time()
    times.append(time1 - time0)
  return np.asarray(times).mean(), result


bm.clear_taichi_aot_caches()
for shape in size:
  for _transpose in transpose:
    cusparse_times, cusparse_r = run_spmm_homo(generate_op(jaxlib_csrmm), shape, _transpose, use_heter_data=True)
    homo1_times, homo1_r = run_spmm_homo(generate_op(taichi_csrmm), shape, _transpose)
    homo2_times, homo2_r = run_spmm_homo(generate_op(taichi_csrmm2), shape, _transpose)
    print(jnp.allclose(cusparse_r, homo1_r), jnp.allclose(cusparse_r, homo2_r))
    print(f'shape={shape}, transpose={_transpose}, cusparse/homo1 = {cusparse_times / homo1_times}, '
          f'cusparse/homo2 = {cusparse_times / homo2_times}')
    print(homo2_r)

# def test_sparse_csrmm(shape, values_type, events_type, transpose):
#   rng = bm.random.RandomState(seed=1234)
#   matrix1_shape = (shape[1], shape[0]) if transpose else (shape[0], shape[1])
#   matrix2_shape = (shape[1], shape[2])
#   indices, indptr = bp.conn.FixedProb(SPARSITY, seed=1234, allow_multi_conn=True)(*matrix1_shape).require('pre2post')
#   matrix = rng.random(matrix2_shape)
#   matrix = bm.as_jax(matrix)
#   weight = 1.
#
#   heter_data = bm.ones(indices.shape) * weight
#
#   if events_type == 'float':
#     matrix = matrix.astype(bm.float32)
#   # if values_type == 'heter':
#   #   weight = heter_data
#
#   result = jax.block_until_ready(
#     csrmm_taichi(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   result = jax.block_until_ready(
#     csrmm_taichi(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   result = jax.block_until_ready(
#     csrmm_taichi(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   result = jax.block_until_ready(
#     csrmm_taichi(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   result = jax.block_until_ready(
#     csrmm_taichi(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#
#   time0 = time.time()
#   result = jax.block_until_ready(
#     csrmm_taichi(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time1 = time.time()
#
#   time2 = time.time()
#   result = jax.block_until_ready(
#     csrmm_taichi(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time3 = time.time()
#
#   time4 = time.time()
#   result = jax.block_until_ready(
#     csrmm_taichi(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time5 = time.time()
#
#   time6 = time.time()
#   result = jax.block_until_ready(
#     csrmm_taichi(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time7 = time.time()
#
#   time8 = time.time()
#   result = jax.block_until_ready(
#     csrmm_taichi(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time9 = time.time()
#
#   time10 = time.time()
#   result = jax.block_until_ready(
#     csrmm_taichi(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time11 = time.time()
#
#   time12 = time.time()
#   result = jax.block_until_ready(
#     csrmm_taichi(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time13 = time.time()
#
#   time14 = time.time()
#   result = jax.block_until_ready(
#     csrmm_taichi(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time15 = time.time()
#
#   time16 = time.time()
#   result = jax.block_until_ready(
#     csrmm_taichi(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time17 = time.time()
#
#   time18 = time.time()
#   result = jax.block_until_ready(
#     csrmm_taichi(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time19 = time.time()
#
#   result1 = result
#
#   result = jax.block_until_ready(csrmm(heter_data, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   result = jax.block_until_ready(csrmm(heter_data, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   result = jax.block_until_ready(csrmm(heter_data, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   result = jax.block_until_ready(csrmm(heter_data, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   result = jax.block_until_ready(csrmm(heter_data, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#
#   time20 = time.time()
#   result = jax.block_until_ready(csrmm(heter_data, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time21 = time.time()
#
#   result2 = result
#
#   time22 = time.time()
#   result = jax.block_until_ready(csrmm(heter_data, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time23 = time.time()
#
#   time24 = time.time()
#   result = jax.block_until_ready(csrmm(heter_data, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time25 = time.time()
#
#   time26 = time.time()
#   result = jax.block_until_ready(csrmm(heter_data, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time27 = time.time()
#
#   time28 = time.time()
#   result = jax.block_until_ready(csrmm(heter_data, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time29 = time.time()
#
#   time30 = time.time()
#   result = jax.block_until_ready(csrmm(heter_data, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time31 = time.time()
#
#   time32 = time.time()
#   result = jax.block_until_ready(csrmm(heter_data, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time33 = time.time()
#
#   time34 = time.time()
#   result = jax.block_until_ready(csrmm(heter_data, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time35 = time.time()
#
#   time36 = time.time()
#   result = jax.block_until_ready(csrmm(heter_data, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time37 = time.time()
#
#   time38 = time.time()
#   result = jax.block_until_ready(csrmm(heter_data, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
#   time39 = time.time()
#
#   taichi_aot_time1 = (time1 - time0) * 1000
#   taichi_aot_time2 = (time3 - time2) * 1000
#   taichi_aot_time3 = (time5 - time4) * 1000
#   taichi_aot_time4 = (time7 - time6) * 1000
#   taichi_aot_time5 = (time9 - time8) * 1000
#   taichi_aot_time6 = (time11 - time10) * 1000
#   taichi_aot_time7 = (time13 - time12) * 1000
#   taichi_aot_time8 = (time15 - time14) * 1000
#   taichi_aot_time9 = (time17 - time16) * 1000
#   taichi_aot_time10 = (time19 - time18) * 1000
#   brainpy_time1 = (time21 - time20) * 1000
#   brainpy_time2 = (time23 - time22) * 1000
#   brainpy_time3 = (time25 - time24) * 1000
#   brainpy_time4 = (time27 - time26) * 1000
#   brainpy_time5 = (time29 - time28) * 1000
#   brainpy_time6 = (time31 - time30) * 1000
#   brainpy_time7 = (time33 - time32) * 1000
#   brainpy_time8 = (time35 - time34) * 1000
#   brainpy_time9 = (time37 - time36) * 1000
#   brainpy_time10 = (time39 - time38) * 1000
#   print('shape: ', shape, 'values_type: ', values_type, 'events_type: ', events_type, 'transpose: ', transpose)
#   print('taichi_aot_1: ', taichi_aot_time1, 'ms')
#   print('taichi_aot_3: ', taichi_aot_time3, 'ms')
#   print('taichi_aot_5: ', taichi_aot_time5, 'ms')
#   print('taichi_aot_7: ', taichi_aot_time7, 'ms')
#   print('taichi_aot_9: ', taichi_aot_time9, 'ms')
#   print('brainpylib_1: ', brainpy_time1, 'ms')
#   print('brainpylib_3: ', brainpy_time3, 'ms')
#   print('brainpylib_5: ', brainpy_time5, 'ms')
#   print('brainpylib_7: ', brainpy_time7, 'ms')
#   print('brainpylib_9: ', brainpy_time9, 'ms')
#   print(bm.allclose(result1, result2))
#
#   return taichi_aot_time1, taichi_aot_time2, taichi_aot_time3, taichi_aot_time4, taichi_aot_time5, \
#     taichi_aot_time6, taichi_aot_time7, taichi_aot_time8, taichi_aot_time9, taichi_aot_time10, \
#     brainpy_time1, brainpy_time2, brainpy_time3, brainpy_time4, brainpy_time5, \
#     brainpy_time6, brainpy_time7, brainpy_time8, brainpy_time9, brainpy_time10

# PATH = os.path.dirname(os.path.abspath(__file__))
#
# # init dataframe
# df = pd.DataFrame(
#   columns=['s', 'p', 'shape[0]', 'shape[1]', 'shape[2]', 'backend', 'values type', 'events type', 'transpose',
#            'taichi aot time1(ms)', 'taichi aot time2(ms)', 'taichi aot time3(ms)', 'taichi aot time4(ms)',
#            'taichi aot time5(ms)',
#            'taichi aot time6(ms)', 'taichi aot time7(ms)', 'taichi aot time8(ms)', 'taichi aot time9(ms)',
#            'taichi aot time10(ms)',
#            'brainpy time1(ms)', 'brainpy time2(ms)', 'brainpy time3(ms)', 'brainpy time4(ms)', 'brainpy time5(ms)',
#            'brainpy time6(ms)', 'brainpy time7(ms)', 'brainpy time8(ms)', 'brainpy time9(ms)', 'brainpy time10(ms)'])
#
# for shape in size:
#   for _values_type in values_type:
#     for _events_type in events_type:
#       for _transpose in transpose:
#         taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5, \
#           taichi_aot_time_6, taichi_aot_time_7, taichi_aot_time_8, taichi_aot_time_9, taichi_aot_time_10, \
#           brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, \
#           brainpy_time_6, brainpy_time_7, brainpy_time_8, brainpy_time_9, brainpy_time_10 = test_sparse_csrmm(shape,
#                                                                                                               _values_type,
#                                                                                                               _events_type,
#                                                                                                               _transpose)
#         # append to dataframe
#         df.loc[df.shape[0]] = [shape, 0.5, shape[0], shape[1], shape[2], 'gpu', _values_type, _events_type,
#                                _transpose,
#                                taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4,
#                                taichi_aot_time_5,
#                                taichi_aot_time_6, taichi_aot_time_7, taichi_aot_time_8, taichi_aot_time_9,
#                                taichi_aot_time_10,
#                                brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5,
#                                brainpy_time_6, brainpy_time_7, brainpy_time_8, brainpy_time_9, brainpy_time_10]
#
#         print(shape, _values_type, _events_type, _transpose)
#         a = np.asarray([taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4,
#                         taichi_aot_time_5,
#                         taichi_aot_time_6, taichi_aot_time_7, taichi_aot_time_8, taichi_aot_time_9,
#                         taichi_aot_time_10])
#         b = np.asarray([brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5,
#                         brainpy_time_6, brainpy_time_7, brainpy_time_8, brainpy_time_9, brainpy_time_10])
#         print(a)
#         print(b)
#         print(a.sum() / b.sum())
#         df.to_csv(f'{PATH}/csrmm_{bm.get_platform()}.csv', index=False)
