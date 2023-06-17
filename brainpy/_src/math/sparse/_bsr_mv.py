from functools import partial
from typing import Union, Tuple

import numba
import numpy as np
from jax import numpy as jnp
from jax.core import ShapedArray, Primitive
from jax.interpreters import ad, xla
from jax.lib import xla_client

from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.op_registers import (compile_cpu_signature_with_numba,
                                            register_general_batching)
from brainpy._src.math.sparse._utils import csr_to_coo
from brainpy.errors import GPUOperatorNotFound

try:
  from brainpylib import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  'cusparse_bcsr_matvec'
]


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _cusparse_bcsr_matvec_bsr_matvec_numba_imp(outs, ins):
  data, indices, indptr, vector, blocksize, shape, nnzb, transpose = ins
  blocksize = blocksize[()]
  outs.fill(0)
  for i in range(shape[0]):
    tmp = np.zeros(blocksize, dtype=data.dtype)
    for j in range(indptr[i], indptr[i + 1]):
      start = indices[j] * blocksize
      end = start + blocksize
      tmp += data[start: end] @ vector[start: end]
    outs[i * blocksize: (i + 1) * blocksize] = tmp


# @numba.njit(fastmath=True, parallel=True, nogil=True)
# def _cusparse_bcsr_matvec_bsr_matvec_numba_imp(outs, ins):
#   data, indices, indptr, vector,  blocksize , shape,nnzb,transpose = ins
#   blocksize = blocksize[()]
#   outs.fill(0)

#   cnt=0
#   for i in range(0,shape[0]):
#       outs.fill(0.0)
#       tmp=[0.0]*blocksize
#       for j in range(indptr[i], indptr[i + 1]):
#         for p in range(0,blocksize):
#           for q in range(0,blocksize):
#             tmp[p] += vector[indices[j]*blocksize+q]*data[j*blocksize+p][q]
#       for j in range(0,blocksize):
#         outs[cnt] = tmp[j]
#         cnt+=1


def _cusprase_bcsr_matvec_values(values, indices, indptr, vector, *, blocksize, nnzb, shape, transpose):
  return cusparse_bcsr_matvec(values, indices, indptr, vector, blocksize, nnzb=nnzb, shape=shape, transpose=transpose)


def cusparse_bcsr_matvec(
    data: Union[float, jnp.ndarray],
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    vector: jnp.ndarray,
    *,
    blocksize: int,
    nnzb: int,
    shape: Tuple[int, int],
    method: str = 'vector',
    transpose: bool = False
) -> jnp.ndarray:
  data = as_jax(data)
  indices = as_jax(indices)
  indptr = as_jax(indptr)
  vector = as_jax(vector)
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
  # assert data.ndim == indices.ndim == indptr.ndim == vector.ndim == 1

  return cusparse_bcsr_matvec_vector_p.bind(data, indices, indptr, vector, blocksize=blocksize, shape=shape, nnzb=nnzb,
                                            transpose=transpose)


def _cusparse_bcsr_matvec_vector_cpu_translation(c, data, indices, indptr, vector, *, blocksize, shape, nnzb,
                                                 transpose):
  inputs = (data, indices, indptr, vector)
  print(c.get_shape(data))
  description = dict(blocksize=blocksize, shape=shape, nnzb=nnzb, transpose=transpose, )
  if transpose:
    skip = 1
  else:
    name, inputs, in_layouts, out_layouts = compile_cpu_signature_with_numba(
      c,
      _cusparse_bcsr_matvec_bsr_matvec_numba_imp,
      abs_eval_fn=_cusparse_bcsr_matvec_abstract,
      multiple_results=False,
      inputs=inputs,
      description=description
    )
  return xla_client.ops.CustomCallWithLayout(
    c, name,
    operands=inputs,
    operand_shapes_with_layout=in_layouts,
    shape_with_layout=out_layouts,
  )


def _cusparse_bcsr_matvec_vector_gpu_translation(c, data, indices, indptr, vector, *, blocksize, shape, nnzb):
  if gpu_ops is None:
    raise GPUOperatorNotFound(cusparse_bcsr_matvec_vector_p.name)

  data_shape = c.get_shape(data)
  if data_shape.element_type() == np.float32:
    type_name = b'float'
  elif data_shape.element_type() == np.double:
    type_name = b'double'
  else:
    raise ValueError('data_type not support(except float/double)')
  # 有可能不是这个

  opaque = gpu_ops.build_bcsrcusparsespmv_descriptor(shape[0], shape[1], blocksize, nnzb)
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'gpu_bcsr_cusparse_spmv_' + type_name,
    operands=(data, indices, indptr, vector),
    operand_shapes_with_layout=(c.get_shape(data),
                                c.get_shape(indices),
                                c.get_shape(indptr),
                                c.get_shape(vector),
                                ),
    shape_with_layout=xla_client.Shape.array_shape(data_shape.element_type(), (shape[0] * blocksize,), (0,)),
    opaque=opaque,
  )


# def _bcsr_matvec_abstract(*args, **kwargs):
#   data = args[0]
#   assert len(kwargs) == 1
#   shape = kwargs['shape']
#   return ShapedArray(dtype=data.dtype, shape=(shape[0],))

# bcsr_matvec_vector_p = register_op_with_numba(
#   'bcsr_matvec_vector',
#   cpu_func=None,
#   out_shapes=_bcsr_matvec_abstract,
#   gpu_func_translation=_bcsr_matvec_vector_gpu_translation,
# )


# def _batch_bcsr_matvec_abstract(
#     values, indices, indptr, vector,block_size, *, shape, transpose=False
# ):
#   return ShapedArray(dtype=values.dtype, shape=(batch_size, shape[1] if transpose else shape[0]))

def _cusparse_bcsr_matvec_abstract(data, indices, indptr, vector, *, blocksize, shape, nnzb, transpose=False):
  return ShapedArray(dtype=data.dtype, shape=(shape[0] * blocksize,))


def _cusparse_bcsr_matvec_jvp_values(data_dot, data, indices, indptr, vector, *, blocksize, shape, nnzb, transpose):
  return cusparse_bcsr_matvec(data_dot, indices, indptr, vector, blocksize=blocksize, nnzb=nnzb, shape=shape,
                              transpose=transpose)


def _cusparse_bcsr_transpose(ct, data, indices, indptr, vector, *, blocksize, shape, transpose):
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices.")
  if ad.is_undefined_primal(vector):
    ct_events = cusparse_bcsr_matvec(data, indices, indptr, ct, shape=shape, transpose=not transpose)
    return data, indices, indptr, (ad.Zero(vector) if type(ct) is ad.Zero else ct_events)
  else:
    if type(ct) is ad.Zero:
      ct_values = ad.Zero(data)
    else:
      row, col = csr_to_coo(indices, indptr)
      cnt = 0
      ct_values = []
      for i in row:
        for j in col:
          for p in range(0, blocksize):
            cntq = 0
            for q in range(0, blocksize):
              if transpose:
                ct_values[cnt][cntq] = vector[i * blocksize + p] * ct[j * blocksize + q]
              else:
                ct_values[cnt][cntq] = vector[j * blocksize + q] * ct[i * blocksize + p]
              cntq += 1
            cnt += 1
    return ct_values, indices, indptr, vector


cusparse_bcsr_matvec_vector_p = Primitive('cusparse_block_spmv')
cusparse_bcsr_matvec_vector_p.def_abstract_eval(_cusparse_bcsr_matvec_abstract)
cusparse_bcsr_matvec_vector_p.def_impl(partial(xla.apply_primitive, cusparse_bcsr_matvec_vector_p))
xla.backend_specific_translations['gpu'][cusparse_bcsr_matvec_vector_p] = _cusparse_bcsr_matvec_vector_gpu_translation
xla.backend_specific_translations['cpu'][cusparse_bcsr_matvec_vector_p] = _cusparse_bcsr_matvec_vector_cpu_translation
ad.defjvp(cusparse_bcsr_matvec_vector_p, _cusparse_bcsr_matvec_jvp_values)
ad.primitive_transposes[cusparse_bcsr_matvec_vector_p] = _cusparse_bcsr_transpose
register_general_batching(cusparse_bcsr_matvec_vector_p)
# batching.primitive_batchers[event_csr_matvec_p] = _event_csr_matvec_batching_rule
