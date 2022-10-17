# -*- coding: utf-8 -*-

import ctypes
import ctypes.util
import sys

import numpy as np
from cffi import FFI
from jax.abstract_arrays import ShapedArray
from jax.lib import xla_client
from jax import dtypes
from numba import cuda, cfunc, types


class Dl_info(ctypes.Structure):
  """
  Structure of the Dl_info returned by the CFFI of dl.dladdr
  """

  _fields_ = (
    ("dli_fname", ctypes.c_char_p),
    ("dli_fbase", ctypes.c_void_p),
    ("dli_sname", ctypes.c_char_p),
    ("dli_saddr", ctypes.c_void_p),
  )


# Find the dynamic linker library path. Only works on unix-like os
libdl_path = ctypes.util.find_library("dl")
if libdl_path:
  # Load the dynamic linker dynamically
  libdl = ctypes.CDLL(libdl_path)

  # Define dladdr to get the pointer to a symbol in a shared
  # library already loaded.
  # https://man7.org/linux/man-pages/man3/dladdr.3.html
  libdl.dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(Dl_info))
  # restype is None as it returns by reference
else:
  # On Windows it is nontrivial to have libdl, so we disable everything about
  # it and use other ways to find paths of libraries
  libdl = None


def find_path_of_symbol_in_library(symbol):
  if libdl is None:
    raise ValueError("libdl not found.")

  info = Dl_info()
  result = libdl.dladdr(symbol, ctypes.byref(info))
  if result and info.dli_fname:
    return info.dli_fname.decode(sys.getfilesystemencoding())
  else:
    raise ValueError("Cannot determine path of Library.")


try:
  _libcuda = cuda.driver.find_driver()
  if sys.platform == "win32":
    libcuda_path = ctypes.util.find_library(_libcuda._name)
  else:
    libcuda_path = find_path_of_symbol_in_library(_libcuda.cuMemcpy)
  numba_cffi_loaded = True
except Exception:
  numba_cffi_loaded = False


if numba_cffi_loaded:
  # functions needed
  ffi = FFI()
  ffi.cdef("int cuMemcpy(void* dst, void* src, unsigned int len, int type);")
  ffi.cdef("int cuMemcpyAsync(void* dst, void* src, unsigned int len, int type, void* stream);")
  ffi.cdef("int cuStreamSynchronize(void* stream);")
  ffi.cdef("int cudaMallocHost(void** ptr, size_t size);")
  ffi.cdef("int cudaFreeHost(void* ptr);")

  # load libraray
  # could  ncuda.driver.find_library()
  libcuda = ffi.dlopen(libcuda_path)
  cuMemcpy = libcuda.cuMemcpy
  cuMemcpyAsync = libcuda.cuMemcpyAsync
  cuStreamSynchronize = libcuda.cuStreamSynchronize

  memcpyHostToHost = types.int32(0)
  memcpyHostToDevice = types.int32(1)
  memcpyDeviceToHost = types.int32(2)
  memcpyDeviceToDevice = types.int32(3)


_lambda_no = 0
ctypes.pythonapi.PyCapsule_New.argtypes = [
  ctypes.c_void_p,  # void* pointer
  ctypes.c_char_p,  # const char *name
  ctypes.c_void_p,  # PyCapsule_Destructor destructor
]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object


def _compile_gpu_signature(func, input_dtypes, input_shapes,
                           output_dtypes, output_shapes):
  input_byte_size = tuple(
    np.prod(shape) * dtype.itemsize
    for (shape, dtype) in zip(input_shapes, input_dtypes)
  )
  output_byte_size = tuple(
    np.prod(shape) * dtype.itemsize
    for (shape, dtype) in zip(output_shapes, output_dtypes)
  )

  code_scope = dict(
    func_to_call=func,
    input_shapes=input_shapes,
    input_dtypes=input_dtypes,
    output_shapes=output_shapes,
    output_dtypes=output_dtypes,
    empty=np.empty,
    input_byte_size=input_byte_size,
    output_byte_size=output_byte_size,
    cuMemcpyAsync=cuMemcpyAsync,
    cuStreamSynchronize=cuStreamSynchronize,
    memcpyHostToHost=memcpyHostToHost,
    memcpyHostToDevice=memcpyHostToDevice,
    memcpyDeviceToHost=memcpyDeviceToHost,
    memcpyDeviceToDevice=memcpyDeviceToDevice,
    n_in=len(input_shapes),
  )

  args_in = [
    f'empty(input_shapes[{i}], dtype=input_dtypes[{i}]),'
    for i in range(len(input_shapes))
  ]
  cuMemcpyAsync_in = [
    f'cuMemcpyAsync(args_in[{i}].ctypes.data, inout_gpu_ptrs[{i}], input_byte_size[{i}], memcpyDeviceToHost, stream)'
    for i in range(len(input_shapes))
  ]
  args_out = [
    f'empty(output_shapes[{i}], dtype=output_dtypes[{i}]),'
    for i in range(len(output_shapes))
  ]
  cuMemcpyAsync_out = [
    f'cuMemcpyAsync(inout_gpu_ptrs[n_in + {i}], args_out[{i}].ctypes.data, output_byte_size[{i}], ' \
    f'memcpyHostToDevice, stream)'
    for i in range(len(output_shapes))
  ]

  code_string = '''
def xla_gpu_custom_call_target(stream, inout_gpu_ptrs, opaque, opaque_len):
  args_out = (
    {args_out}
  )
  args_in = (
    {args_in}
  )
  {cuMemcpyAsync_in}
  cuStreamSynchronize(stream)
  func_to_call(args_out, args_in)
  {cuMemcpyAsync_out}
    '''.format(args_in="\n    ".join(args_in),
               args_out="\n    ".join(args_out),
               cuMemcpyAsync_in="\n  ".join(cuMemcpyAsync_in),
               cuMemcpyAsync_out="\n  ".join(cuMemcpyAsync_out))
  # print(code_string)
  exec(compile(code_string.strip(), '', 'exec'), code_scope)

  new_f = code_scope['xla_gpu_custom_call_target']
  wrapper = cfunc(types.void(
    types.voidptr,
    types.CPointer(types.voidptr),
    types.voidptr, types.uint64))
  xla_c_rule = wrapper(new_f)
  target_name = xla_c_rule.native_name.encode("ascii")
  capsule = ctypes.pythonapi.PyCapsule_New(
    xla_c_rule.address,  # A CFFI pointer to a function
    b"xla._CUSTOM_CALL_TARGET",  # A binary string
    None  # PyCapsule object run at destruction
  )
  xla_client.register_custom_call_target(target_name, capsule, "gpu")
  return target_name


def func_gpu_translation(func, abs_eval_fn, c, *inputs, **info):
  if not numba_cffi_loaded:
    raise RuntimeError("Numba cffi could not be loaded.")

  input_shapes = [c.get_shape(arg) for arg in inputs]
  for v in info.values():
    input_shapes.append(xla_client.Shape.array_shape(dtypes.canonicalize_dtype(type(v)), (), ()))
  input_dtypes = tuple(shape.element_type() for shape in input_shapes)
  input_dimensions = tuple(shape.dimensions() for shape in input_shapes)
  output_abstract_arrays = abs_eval_fn(*tuple(ShapedArray(shape.dimensions(), shape.element_type())
                                              for shape in input_shapes[:len(inputs)]),
                                       **info)
  output_shapes = tuple(array.shape for array in output_abstract_arrays)
  output_dtypes = tuple(array.dtype for array in output_abstract_arrays)
  output_layouts = map(lambda shape: range(len(shape) - 1, -1, -1), output_shapes)
  xla_output_shapes = [xla_client.Shape.array_shape(*arg)
                       for arg in zip(output_dtypes, output_shapes, output_layouts)]
  xla_output_shape = xla_client.Shape.tuple_shape(xla_output_shapes)
  target_name = _compile_gpu_signature(func,
                                       input_dtypes, input_dimensions,
                                       output_dtypes, output_shapes)

  return xla_client.ops.CustomCallWithLayout(
    c,
    target_name,
    operands=inputs + tuple(xla_client.ops.ConstantLiteral(c, i) for i in info.values()),
    operand_shapes_with_layout=input_shapes,
    shape_with_layout=xla_output_shape,
  )



