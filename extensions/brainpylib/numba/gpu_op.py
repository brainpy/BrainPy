from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax import core
from jax.interpreters import xla
from jax.lib import xla_client
from jax.abstract_arrays import ShapedArray
import collections
import numba
from numba import types
from numba import cuda
import ctypes
import _cuda

x_shape = xla_client.Shape.array_shape
x_ops = xla_client.ops

ctypes.pythonapi.PyCapsule_New.argtypes = [
  ctypes.c_void_p,  # void* pointer
  ctypes.c_char_p,  # const char *name
  ctypes.c_void_p,  # PyCapsule_Destructor destructor
]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object

xla_call_sig = types.void(
  types.voidptr,  # cudaStream_t* stream
  types.CPointer(types.voidptr),  # void** buffers
  types.voidptr,  # const char* opaque
  types.uint64,  # size_t opaque_len
)


def bind_primitive(primitive, abs_eval_fn, *args):
  result = primitive.bind(*args)

  output_shapes = abs_eval_fn(*args)
  # Special-casing when only a single tensor is returned.
  if not isinstance(output_shapes, collections.abc.Collection):
    assert len(result) == 1
    return result[0]
  else:
    return result


def abs_eval_rule(abs_eval, *args, **kwargs):
  # Special-casing when only a single tensor is returned.
  shapes = abs_eval(*args, **kwargs)
  if not isinstance(shapes, collections.abc.Collection):
    return [shapes]
  else:
    return shapes


def eval_rule(call_fn, abs_eval, *args, **kwargs):
  # compute the output shapes
  output_shapes = abs_eval(*args)
  # Preallocate the outputs
  outputs = tuple(np.zeros(shape.shape, dtype=shape.dtype) for shape in output_shapes)
  # convert inputs to a tuple
  inputs = tuple(np.asarray(arg) for arg in args)
  # call the kernel
  call_fn(outputs + inputs, **kwargs)
  # Return the outputs
  return tuple(outputs)


def pycapsule_new(ptr, name, destructor=None) -> ctypes.py_object:
  """
  Wraps a C function pointer into an XLA-compatible PyCapsule.

  Args:
      ptr: A CFFI pointer to a function
      name: A binary string
      destructor: Optional PyCapsule object run at destruction

  Returns
      a PyCapsule (ctypes.py_object)
  """
  return ctypes.pythonapi.PyCapsule_New(ptr, name, None)


# todo: How to decode inputs and outputs?
def create_numba_api_wrapper(func,
                             input_dtypes, input_shapes,
                             output_dtypes, output_shapes):
  n_in = len(input_shapes)
  n_out = len(output_shapes)
  if n_in > 6:
    raise NotImplementedError(
      "n_in ∈ [0, 6] inputs are supported ({n_in} detected)."
      "Please open a bug report."
    )
  if n_out > 4 or n_out == 0:
    raise NotImplementedError(
      "n_out ∈ [1, 4] outputs are supported ({n_out} detected)."
      "Please open a bug report."
    )

  @numba.cfunc(xla_call_sig)
  def xla_gpu_custom_call_target(output_ptrs, input_ptrs):
    n_in = len(input_shapes)
    n_out = len(output_shapes)

    input_byte_size = tuple(
      np.prod(shape) * dtype.itemsize
      for (shape, dtype) in zip(input_shapes, input_dtypes)
    )
    output_byte_size = tuple(
      np.prod(shape) * dtype.itemsize
      for (shape, dtype) in zip(output_shapes, output_dtypes)
    )

    @numba.cfunc(xla_call_sig)
    def xla_custom_call_target(stream, inout_gpu_ptrs, opaque, opaque_len):
      # manually unroll input and output args because numba is
      # relatively dummb and cannot always infer getitem on inhomogeneous tuples

      # allocate output cpu bufferess
      if n_out == 1:
        args_out = (np.empty(output_shapes[0], dtype=output_dtypes[0]),)
      elif n_out == 2:
        args_out = (
          np.empty(output_shapes[0], dtype=output_dtypes[0]),
          np.empty(output_shapes[1], dtype=output_dtypes[1]),
        )
      elif n_out == 3:
        args_out = (
          np.empty(output_shapes[0], dtype=output_dtypes[0]),
          np.empty(output_shapes[1], dtype=output_dtypes[1]),
          np.empty(output_shapes[2], dtype=output_dtypes[2]),
        )
      elif n_out == 4:
        args_out = (
          np.empty(output_shapes[0], dtype=output_dtypes[0]),
          np.empty(output_shapes[1], dtype=output_dtypes[1]),
          np.empty(output_shapes[2], dtype=output_dtypes[2]),
          np.empty(output_shapes[3], dtype=output_dtypes[3]),
        )

      # allocate input cpu buffers and
      if n_in == 1:
        args_in = (np.empty(input_shapes[0], dtype=input_dtypes[0]),)
        _cuda.cuMemcpyAsync(
          args_in[0].ctypes.data,
          inout_gpu_ptrs[0],
          input_byte_size[0],
          _cuda.memcpyDeviceToHost,
          stream,
        )
      elif n_in == 2:
        args_in = (
          np.empty(input_shapes[0], dtype=input_dtypes[0]),
          np.empty(input_shapes[1], dtype=input_dtypes[1]),
        )
        _cuda.cuMemcpyAsync(
          args_in[0].ctypes.data,
          inout_gpu_ptrs[0],
          input_byte_size[0],
          _cuda.memcpyDeviceToHost,
          stream,
        )
        _cuda.cuMemcpyAsync(
          args_in[1].ctypes.data,
          inout_gpu_ptrs[1],
          input_byte_size[1],
          _cuda.memcpyDeviceToHost,
          stream,
        )
      elif n_in == 3:
        args_in = (
          np.empty(input_shapes[0], dtype=input_dtypes[0]),
          np.empty(input_shapes[1], dtype=input_dtypes[1]),
          np.empty(input_shapes[2], dtype=input_dtypes[2]),
        )
        _cuda.cuMemcpyAsync(
          args_in[0].ctypes.data,
          inout_gpu_ptrs[0],
          input_byte_size[0],
          _cuda.memcpyDeviceToHost,
          stream,
        )
        _cuda.cuMemcpyAsync(
          args_in[1].ctypes.data,
          inout_gpu_ptrs[1],
          input_byte_size[1],
          _cuda.memcpyDeviceToHost,
          stream,
        )
        _cuda.cuMemcpyAsync(
          args_in[2].ctypes.data,
          inout_gpu_ptrs[2],
          input_byte_size[2],
          _cuda.memcpyDeviceToHost,
          stream,
        )
      elif n_in == 4:
        args_in = (
          np.empty(input_shapes[0], dtype=input_dtypes[0]),
          np.empty(input_shapes[1], dtype=input_dtypes[1]),
          np.empty(input_shapes[2], dtype=input_dtypes[2]),
          np.empty(input_shapes[3], dtype=input_dtypes[3]),
        )
        _cuda.cuMemcpyAsync(
          args_in[0].ctypes.data,
          inout_gpu_ptrs[0],
          input_byte_size[0],
          _cuda.memcpyDeviceToHost,
          stream,
        )
        _cuda.cuMemcpyAsync(
          args_in[1].ctypes.data,
          inout_gpu_ptrs[1],
          input_byte_size[1],
          _cuda.memcpyDeviceToHost,
          stream,
        )
        _cuda.cuMemcpyAsync(
          args_in[2].ctypes.data,
          inout_gpu_ptrs[2],
          input_byte_size[2],
          _cuda.memcpyDeviceToHost,
          stream,
        )
        _cuda.cuMemcpyAsync(
          args_in[3].ctypes.data,
          inout_gpu_ptrs[3],
          input_byte_size[3],
          _cuda.memcpyDeviceToHost,
          stream,
        )
      elif n_in == 5:
        args_in = (
          np.empty(input_shapes[0], dtype=input_dtypes[0]),
          np.empty(input_shapes[1], dtype=input_dtypes[1]),
          np.empty(input_shapes[2], dtype=input_dtypes[2]),
          np.empty(input_shapes[3], dtype=input_dtypes[3]),
          np.empty(input_shapes[4], dtype=input_dtypes[4]),
        )
        _cuda.cuMemcpyAsync(
          args_in[0].ctypes.data,
          inout_gpu_ptrs[0],
          input_byte_size[0],
          _cuda.memcpyDeviceToHost,
          stream,
        )
        _cuda.cuMemcpyAsync(
          args_in[1].ctypes.data,
          inout_gpu_ptrs[1],
          input_byte_size[1],
          _cuda.memcpyDeviceToHost,
          stream,
        )
        _cuda.cuMemcpyAsync(
          args_in[2].ctypes.data,
          inout_gpu_ptrs[2],
          input_byte_size[2],
          _cuda.memcpyDeviceToHost,
          stream,
        )
        _cuda.cuMemcpyAsync(
          args_in[3].ctypes.data,
          inout_gpu_ptrs[3],
          input_byte_size[3],
          _cuda.memcpyDeviceToHost,
          stream,
        )
        _cuda.cuMemcpyAsync(
          args_in[4].ctypes.data,
          inout_gpu_ptrs[4],
          input_byte_size[4],
          _cuda.memcpyDeviceToHost,
          stream,
        )
      elif n_in == 6:
        args_in = (
          np.empty(input_shapes[0], dtype=input_dtypes[0]),
          np.empty(input_shapes[1], dtype=input_dtypes[1]),
          np.empty(input_shapes[2], dtype=input_dtypes[2]),
          np.empty(input_shapes[3], dtype=input_dtypes[3]),
          np.empty(input_shapes[4], dtype=input_dtypes[4]),
          np.empty(input_shapes[5], dtype=input_dtypes[5]),
        )
        _cuda.cuMemcpyAsync(
          args_in[0].ctypes.data,
          inout_gpu_ptrs[0],
          input_byte_size[0],
          _cuda.memcpyDeviceToHost,
          stream,
        )
        _cuda.cuMemcpyAsync(
          args_in[1].ctypes.data,
          inout_gpu_ptrs[1],
          input_byte_size[1],
          _cuda.memcpyDeviceToHost,
          stream,
        )
        _cuda.cuMemcpyAsync(
          args_in[2].ctypes.data,
          inout_gpu_ptrs[2],
          input_byte_size[2],
          _cuda.memcpyDeviceToHost,
          stream,
        )
        _cuda.cuMemcpyAsync(
          args_in[3].ctypes.data,
          inout_gpu_ptrs[3],
          input_byte_size[3],
          _cuda.memcpyDeviceToHost,
          stream,
        )
        _cuda.cuMemcpyAsync(
          args_in[4].ctypes.data,
          inout_gpu_ptrs[4],
          input_byte_size[4],
          _cuda.memcpyDeviceToHost,
          stream,
        )
        _cuda.cuMemcpyAsync(
          args_in[5].ctypes.data,
          inout_gpu_ptrs[5],
          input_byte_size[5],
          _cuda.memcpyDeviceToHost,
          stream,
        )
      _cuda.cuStreamSynchronize(stream)
      func(args_out + args_in)

      if n_out == 1:
        _cuda.cuMemcpyAsync(
          inout_gpu_ptrs[n_in + 0],
          args_out[0].ctypes.data,
          output_byte_size[0],
          _cuda.memcpyHostToDevice,
          stream,
        )
      elif n_out == 2:
        _cuda.cuMemcpyAsync(
          inout_gpu_ptrs[n_in + 0],
          args_out[0].ctypes.data,
          output_byte_size[0],
          _cuda.memcpyHostToDevice,
          stream,
        )
        _cuda.cuMemcpyAsync(
          inout_gpu_ptrs[n_in + 1],
          args_out[1].ctypes.data,
          output_byte_size[1],
          _cuda.memcpyHostToDevice,
          stream,
        )
      elif n_out == 3:
        _cuda.cuMemcpyAsync(
          inout_gpu_ptrs[n_in + 0],
          args_out[0].ctypes.data,
          output_byte_size[0],
          _cuda.memcpyHostToDevice,
          stream,
        )
        _cuda.cuMemcpyAsync(
          inout_gpu_ptrs[n_in + 1],
          args_out[1].ctypes.data,
          output_byte_size[1],
          _cuda.memcpyHostToDevice,
          stream,
        )
        _cuda.cuMemcpyAsync(
          inout_gpu_ptrs[n_in + 2],
          args_out[2].ctypes.data,
          output_byte_size[2],
          _cuda.memcpyHostToDevice,
          stream,
        )
      elif n_out == 4:
        _cuda.cuMemcpyAsync(
          inout_gpu_ptrs[n_in + 0],
          args_out[0].ctypes.data,
          output_byte_size[0],
          _cuda.memcpyHostToDevice,
          stream,
        )
        _cuda.cuMemcpyAsync(
          inout_gpu_ptrs[n_in + 1],
          args_out[1].ctypes.data,
          output_byte_size[1],
          _cuda.memcpyHostToDevice,
          stream,
        )
        _cuda.cuMemcpyAsync(
          inout_gpu_ptrs[n_in + 2],
          args_out[2].ctypes.data,
          output_byte_size[2],
          _cuda.memcpyHostToDevice,
          stream,
        )
        _cuda.cuMemcpyAsync(
          inout_gpu_ptrs[n_in + 3],
          args_out[3].ctypes.data,
          output_byte_size[3],
          _cuda.memcpyHostToDevice,
          stream,
        )

      _cuda.cuStreamSynchronize(stream)

  return xla_gpu_custom_call_target


def compile_gpu_signature(func,
                          input_dtypes, input_shapes,
                          output_dtypes, output_shapes):
  xla_c_rule = create_numba_api_wrapper(
    func,
    input_dtypes, input_shapes,
    output_dtypes, output_shapes
  )
  target_name = xla_c_rule.native_name.encode("ascii")
  capsule = pycapsule_new(xla_c_rule.address, b"xla._CUSTOM_CALL_TARGET")
  xla_client.register_custom_call_target(target_name, capsule, "gpu")
  return target_name


def _func_translation(func, abs_eval_fn, c, *args):
  input_shapes = [c.get_shape(arg) for arg in args]
  input_dtypes = tuple(shape.element_type() for shape in input_shapes)
  input_dimensions = tuple(shape.dimensions() for shape in input_shapes)

  output_abstract_arrays = abs_eval_fn(
    *tuple(ShapedArray(shape.dimensions(), shape.element_type()) for shape in input_shapes)
  )

  output_shapes = tuple(array.shape for array in output_abstract_arrays)
  output_dtypes = tuple(array.dtype for array in output_abstract_arrays)

  output_layouts = map(lambda shape: range(len(shape) - 1, -1, -1), output_shapes)
  xla_output_shapes = [
    x_shape(*arg) for arg in zip(output_dtypes, output_shapes, output_layouts)
  ]
  xla_output_shape = xla_client.Shape.tuple_shape(xla_output_shapes)

  target_name = compile_gpu_signature(func,
                                      input_dtypes, input_dimensions,
                                      output_dtypes, output_shapes)

  return x_ops.CustomCallWithLayout(
    c,
    target_name,
    operands=args,
    operand_shapes_with_layout=input_shapes,
    shape_with_layout=xla_output_shape,
  )


def register_op_gpu(func, abs_eval):
  if not _cuda.numba_cffi_loaded:
    raise RuntimeError("Numba cffi could not be loaded.")

  _func_prim = core.Primitive(func.__name__)
  _func_prim.multiple_results = True

  if callable(abs_eval):
    abs_eval_fn = abs_eval
  else:
    abs_eval_fn = lambda *args: abs_eval

  _func_abstract = partial(abs_eval_rule, abs_eval_fn)
  bind_primitive_fn = partial(bind_primitive, _func_prim, abs_eval_fn)

  func = cuda.jit(func)
  _func_prim.def_abstract_eval(_func_abstract)
  _func_prim.def_impl(partial(eval_rule, func, _func_abstract))

  xla.backend_specific_translations['gpu'][_func_prim] = partial(_func_translation, func, _func_abstract)

  return jax.jit(bind_primitive_fn)
