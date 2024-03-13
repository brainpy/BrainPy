from jax.interpreters import xla, mlir
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call
from functools import partial
from brainpy._src.dependency_check import (import_cupy)
from brainpy.errors import PackageMissingError

cp = import_cupy(error_if_not_found=False)


def _cupy_xla_gpu_translation_rule(kernel, c, *args, **kwargs):
  # TODO: implement the translation rule
  mod = cp.RawModule(code=kernel)
  # compile
  try:
    kernel_ptr = mod.get_function('kernel')
  except AttributeError:
    raise ValueError('The \'kernel\' function is not found in the module.')
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'cupy_kernel_call_gpu',

  )
  ...


def register_cupy_xla_gpu_translation_rule(primitive, gpu_kernel):
  xla.backend_specific_translations['gpu'][primitive] = partial(_cupy_xla_gpu_translation_rule, gpu_kernel)


def _cupy_mlir_gpu_translation_rule(kernel, c, *args, **kwargs):
  # TODO: implement the translation rule
  ...

def register_cupy_mlir_gpu_translation_rule(primitive, gpu_kernel):
  if cp is None:
    raise PackageMissingError("cupy", 'register cupy mlir gpu translation rule')

  rule = partial(_cupy_mlir_gpu_translation_rule, gpu_kernel)
  mlir.register_primitive_rule(primitive, rule, platform='gpu')