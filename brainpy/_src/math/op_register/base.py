from functools import partial
from typing import Callable, Sequence, Tuple, Protocol, Optional, Union

import jax
import numpy as np
from jax.interpreters import xla, batching, ad, mlir

from brainpy._src.dependency_check import import_numba, import_cupy_jit
from brainpy._src.math.ndarray import Array
from brainpy._src.math.object_transform.base import BrainPyObject

if jax.__version__ >= '0.4.16':
  from .numba_based import register_numba_mlir_cpu_translation_rule as register_numba_cpu_translation_rule
  from .taichi_aot_based import (register_taichi_aot_mlir_cpu_translation_rule as register_taichi_cpu_translation_rule,
                                 register_taichi_aot_mlir_gpu_translation_rule as register_taichi_gpu_translation_rule)
  from .cupy_based import (register_cupy_raw_module_mlir_gpu_translation_rule as register_cupy_raw_module_gpu_translation_rule,
                            register_cupy_jit_kernel_mlir_gpu_translation_rule as register_cupy_jit_kernel_gpu_translation_rule)
else:
  from .numba_based import register_numba_xla_cpu_translation_rule as register_numba_cpu_translation_rule
  from .taichi_aot_based import (register_taichi_aot_xla_cpu_translation_rule as register_taichi_cpu_translation_rule,
                                 register_taichi_aot_xla_gpu_translation_rule as register_taichi_gpu_translation_rule)
  from .cupy_based import (register_cupy_raw_module_xla_gpu_translation_rule as register_cupy_raw_module_gpu_translation_rule,
                            register_cupy_jit_kernel_xla_gpu_translation_rule as register_cupy_jit_kernel_gpu_translation_rule)
from .utils import register_general_batching
from brainpy._src.math.op_register.ad_support import defjvp

numba = import_numba(error_if_not_found=False)
cp_jit = import_cupy_jit(error_if_not_found=False)

__all__ = [
  'XLACustomOp',
]


class ShapeDtype(Protocol):

  @property
  def shape(self) -> Tuple[int, ...]:
    ...

  @property
  def dtype(self) -> np.dtype:
    ...


class XLACustomOp(BrainPyObject):
  """Creating a XLA custom call operator.

  For more information, please refer to the tutorials above:
  Numba Custom Op: https://brainpy.tech/docs/tutorial_advanced/operator_custom_with_numba.html
  Taichi Custom Op: https://brainpy.tech/docs/tutorial_advanced/operator_custom_with_taichi.html
  CuPy Custom Op: https://brainpy.tech/docs/tutorial_advanced/operator_custom_with_cupy.html

  Args:
    cpu_kernel: Callable. The function defines the computation on CPU backend.
    gpu_kernel: Callable. The function defines the computation on GPU backend.
    batching_translation: Callable. The batching translation rule of JAX.
    jvp_translation: Callable. The JVP translation rule of JAX.
    transpose_translation: Callable. The transpose translation rule of JAX.
    outs: optional. The output information.
    name: str. The primitive name.
  """

  def __init__(
      self,
      cpu_kernel: Callable = None,
      gpu_kernel: Union[Callable, str] = None,
      batching_translation: Callable = None,
      jvp_translation: Callable = None,
      transpose_translation: Callable = None,
      outs: Optional[Callable] = None,
      name: str = None,
  ):
    super().__init__(name)

    # set cpu_kernel and gpu_kernel
    self.cpu_kernel = cpu_kernel
    self.gpu_kernel = gpu_kernel

    # primitive
    self.primitive = jax.core.Primitive(self.name)
    self.primitive.multiple_results = True

    # abstract evaluation
    self.outs = outs
    self.primitive.def_abstract_eval(_abstract_eval)
    self.primitive.def_impl(partial(xla.apply_primitive, self.primitive))

    # cpu function
    cpu_checked = False
    if cpu_kernel is None:
      cpu_checked = True
    if numba is not None:  # numba
      from numba.core.dispatcher import Dispatcher
      if isinstance(cpu_kernel, Dispatcher):
        register_numba_cpu_translation_rule(self.primitive, cpu_kernel)
        cpu_checked = True
    if hasattr(cpu_kernel, '_is_wrapped_kernel') and cpu_kernel._is_wrapped_kernel:  # taichi
      register_taichi_cpu_translation_rule(self.primitive, cpu_kernel)
      cpu_checked = True
    if not cpu_checked:
      raise ValueError(f'"cpu_kernel" must be a numba jitted function or a taichi kernel function. '
                       f'But we got {cpu_kernel}')

    # gpu function
    gpu_checked = False
    if gpu_kernel is None:
      gpu_checked = True
    elif hasattr(gpu_kernel, 'kernel'):  # cupy RawModule
      register_cupy_raw_module_gpu_translation_rule(self.primitive, gpu_kernel)
      gpu_checked = True
    elif hasattr(gpu_kernel, '_mode'):  # cupy JIT Kernel
      register_cupy_jit_kernel_gpu_translation_rule(self.primitive, gpu_kernel)
      gpu_checked = True
    elif hasattr(gpu_kernel, '_is_wrapped_kernel') and gpu_kernel._is_wrapped_kernel:  # taichi
      register_taichi_gpu_translation_rule(self.primitive, gpu_kernel)
      gpu_checked = True
    if not gpu_checked:
      raise ValueError(f'"gpu_kernel" must be a taichi kernel function, cupy raw module or cupy jit kernel. But we got {gpu_kernel}')

    # batching rule
    if batching_translation is None:
      register_general_batching(self.primitive)
    else:
      batching.primitive_batchers[self.primitive] = batching_translation

    # jvp rule
    if jvp_translation is not None:
      ad.primitive_jvps[self.primitive] = jvp_translation

    # transpose rule
    if transpose_translation is not None:
      ad.primitive_transposes[self.primitive] = transpose_translation

  def __call__(self, *ins, outs: Optional[Sequence[ShapeDtype]] = None, **kwargs):
    if outs is None:
      if self.outs is None:
        raise ValueError('The output information is not defined.')
      outs = self.outs(*ins, **kwargs)
    assert outs is not None
    outs = tuple([_transform_to_shapedarray(o) for o in outs])
    ins = jax.tree_util.tree_map(_transform_to_array, ins, is_leaf=_is_bp_array)
    return self.primitive.bind(*ins, outs=outs, **kwargs)

  def def_abstract_eval(self, fun):
    """Define the abstract evaluation function.

    Args:
      fun: The abstract evaluation function.
    """
    self.primitive.def_abstract_eval(fun)

  def def_batching_rule(self, fun):
    """Define the batching rule.

    Args:
      fun: The batching rule.
    """
    batching.primitive_batchers[self.primitive] = fun

  def def_jvp_rule(self, fun):
    """Define the JVP rule.

    Args:
      fun: The JVP rule.
    """
    ad.primitive_jvps[self.primitive] = fun

  def defjvp(self, *jvp_rules):
    """Define the JVP rule. Similar to ``jax.interpreters.ad.defjvp``, but supports the Primitive with multiple results.

    Args:
      jvp_rules: The JVP rules.
    """
    defjvp(self.primitive, *jvp_rules)

  def def_transpose_rule(self, fun):
    """Define the transpose rule.

    Args:
      fun: The transpose rule.
    """
    ad.primitive_transposes[self.primitive] = fun

  def def_xla_translation(self, platform, fun):
    """Define the XLA translation rule.

    Args:
      platform: str. The computing platform.
      fun: The XLA translation rule.
    """
    xla.backend_specific_translations[platform][self.primitive] = fun

  def def_mlir_lowering(self, platform, fun):
    """Define the MLIR lowering rule.

    Args:
      platform: str. The computing platform.
      fun: The lowering rule.
    """
    mlir.register_lowering(self.primitive, fun, platform)


def _abstract_eval(*args, **kwargs):
  return [jax.core.ShapedArray(out_shape.shape, out_shape.dtype)
          for out_shape in kwargs['outs']]


def _is_bp_array(a):
  return isinstance(a, Array)


def _transform_to_array(a):
  if isinstance(a, Array):
    return a.value
  elif isinstance(a, jax.Array):
    return a
  else:
    return jax.numpy.asarray(a)


def _transform_to_shapedarray(a):
  return jax.core.ShapedArray(a.shape, a.dtype)
