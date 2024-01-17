from functools import partial
from typing import Callable, Sequence, Tuple, Protocol, Optional

import jax
import numpy as np
from jax.interpreters import xla, batching, ad, mlir
from numba.core.dispatcher import Dispatcher

from brainpy._src.math.ndarray import Array
from brainpy._src.math.object_transform.base import BrainPyObject
# if jax.__version__ >= '0.4.16':
#   from .numba_based import register_numba_mlir_cpu_translation_rule as register_numba_cpu_translation_rule
# else:
#   from .numba_based import register_numba_xla_cpu_translation_rule as register_numba_cpu_translation_rule
from .numba_based import register_numba_xla_cpu_translation_rule as register_numba_cpu_translation_rule
from .taichi_aot_based import (register_taichi_cpu_translation_rule,
                               register_taichi_gpu_translation_rule,
                               clean_caches)
from .utils import register_general_batching
from brainpy._src.math.op_register.ad_support import defjvp


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

  >>> import numba as nb
  >>> import taichi as ti
  >>> import numpy as np
  >>> import jax
  >>>
  >>> @nb.njit
  >>> def numba_cpu_fun(a, b, out_a, out_b):
  >>>     out_a[:] = a
  >>>     out_b[:] = b
  >>>
  >>> @ti.kernel
  >>>  def taichi_gpu_fun(a, b, out_a, out_b):
  >>>    for i in range(a.size):
  >>>      out_a[i] = a[i]
  >>>    for i in range(b.size):
  >>>      out_b[i] = b[i]
  >>>
  >>> # option 1
  >>> prim = XLACustomOp(cpu_kernel=numba_cpu_fun, gpu_kernel=taichi_gpu_fun)
  >>> a2, b2 = prim(np.random.random(1000), np.random.random(1000),
  >>>               outs=[jax.ShapeDtypeStruct(1000, dtype=np.float32),
  >>>                     jax.ShapeDtypeStruct(1000, dtype=np.float32)])
  >>>
  >>> # option 2
  >>> prim2 = XLACustomOp(cpu_kernel=numba_cpu_fun, gpu_kernel=taichi_gpu_fun,
  >>>                     outs=[jax.ShapeDtypeStruct(1000, dtype=np.float32),
  >>>                           jax.ShapeDtypeStruct(1000, dtype=np.float32)])
  >>> a3, b3 = prim2(np.random.random(1000), np.random.random(1000))

  Args:
    cpu_kernel: Callable. The function defines the computation on CPU backend.
    gpu_kernel: Callable. The function defines the computation on GPU backend.
    batching_translation: Callable. The batching translation rule of JAX.
    jvp_translation: Callable. The JVP translation rule of JAX.
    transpose_translation: Callable. The transpose translation rule of JAX.
    outs: optional, sequence of `ShapeDtype`. The output information.
    name: str. The primitive name.
  """

  def __init__(
      self,
      cpu_kernel: Callable = None,
      gpu_kernel: Callable = None,
      batching_translation: Callable = None,
      jvp_translation: Callable = None,
      transpose_translation: Callable = None,
      outs: Optional[Sequence[ShapeDtype]] = None,
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
    if outs is not None:
      outs = tuple([_transform_to_shapedarray(o) for o in outs])
    self.outs = outs
    self.primitive.def_abstract_eval(_abstract_eval)
    self.primitive.def_impl(partial(xla.apply_primitive, self.primitive))

    # cpu function
    if cpu_kernel is None:
      pass
    elif isinstance(cpu_kernel, Dispatcher):  # numba
      register_numba_cpu_translation_rule(self.primitive, cpu_kernel)
    elif hasattr(cpu_kernel, '_is_wrapped_kernel') and cpu_kernel._is_wrapped_kernel:  # taichi
      register_taichi_cpu_translation_rule(self.primitive, cpu_kernel)
    else:
      raise ValueError(f'"cpu_kernel" must be a numba jitted function or a taichi kernel function. '
                       f'But we got {cpu_kernel}')

    # gpu function
    if gpu_kernel is None:
      pass
    elif hasattr(gpu_kernel, '_is_wrapped_kernel') and gpu_kernel._is_wrapped_kernel:  # taichi
      register_taichi_gpu_translation_rule(self.primitive, gpu_kernel)
    else:
      raise ValueError(f'"cpu_kernel" must be a taichi kernel function. '
                       f'But we got {gpu_kernel}')

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
      outs = self.outs
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


