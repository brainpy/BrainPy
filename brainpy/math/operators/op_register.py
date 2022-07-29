# -*- coding: utf-8 -*-

from typing import Union, Sequence, Callable

from jax.abstract_arrays import ShapedArray
from jax.tree_util import tree_map

from brainpy.base import Base
from brainpy.math.jaxarray import JaxArray
from brainpy import tools
from .utils import _check_brainpylib

try:
  import brainpylib
except ModuleNotFoundError:
  brainpylib = None

__all__ = [
  'XLACustomOp',
  'register_op',
]


class XLACustomOp(Base):
  def __init__(self, name=None, apply_cpu_func_to_gpu: bool = False):
    _check_brainpylib(register_op.__name__)
    super(XLACustomOp, self).__init__(name=name)

    # abstract evaluation function
    if hasattr(self.eval_shape, 'not_customized') and self.eval_shape.not_customized:
      raise ValueError('Must implement "eval_shape" for abstract evaluation.')

    # cpu function
    if hasattr(self.con_compute, 'not_customized') and self.con_compute.not_customized:
      if hasattr(self.cpu_func, 'not_customized') and self.cpu_func.not_customized:
        raise ValueError('Must implement one of "cpu_func" or "con_compute".')
      else:
        cpu_func = self.cpu_func
    else:
      cpu_func = self.con_compute

    # gpu function
    if hasattr(self.gpu_func, 'not_customized') and self.gpu_func.not_customized:
      gpu_func = None
    else:
      gpu_func = self.gpu_func

    # register OP
    self.op = brainpylib.register_op(self.name,
                                     cpu_func=cpu_func,
                                     gpu_func=gpu_func,
                                     out_shapes=self.eval_shape,
                                     apply_cpu_func_to_gpu=apply_cpu_func_to_gpu)

  @tools.not_customized
  def eval_shape(self, *args, **kwargs):
    raise NotImplementedError

  @staticmethod
  @tools.not_customized
  def con_compute(*args, **kwargs):
    raise NotImplementedError

  @staticmethod
  @tools.not_customized
  def cpu_func(*args, **kwargs):
    raise NotImplementedError

  @staticmethod
  @tools.not_customized
  def gpu_func(*args, **kwargs):
    raise NotImplementedError

  def __call__(self, *args, **kwargs):
    args = tree_map(lambda a: a.value if isinstance(a, JaxArray) else a,
                    args, is_leaf=lambda a: isinstance(a, JaxArray))
    kwargs = tree_map(lambda a: a.value if isinstance(a, JaxArray) else a,
                      kwargs, is_leaf=lambda a: isinstance(a, JaxArray))
    res = self.op.bind(*args, **kwargs)
    return res[0] if len(res) == 1 else res


def register_op(
    op_name: str,
    cpu_func: Callable,
    out_shapes: Union[Callable, ShapedArray, Sequence[ShapedArray]],
    gpu_func: Callable = None,
    apply_cpu_func_to_gpu: bool = False
):
  """
  Converting the numba-jitted function in a Jax/XLA compatible primitive.

  Parameters
  ----------
  op_name: str
    Name of the operators.
  cpu_func: Callble
    A callable numba-jitted function or pure function (can be lambda function) running on CPU.
  gpu_func: Callable, default = None
    A callable cuda-jitted kernel running on GPU.
  out_shapes: Callable, ShapedArray, Sequence[ShapedArray], default = None
    Outputs shapes of target function. `out_shapes` can be a `ShapedArray` or
    a sequence of `ShapedArray`. If it is a function, it takes as input the argument
    shapes and dtypes and should return correct output shapes of `ShapedArray`.
  apply_cpu_func_to_gpu: bool, default = False
    True when gpu_func is implemented on CPU and other logics(data transfer) is implemented on GPU.

  Returns
  -------
  A jitable JAX function.
  """
  _check_brainpylib(register_op.__name__)
  f = brainpylib.register_op(op_name,
                             cpu_func=cpu_func,
                             gpu_func=gpu_func,
                             out_shapes=out_shapes,
                             apply_cpu_func_to_gpu=apply_cpu_func_to_gpu)

  def fixed_op(*inputs):
    inputs = tuple([i.value if isinstance(i, JaxArray) else i for i in inputs])
    return f(*inputs)

  return fixed_op
