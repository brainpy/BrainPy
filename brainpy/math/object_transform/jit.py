# -*- coding: utf-8 -*-

"""
The JIT compilation tools for JAX backend.

1. Just-In-Time compilation is implemented by the 'jit()' function

"""

from typing import Callable, Union, Optional, Sequence, Dict, Any

import jax

try:
  from jax.errors import UnexpectedTracerError, ConcretizationTypeError
except ImportError:
  from jax.core import UnexpectedTracerError, ConcretizationTypeError

from brainpy import errors, tools, check
from .base_transform import ObjectTransform
from .base_object import BrainPyObject
from ..ndarray import Variable, add_context, del_context

__all__ = [
  'jit',
]


class JITTransform(ObjectTransform):
  """Object-oriented JIT transformation in BrainPy."""

  def __init__(
      self,
      target: callable,
      dyn_vars: Dict[str, Variable],
      child_objs: Dict[str, BrainPyObject],
      static_argnames: Optional[Any] = None,
      device: Optional[Any] = None,
      name: Optional[str] = None
  ):
    super().__init__(name=name)

    self.register_implicit_vars(dyn_vars)
    self.register_implicit_nodes(child_objs)

    self.target = target
    self._all_vars = self.vars().unique()

    # transformation
    self._f = jax.jit(self._transform_function, static_argnames=static_argnames, device=device)

  def _transform_function(self, variable_data: Dict, *args, **kwargs):
    for key, v in self._all_vars.items():
      v._value = variable_data[key]
    out = self.target(*args, **kwargs)
    changes = self._all_vars.dict()
    return out, changes

  def __call__(self, *args, **kwargs):
    variable_data = self._all_vars.dict()
    try:
      add_context(self.name)
      out, changes = self._f(variable_data, *args, **kwargs)
      del_context(self.name)
    except UnexpectedTracerError as e:
      del_context(self.name)
      for key, v in self._all_vars.items(): v._value = variable_data[key]
      raise errors.JaxTracerError(variables=self._all_vars) from e
    except ConcretizationTypeError as e:
      del_context(self.name)
      for key, v in self._all_vars.items(): v._value = variable_data[key]
      raise errors.ConcretizationTypeError() from e
    except Exception as e:
      del_context(self.name)
      for key, v in self._all_vars.items(): v._value = variable_data[key]
      raise e
    else:
      for key, v in self._all_vars.items(): v._value = changes[key]
    return out

  def __repr__(self):
    name = self.__class__.__name__
    f = tools.repr_object(self.target)
    f = tools.repr_context(f, " " * (len(name) + 6))
    format_ref = (f'{name}(target={f}, \n' +
                  f'{" " * len(name)} num_of_vars={len(self.vars().unique())})')
    return format_ref


def jit(
    func: Callable,
    dyn_vars: Optional[Union[Variable, Sequence[Variable], Dict[str, Variable]]] = None,
    child_objs: Optional[Union[BrainPyObject, Sequence[BrainPyObject], Dict[str, BrainPyObject]]] = None,
    static_argnames: Optional[Union[str, Any]] = None,
    device: Optional[Any] = None,
) -> JITTransform:
  """JIT (Just-In-Time) compilation for class objects.

  This function has the same ability to Just-In-Time compile a pure function,
  but it can also JIT compile a :py:class:`brainpy.DynamicalSystem`, or a
  :py:class:`brainpy.Base` object, or a bounded method for a
  :py:class:`brainpy.Base` object.

  .. note::
    There are several notes when using JIT compilation.

    1. Avoid using scalar in a Variable, TrainVar, etc.

    For example,

    >>> import brainpy as bp
    >>> import brainpy.math as bm
    >>>
    >>> class Test(bp.BrainPyObject):
    >>>   def __init__(self):
    >>>     super(Test, self).__init__()
    >>>     self.a = bm.Variable(1.)  # Avoid! DO NOT USE!
    >>>   def __call__(self, *args, **kwargs):
    >>>     self.a += 1.

    The above usage is deprecated, because it may cause several errors.
    Instead, we recommend you define the scalar value variable as:

    >>> class Test(bp.BrainPyObject):
    >>>   def __init__(self):
    >>>     super(Test, self).__init__()
    >>>     self.a = bm.Variable(bm.array([1.]))  # use array to wrap a scalar is recommended
    >>>   def __call__(self, *args, **kwargs):
    >>>     self.a += 1.

    Here, a ndarray is recommended to used to update the variable ``a``.

    2. ``jit`` compilation in ``brainpy.math`` does not support `static_argnums`.
       Instead, users should use `static_argnames`, and call the jitted function with
       keywords like ``jitted_func(arg1=var1, arg2=var2)``. For example,

    >>> def f(a, b, c=1.):
    >>>   if c > 0.: return a + b
    >>>   else: return a * b
    >>>
    >>> # ERROR! https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-jit
    >>> bm.jit(f)(1, 2, 0)
    jax._src.errors.ConcretizationTypeError: Abstract tracer value encountered where
    concrete value is expected: Traced<ShapedArray(bool[], weak_type=True)
    >>> # this is right
    >>> bm.jit(f, static_argnames='c')(1, 2, 0)
    DeviceArray(2, dtype=int32, weak_type=True)

  Examples
  --------

  You can JIT a :py:class:`brainpy.DynamicalSystem`

  >>> import brainpy as bp
  >>>
  >>> class LIF(bp.NeuGroup):
  >>>   pass
  >>> lif = bp.math.jit(LIF(10))

  You can JIT a :py:class:`brainpy.Base` object with ``__call__()`` implementation.

  >>> mlp = bp.layers.GRU(100, 200)
  >>> jit_mlp = bp.math.jit(mlp)

  You can also JIT a bounded method of a :py:class:`brainpy.Base` object.

  >>> class Hello(bp.BrainPyObject):
  >>>   def __init__(self):
  >>>     super(Hello, self).__init__()
  >>>     self.a = bp.math.Variable(bp.math.array(10.))
  >>>     self.b = bp.math.Variable(bp.math.array(2.))
  >>>   def transform(self):
  >>>     return self.a ** self.b
  >>>
  >>> test = Hello()
  >>> bp.math.jit(test.transform)

  Further, you can JIT a normal function, just used like in JAX.

  >>> @bp.math.jit
  >>> def selu(x, alpha=1.67, lmbda=1.05):
  >>>   return lmbda * bp.math.where(x > 0, x, alpha * bp.math.exp(x) - alpha)


  Parameters
  ----------
  func : Base, function, callable
    The instance of Base or a function.
  dyn_vars : optional, dict, sequence of Variable, Variable
    These variables will be changed in the function, or needed in the computation.
  child_objs: optional, dict, sequence of BrainPyObject, BrainPyObject
    The children objects used in the target function.

    .. versionadded:: 2.3.1
  static_argnames : optional, str, list, tuple, dict
    An optional string or collection of strings specifying which named arguments to treat
    as static (compile-time constant). See the comment on ``static_argnums`` for details.
    If not provided but ``static_argnums`` is set, the default is based on calling
    ``inspect.signature(fun)`` to find corresponding named arguments.
  device: optional, Any
    This is an experimental feature and the API is likely to change.
    Optional, the Device the jitted function will run on. (Available devices
    can be retrieved via :py:func:`jax.devices`.) The default is inherited
    from XLA's DeviceAssignment logic and is usually to use
    ``jax.devices()[0]``.

  Returns
  -------
  func : JITTransform
    A callable jitted function, set up for just-in-time compilation.
  """
  if callable(func):
    dyn_vars = check.is_all_vars(dyn_vars, out_as='dict')
    child_objs = check.is_all_objs(child_objs, out_as='dict')

    # BrainPyObject object which implements __call__,
    # or bounded method of BrainPyObject object
    return JITTransform(target=func,
                        dyn_vars=dyn_vars,
                        child_objs=child_objs,
                        static_argnames=static_argnames,
                        device=device)

  else:
    raise errors.BrainPyError(f'Only support instance of {BrainPyObject.__name__}, or a callable '
                              f'function, but we got {type(func)}.')
