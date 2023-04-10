# -*- coding: utf-8 -*-

"""
The JIT compilation tools for JAX backend.

1. Just-In-Time compilation is implemented by the 'jit()' function

"""

from functools import partial, wraps
from typing import Callable, Union, Optional, Sequence, Dict, Any, Iterable

import jax

from brainpy import tools, check
from brainpy._src.math.ndarray import Variable, VariableStack
from brainpy._src.math.object_transform.naming import get_stack_cache, cache_stack
from ._abstract import ObjectTransform
from ._tools import dynvar_deprecation, node_deprecation, evaluate_dyn_vars
from .base import BrainPyObject

__all__ = [
  'jit',
]


class JITTransform(ObjectTransform):
  """Object-oriented JIT transformation in BrainPy."""

  def __init__(
      self,
      target: callable,
      static_argnums: Union[int, Iterable[int], None] = None,
      static_argnames: Union[str, Iterable[str], None] = None,
      device: Optional[Any] = None,
      inline: bool = False,
      keep_unused: bool = False,
      abstracted_axes: Optional[Any] = None,
      name: Optional[str] = None,

      # deprecated
      dyn_vars: Dict[str, Variable] = None,
      child_objs: Dict[str, BrainPyObject] = None,
  ):
    super().__init__(name=name)

    # variables and nodes
    if dyn_vars is not None:
      self.register_implicit_vars(dyn_vars)
    if child_objs is not None:
      self.register_implicit_nodes(child_objs)

    # target
    if hasattr(target, '__self__') and isinstance(getattr(target, '__self__'), BrainPyObject):
      self.register_implicit_nodes(getattr(target, '__self__'))
    self.target = target

    # parameters
    self._static_argnums = static_argnums
    self._static_argnames = static_argnames
    self._device = device
    self._inline = inline
    self._keep_unused = keep_unused
    self._abstracted_axes = abstracted_axes

    # transformation function
    self._transform = None
    self._dyn_vars = None

  def _transform_function(self, variable_data: Dict, *args, **kwargs):
    for key, v in self._dyn_vars.items():
      v._value = variable_data[key]
    out = self.target(*args, **kwargs)
    changes = self._dyn_vars.dict_data()
    return out, changes

  def __call__(self, *args, **kwargs):
    if self._transform is None:
      self._dyn_vars = evaluate_dyn_vars(self.target, *args, **kwargs)
      self._transform = jax.jit(
        self._transform_function,
        static_argnums=jax.tree_util.tree_map(lambda a: a + 1, self._static_argnums),
        static_argnames=self._static_argnames,
        device=self._device,
        inline=self._inline,
        keep_unused=self._keep_unused,
        abstracted_axes=self._abstracted_axes
      )
    out, changes = self._transform(self._dyn_vars.dict_data(), *args, **kwargs)
    for key, v in self._dyn_vars.items():
      v._value = changes[key]
    return out

  def __repr__(self):
    name = self.__class__.__name__
    f = tools.repr_object(self.target)
    f = tools.repr_context(f, " " * (len(name) + 6))
    format_ref = (f'{name}(target={f}, \n' +
                  f'{" " * len(name)} num_of_vars={len(self.vars().unique())})')
    return format_ref


def jit(
    func: Callable = None,
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
    device: Optional[Any] = None,
    inline: bool = False,
    keep_unused: bool = False,
    abstracted_axes: Optional[Any] = None,

    # deprecated
    dyn_vars: Optional[Union[Variable, Sequence[Variable], Dict[str, Variable]]] = None,
    child_objs: Optional[Union[BrainPyObject, Sequence[BrainPyObject], Dict[str, BrainPyObject]]] = None,
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

  >>> mlp = bp.layers.Linear(100, 200)
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
  dyn_vars : optional, dict, sequence of Variable, Variable
    These variables will be changed in the function, or needed in the computation.

    .. deprecated:: 2.4.0
       No longer need to provide ``dyn_vars``. This function is capable of automatically
       collecting the dynamical variables used in the target ``func``.
  child_objs: optional, dict, sequence of BrainPyObject, BrainPyObject
    The children objects used in the target function.

    .. versionadded:: 2.3.1

    .. deprecated:: 2.4.0
       No longer need to provide ``child_objs``. This function is capable of automatically
       collecting the children objects used in the target ``func``.

  Returns
  -------
  func : callable
    A callable jitted function, set up for just-in-time compilation.
  """

  dynvar_deprecation(dyn_vars)
  node_deprecation(child_objs)
  if dyn_vars is not None:
    dyn_vars = check.is_all_vars(dyn_vars, out_as='dict')
  if child_objs is not None:
    child_objs = check.is_all_objs(child_objs, out_as='dict')

  if func is None:
    return lambda f: JITTransform(target=f,
                                  dyn_vars=dyn_vars,
                                  child_objs=child_objs,
                                  static_argnums=static_argnums,
                                  static_argnames=static_argnames,
                                  device=device,
                                  inline=inline,
                                  keep_unused=keep_unused,
                                  abstracted_axes=abstracted_axes)
  else:
    return JITTransform(target=func,
                        dyn_vars=dyn_vars,
                        child_objs=child_objs,
                        static_argnums=static_argnums,
                        static_argnames=static_argnames,
                        device=device,
                        inline=inline,
                        keep_unused=keep_unused,
                        abstracted_axes=abstracted_axes)


def cls_jit(
    func: Callable = None,
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
    device: Optional[Any] = None,
    inline: bool = False,
    keep_unused: bool = False,
    abstracted_axes: Optional[Any] = None,
) -> Callable:
  if func is None:
    return lambda f: _make_jit_fun(fun=f,
                                   static_argnums=static_argnums,
                                   static_argnames=static_argnames,
                                   device=device,
                                   inline=inline,
                                   keep_unused=keep_unused,
                                   abstracted_axes=abstracted_axes)
  else:
    return _make_jit_fun(fun=func,
                         static_argnums=static_argnums,
                         static_argnames=static_argnames,
                         device=device,
                         inline=inline,
                         keep_unused=keep_unused,
                         abstracted_axes=abstracted_axes)


def _make_jit_fun(
    fun: Callable,
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
    device: Optional[Any] = None,
    inline: bool = False,
    keep_unused: bool = False,
    abstracted_axes: Optional[Any] = None,
):
  @wraps(fun)
  def call_fun(self, *args, **kwargs):
    fun2 = partial(fun, self)
    cache = get_stack_cache(fun2)  # TODO: better cache mechanism
    if cache is None:
      with jax.ensure_compile_time_eval():
        args_, kwargs_ = jax.tree_util.tree_map(jax.api_util.shaped_abstractify, (args, kwargs))
        with VariableStack() as stack:
          _ = jax.eval_shape(fun2, *args_, **kwargs_)
        del args_, kwargs_
      _transform = jax.jit(
        _make_transform(fun2, stack),
        static_argnums=jax.tree_util.tree_map(lambda a: a + 1, static_argnums),
        static_argnames=static_argnames,
        device=device,
        inline=inline,
        keep_unused=keep_unused,
        abstracted_axes=abstracted_axes
      )
      cache_stack(fun2, (stack, _transform))  # cache
    else:
      stack, _transform = cache
    del cache
    out, changes = _transform(stack.dict_data(), *args, **kwargs)
    for key, v in stack.items():
      v._value = changes[key]
    return out

  return call_fun


def _make_transform(fun, stack):
  def _transform_function(variable_data: dict, *args, **kwargs):
    for key, v in stack.items():
      v._value = variable_data[key]
    out = fun(*args, **kwargs)
    changes = stack.dict_data()
    return out, changes

  return _transform_function


def cls_jit_inline(
    func: Callable = None,
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
    device: Optional[Any] = None,
    keep_unused: bool = False,
    abstracted_axes: Optional[Any] = None,
) -> Callable:
  return cls_jit(func=func,
                 static_argnums=static_argnums,
                 static_argnames=static_argnames,
                 inline=True,
                 device=device,
                 keep_unused=keep_unused,
                 abstracted_axes=abstracted_axes)