# -*- coding: utf-8 -*-

"""
The JIT compilation tools for JAX backend.

1. Just-In-Time compilation is implemented by the 'jit()' function

"""

import functools
import logging

import jax

try:
  from jax.errors import UnexpectedTracerError, ConcretizationTypeError
except ImportError:
  from jax.core import UnexpectedTracerError, ConcretizationTypeError

from brainpy import errors
from brainpy.base.base import Base
from brainpy.base.naming import get_unique_name
from brainpy.base.collector import TensorCollector
from brainpy.math.jaxarray import JaxArray, add_context, del_context
from brainpy.tools.codes import change_func_name

__all__ = [
  'jit',
]

logger = logging.getLogger('brainpy.math.jit')


def _make_jit_with_vars(func, vars, static_argnames=None, device=None, f_name=None):
  @functools.partial(jax.jit, static_argnames=static_argnames, device=device)
  def jitted_func(variable_data, *args, **kwargs):
    for key, v in vars.items(): v._value = variable_data[key]
    out = func(*args, **kwargs)
    changes = vars.dict()
    return out, changes

  name = get_unique_name('_brainpy_object_oriented_jit_')

  def call(*args, **kwargs):
    variable_data = vars.dict()
    try:
      add_context(name)
      out, changes = jitted_func(variable_data, *args, **kwargs)
      del_context(name)
    except UnexpectedTracerError as e:
      del_context(name)
      for key, v in vars.items(): v._value = variable_data[key]
      raise errors.JaxTracerError(variables=vars) from e
    except ConcretizationTypeError as e:
      del_context(name)
      for key, v in vars.items(): v._value = variable_data[key]
      raise errors.ConcretizationTypeError() from e
    except Exception as e:
      del_context(name)
      for key, v in vars.items(): v._value = variable_data[key]
      raise e
    for key, v in vars.items(): v._value = changes[key]
    return out

  return change_func_name(name=f_name, f=call) if f_name else call


def _make_jit_without_vars(func, static_argnames=None, device=None, f_name=None):
  jit_f = jax.jit(func, static_argnames=static_argnames, device=device)
  name = get_unique_name('_jax_functional_jit_')

  def call(*args, **kwargs):
    add_context(name)
    r = jit_f(*args, **kwargs)
    del_context(name)
    return r

  return change_func_name(name=f_name, f=call) if f_name else call


def jit(func, dyn_vars=None, static_argnames=None, device=None, auto_infer=True):
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
    >>> class Test(bp.Base):
    >>>   def __init__(self):
    >>>     super(Test, self).__init__()
    >>>     self.a = bm.Variable(1.)  # Avoid! DO NOT USE!
    >>>   def __call__(self, *args, **kwargs):
    >>>     self.a += 1.

    The above usage is deprecated, because it may cause several errors.
    Instead, we recommend you define the scalar value variable as:

    >>> class Test(bp.Base):
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

  >>> class Hello(bp.Base):
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
  dyn_vars : optional, dict, tuple, list, JaxArray
    These variables will be changed in the function, or needed in the computation.
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
  auto_infer : bool
    Automatical infer the dynamical variables.

  Returns
  -------
  func : callable
    A callable jitted function, set up for just-in-time compilation.
  """
  if callable(func):
    if dyn_vars is not None:
      if isinstance(dyn_vars, JaxArray):
        dyn_vars = TensorCollector({'_': dyn_vars})
      elif isinstance(dyn_vars, dict):
        dyn_vars = TensorCollector(dyn_vars)
      elif isinstance(dyn_vars, (tuple, list)):
        dyn_vars = TensorCollector({f'_v{i}': v for i, v in enumerate(dyn_vars)})
      else:
        raise ValueError
    else:
      if auto_infer:
        if isinstance(func, Base):
          dyn_vars = func.vars().unique()
        elif hasattr(func, '__self__') and isinstance(func.__self__, Base):
          dyn_vars = func.__self__.vars().unique()
        else:
          dyn_vars = TensorCollector()
      else:
        dyn_vars = TensorCollector()

    if len(dyn_vars) == 0:  # pure function
      return _make_jit_without_vars(func, static_argnames=static_argnames, device=device)

    else:  # Base object which implements __call__, or bounded method of Base object
      return _make_jit_with_vars(vars=dyn_vars,
                                 func=func,
                                 static_argnames=static_argnames,
                                 device=device)

  else:
    raise errors.BrainPyError(f'Only support instance of {Base.__name__}, or a callable '
                              f'function, but we got {type(func)}.')
