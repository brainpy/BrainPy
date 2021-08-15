# -*- coding: utf-8 -*-


"""
The compilation tools for JAX backend.

1. JIT compilation is implemented by the 'jit()' function
2. Vectorize compilation is implemented by the 'vmap()' function
3. Parallel compilation is implemented by the 'pmap()' function

"""

import functools

import jax

from brainpy import errors

from brainpy.tools.codes import change_func_name

__all__ = [
  'jit',
  'vmap',
  'pmap',
]


def _make_jit(all_vars, func, static_argnums, static_argnames=None, device=None,
              backend=None, donate_argnums=(), inline=False, f_name=None):
  @functools.partial(jax.jit, static_argnums=static_argnums,
                     static_argnames=static_argnames,
                     device=device, backend=backend,
                     donate_argnums=donate_argnums,
                     inline=inline)
  def jitted_func(all_data, *args, **kwargs):
    all_vars.unique_assign(all_data)
    out = func(*args, **kwargs)
    changed_data = all_vars.unique_data()
    return out, changed_data

  def call(*args, **kwargs):
    data = all_vars.unique_data()
    out, changed_data = jitted_func(data, *args, **kwargs)
    all_vars.unique_assign(changed_data)
    return out

  return change_func_name(name=f_name, f=call) if f_name else call


def jit(obj_or_func, static_argnums=None, static_argnames=None, device=None,
        backend=None, donate_argnums=(), inline=False):
  """JIT (Just-In-Time) Compilation.

  Takes a function or an instance of DynamicSystem,
  and compiles it for faster execution.

  Parameters
  ----------
  obj_or_func : Primary, function
    The instance of DynamicSystem or a function.
  static_argnums : Optional, str
    An optional int or collection of ints that specify which
    positional arguments to treat as static (compile-time constant).
  static_argnames : optional, str
    An optional string or collection of strings specifying
    which named arguments to treat as static (compile-time constant). See the
    comment on ``static_argnums`` for details. If not
    provided but ``static_argnums`` is set, the default is based on calling
    ``inspect.signature(fun)`` to find corresponding named arguments.
  device: optional, Any
    This is an experimental feature and the API is likely to change.
    Optional, the Device the jitted function will run on. (Available devices
    can be retrieved via :py:func:`jax.devices`.) The default is inherited
    from XLA's DeviceAssignment logic and is usually to use
    ``jax.devices()[0]``.
  backend: optional, str
    This is an experimental feature and the API is likely to change.
    Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
    ``'tpu'``.
  donate_argnums:
    Specify which arguments are "donated" to the computation.
    It is safe to donate arguments if you no longer need them once the
    computation has finished. In some cases XLA can make use of donated
    buffers to reduce the amount of memory needed to perform a computation,
    for example recycling one of your input buffers to store a result. You
    should not reuse buffers that you donate to a computation, JAX will raise
    an error if you try to. By default, no arguments are donated.
  inline: bool
    Specify whether this function should be inlined into enclosing
    jaxprs (rather than being represented as an application of the xla_call
    primitive with its own subjaxpr). Default False.

  Returns
  -------
  ds_of_func : DynamicSystem, function
    A wrapped version of DynamicSystem or function,
    set up for just-in-time compilation.
  """
  from brainpy.primary.base import Primary
  from brainpy.simulation.brainobjects.base import DynamicSystem

  if isinstance(obj_or_func, DynamicSystem):
    # DynamicSystem has step functions
    if len(obj_or_func.steps):
      for key in obj_or_func.steps.keys():
        static_argnums = tuple(x + 1 for x in sorted(static_argnums or ()))
        step = obj_or_func.steps[key]
        all_vars = step.__self__.vars()
        obj_or_func.steps[key] = _make_jit(all_vars=all_vars,
                                           func=step,
                                           static_argnums=static_argnums,
                                           static_argnames=static_argnames,
                                           device=device,
                                           backend=backend,
                                           donate_argnums=donate_argnums,
                                           inline=inline,
                                           f_name=key)
      return obj_or_func
    else:
      raise NotImplementedError

  elif isinstance(obj_or_func, Primary):
    # Primary has '__call__()' function implementation
    if callable(obj_or_func):
      static_argnums = tuple(x + 1 for x in sorted(static_argnums or ()))
      all_vars = obj_or_func.vars()
      obj_or_func.__call__ = _make_jit(all_vars=all_vars,
                                       func=obj_or_func.__call__,
                                       static_argnums=static_argnums,
                                       static_argnames=static_argnames,
                                       device=device,
                                       backend=backend,
                                       donate_argnums=donate_argnums,
                                       inline=inline)
      return obj_or_func

    else:
      raise errors.ModelUseError(f'Cannot JIT {obj_or_func}, because it does not have '
                                 f'step functions (len(steps) = 0) and not implement '
                                 f'"__call__" function. ')

  elif callable(obj_or_func):
    # function
    return jax.jit(obj_or_func,
                   static_argnums=static_argnums,
                   static_argnames=static_argnames,
                   device=device,
                   backend=backend,
                   donate_argnums=donate_argnums,
                   inline=inline)

  else:
    raise errors.ModelUseError(f'Only support instance of '
                               f'{DynamicSystem.__name__}, '
                               f'or a callable function, '
                               f'but we got {type(obj_or_func)}.')


def vmap(f):
  return f


def pmap(f):
  return f
