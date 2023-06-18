import warnings
from functools import wraps
from typing import Sequence

import jax

from brainpy._src.math.object_transform.naming import (cache_stack,
                                                       get_stack_cache)
from brainpy._src.math.object_transform.variables import VariableStack, current_transform_number


class Empty(object):
  pass


empty = Empty()


def _partial_fun(fun,
                 args: tuple,
                 kwargs: dict,
                 static_argnums: Sequence[int] = (),
                 static_argnames: Sequence[str] = ()):
  static_args, dyn_args = [], []
  for i, arg in enumerate(args):
    if i in static_argnums:
      static_args.append(arg)
    else:
      static_args.append(empty)
      dyn_args.append(arg)
  static_kwargs, dyn_kwargs = {}, {}
  for k, arg in kwargs.items():
    if k in static_argnames:
      static_kwargs[k] = arg
    else:
      dyn_kwargs[k] = arg
  del args, kwargs, static_argnums, static_argnames

  @wraps(fun)
  def new_fun(*dynargs, **dynkwargs):
    args = []
    i = 0
    for arg in static_args:
      if arg == empty:
        args.append(dynargs[i])
        i += 1
      else:
        args.append(arg)
    return fun(*args, **static_kwargs, **dynkwargs)

  return new_fun, dyn_args, dyn_kwargs


def dynvar_deprecation(dyn_vars=None):
  if dyn_vars is not None:
    warnings.warn('\n'
                  'From brainpy>=2.4.0, users no longer need to provide ``dyn_vars`` into '
                  'transformation functions like "jit", "grad", "for_loop", etc. '
                  'Because these transformations are capable of automatically collecting them.',
                  UserWarning)


def node_deprecation(child_objs=None):
  if child_objs is not None:
    warnings.warn('\n'
                  'From brainpy>=2.4.0, users no longer need to provide ``child_objs`` into '
                  'transformation functions like "jit", "grad", "for_loop", etc. '
                  'Because these transformations are capable of automatically collecting them.',
                  UserWarning)


def abstract(x):
  if callable(x):
    return x
  else:
    return jax.api_util.shaped_abstractify(x)


def evaluate_dyn_vars(
    f,
    *args,
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = (),
    use_eval_shape: bool = True,
    **kwargs
):
  # arguments
  if len(static_argnums) or len(static_argnames):
    f2, args, kwargs = _partial_fun(f, args, kwargs,
                                    static_argnums=static_argnums,
                                    static_argnames=static_argnames)
  else:
    f2, args, kwargs = f, args, kwargs
  # stack
  with VariableStack() as stack:
    if use_eval_shape:
      rets = jax.eval_shape(f2, *args, **kwargs)
    else:
      rets = f2(*args, **kwargs)
  return stack, rets


def evaluate_dyn_vars_with_cache(
    f,
    *args,
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = (),
    with_return: bool = False,
    **kwargs
):
  # TODO: better way for cache mechanism
  stack = get_stack_cache(f)
  if stack is None or with_return:
    if len(static_argnums) or len(static_argnames):
      f2, args, kwargs = _partial_fun(f, args, kwargs, static_argnums=static_argnums, static_argnames=static_argnames)
    else:
      f2, args, kwargs = f, args, kwargs

    with jax.ensure_compile_time_eval():
      with VariableStack() as stack:
        rets = jax.eval_shape(f2, *args, **kwargs)
      cache_stack(f, stack)  # cache
      del args, kwargs, f2
    if with_return:
      return stack, rets
    else:
      return stack
  return stack
