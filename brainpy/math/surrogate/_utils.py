# -*- coding: utf-8 -*-


import jax
from functools import wraps, partial
from brainpy.math.ndarray import Array

__all__ = [
  'get_default',
  'make_return',
  'generalized_vjp_custom',
]


def get_default(x, default):
  if x is None:
    return default, False
  else:
    return x, True


def make_return(r, *args):
  if isinstance(r, (tuple, list)):
    r = tuple(r)
  else:
    r = [r]
  for a in args:
    if a:
      r += [None]
  return tuple(r)


def generalized_vjp_custom(*arg_names, **defaults):
  """Generalize a customized gradient function as a general Python function.
  """
  n_args = len(arg_names)
  default_keys = tuple(defaults.keys())
  defaults = tuple(defaults.items())

  def wrapper(fun):

    @wraps(fun)
    def call(*args, **kwargs):
      args = list(args)
      kwargs = dict(kwargs)
      for k in arg_names[len(args):]:
        if k not in kwargs:
          raise ValueError(f'Must provide {k} for function {fun}')
        args.append(kwargs.pop(k))
      for k, v in defaults[len(args) - n_args:]:
        if k not in kwargs:
          args.append(v)
        else:
          args.append(kwargs.pop(k))
      if len(kwargs):
        raise KeyError(f'Unknown arguments {kwargs} for function {fun}')
      args = [a.value if isinstance(a, Array) else a for a in args]
      kwargs = dict(zip(default_keys, args[n_args:]))
      grad_fun = jax.custom_gradient(partial(fun, **kwargs))
      return grad_fun(*args[:n_args])

    call.__doc__ = fun.__doc__

    return call

  return wrapper


