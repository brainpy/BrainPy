# -*- coding: utf-8 -*-

from typing import Callable


__all__ = [
  'wraps'
]


def wraps(fun: Callable):
  """Specialized version of functools.wraps for wrapping numpy functions.

  This produces a wrapped function with a modified docstring. In particular, if
  `update_doc` is True, parameters listed in the wrapped function that are not
  supported by the decorated function will be removed from the docstring. For
  this reason, it is important that parameter names match those in the original
  numpy function.
  """
  def wrap(op):
    docstr = getattr(fun, "__doc__", None)
    op.__doc__ = docstr
    op.__wrapped__ = fun
    for attr in ['__name__', '__qualname__']:
      try:
        value = getattr(fun, attr)
      except AttributeError:
        pass
      else:
        setattr(op, attr, value)
    return op
  return wrap
