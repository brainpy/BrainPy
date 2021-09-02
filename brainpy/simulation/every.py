# -*- coding: utf-8 -*-


__all__ = [
  'every',
]


def _every_decorator(f, time):
  """'every' decorator for step functions.

  Parameters
  ----------
  f : function
      The step function.
  time : int, float, function
      The time.
  """
  f.interval_time_to_run = time
  return f


def every(time, f=None):
  """The decorator to add the interval time for step function running.

  >>> # interval time can be a int/float
  >>> @every(time=5)
  >>> def step():
  >>>   pass
  >>>
  >>> # interval time can also be a bool function
  >>> import numpy as np
  >>> @every(time=lambda: np.random.random() < 0.5)
  >>> def step():
  >>>   pass

  Parameters
  ----------
  time : int, float, function
      It can be int/float/bool_function to denote the `every` time
      to run the step function.
  f : function
      The step function.

  Returns
  -------
  decorated_func : function
      The decorated function.
  """
  if f is None:
    return lambda func: _every_decorator(f=func, time=time)
  else:
    return _every_decorator(f=f, time=time)
