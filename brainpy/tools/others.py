# -*- coding: utf-8 -*-

import _thread as thread
import threading

__all__ = [
  'size2num',
  'timeout',
]


def size2num(size):
  if isinstance(size, int):
    return size
  elif isinstance(size, (tuple, list)):
    a = 1
    for b in size:
      a *= b
    return a
  else:
    raise ValueError

import sys

def timeout(s):
  """Add a timeout parameter to a function and return it.

  Parameters
  ----------
  s : float
      Time limit in seconds.

  Returns
  -------
  func : callable
      Functional results. Or, raise an error of KeyboardInterrupt.
  """

  def outer(fn):
    def inner(*args, **kwargs):
      timer = threading.Timer(s, thread.interrupt_main)
      timer.start()
      try:
        result = fn(*args, **kwargs)
      finally:
        timer.cancel()
      return result
    return inner
  return outer

