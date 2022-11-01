# -*- coding: utf-8 -*-

import collections.abc
import _thread as thread
import threading
from typing import Optional, Tuple, Callable, Union, Sequence, TypeVar

import numpy as np
from jax import lax
from jax.experimental import host_callback
from tqdm.auto import tqdm

__all__ = [
  'replicate',
  'not_customized',
  'to_size',
  'size2num',
  'timeout',
  'init_progress_bar',
]


T = TypeVar('T')


def replicate(
    element: Union[T, Sequence[T]],
    num_replicate: int,
    name: str,
) -> Tuple[T, ...]:
  """Replicates entry in `element` `num_replicate` if needed."""
  if isinstance(element, (str, bytes)) or not isinstance(element, collections.abc.Sequence):
    return (element,) * num_replicate
  elif len(element) == 1:
    return tuple(element * num_replicate)
  elif len(element) == num_replicate:
    return tuple(element)
  else:
    raise TypeError(f"{name} must be a scalar or sequence of length 1 or "
                    f"sequence of length {num_replicate}.")


def not_customized(fun: Callable) -> Callable:
  """Marks the given module method is not implemented.

  Methods wrapped in @not_customized can define submodules directly within the method.

  For instance::

    @not_customized
    init_fb(self):
      ...

    @not_customized
    def feedback(self):
      ...
  """
  fun.not_customized = True
  return fun


def size2num(size):
  if isinstance(size, (int, np.integer)):
    return size
  elif isinstance(size, (tuple, list)):
    a = 1
    for b in size:
      a *= b
    return a
  else:
    raise ValueError(f'Do not support type {type(size)}: {size}')


def to_size(x) -> Optional[Tuple[int]]:
  if isinstance(x, (tuple, list)):
    return tuple(x)
  if isinstance(x, (int, np.integer)):
    return (x, )
  if x is None:
    return x
  raise ValueError(f'Cannot make a size for {x}')


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


def init_progress_bar(duration, dt, report=0.01, message=None):
  """Setup a progress bar."""
  if message is None:
    message = f"Running a duration of {duration}"

  num_samples = int(duration / dt)
  print_rate = int(duration * report / dt)
  remainder = num_samples % print_rate

  tqdm_bars = {}

  def _define_tqdm(arg, transform):
    tqdm_bars[0] = tqdm(np.arange(0, duration, dt))
    tqdm_bars[0].set_description(message, refresh=False)

  def _update_tqdm(num_processed, transform):
    tqdm_bars[0].update(num_processed * dt)

  def _update_progress_bar(num_iter):
    _ = lax.cond(
      num_iter == 0,
      lambda _: host_callback.id_tap(_define_tqdm, None, result=num_iter),
      lambda _: num_iter,
      operand=None,
    )

    _ = lax.cond(
      # update tqdm every multiple of `print_rate` except at the end
      (num_iter % print_rate == 0) & (num_iter != num_samples - remainder),
      lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=num_iter),
      lambda _: num_iter,
      operand=None,
    )

    _ = lax.cond(
      # update tqdm by `remainder`
      num_iter == num_samples - remainder,
      lambda _: host_callback.id_tap(_update_tqdm, remainder, result=num_iter),
      lambda _: num_iter,
      operand=None,
    )

  def _close_tqdm(arg, transform):
    tqdm_bars[0].close()

  def close_tqdm(iter_num):
    return lax.cond(
      iter_num == num_samples - 1,
      lambda _: host_callback.id_tap(_close_tqdm, None, result=None),
      lambda _: None,
      operand=None,
    )

  def _progress_bar(iter_num):
    _update_progress_bar(iter_num)
    close_tqdm(iter_num)

  return _progress_bar
