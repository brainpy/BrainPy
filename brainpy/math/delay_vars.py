# -*- coding: utf-8 -*-


from typing import Union, Callable, Tuple

import jax.numpy as jnp
from jax import vmap
from jax.lax import cond

from brainpy import math as bm
from brainpy.base.base import Base
from brainpy.tools.checking import check_float
from brainpy.tools.others import to_size

__all__ = [
  'AbstractDelay',
  'FixedLenDelay',
  'NeutralDelay',
]


class AbstractDelay(Base):
  def update(self, time, value):
    raise NotImplementedError


_FUNC_BEFORE = 'function'
_DATA_BEFORE = 'data'


class FixedLenDelay(AbstractDelay):
  """Delay variable which has a fixed delay length.

  For example, we create a delay variable which has a maximum delay length of 1 ms

  >>> import brainpy.math as bm
  >>> delay = bm.FixedLenDelay(bm.zeros(3), delay_len=1., dt=0.1)
  >>> delay(-0.5)
  [-0. -0. -0.]

  This function supports multiple dimensions of the tensor. For example,

  1. the one-dimensional delay data
  >>> delay = bm.FixedLenDelay(3, delay_len=1., dt=0.1, before_t0=lambda t: t)
  >>> delay(-0.2)
  [-0.2 -0.2 -0.2]

  2. the two-dimensional delay data
  >>> delay = bm.FixedLenDelay((3, 2), delay_len=1., dt=0.1, before_t0=lambda t: t)
  >>> delay(-0.6)
  [[-0.6 -0.6]
   [-0.6 -0.6]
   [-0.6 -0.6]]

  3. the three-dimensional delay data
  >>> delay = bm.FixedLenDelay((3, 2, 1), delay_len=1., dt=0.1, before_t0=lambda t: t)
  >>> delay(-0.6)
  [[[-0.8]
    [-0.8]]
   [[-0.8]
    [-0.8]]
   [[-0.8]
    [-0.8]]]

  Parameters
  ----------
  shape: int, sequence of int
    The delay data shape.
  t0: float, int
    The zero time.
  delay_len: float, int
    The maximum delay length.
  dt: float, int
    The time precesion.
  before_t0: callable, bm.ndarray, jnp.ndarray
    The delay data before ::math`t_0`.
    - when `before_t0` is a function, it should receive an time argument `t`
    - when `before_to` is an array, it should be a tensor with shape
      of :math:`(num_delay, ...)`, where the longest delay data is aranged in
      the first index.
  name: str
  """

  def __init__(
      self,
      shape: Union[int, Tuple[int, ...]],
      delay_len: Union[float, int],
      before_t0: Union[Callable, bm.ndarray, jnp.ndarray] = None,
      t0: Union[float, int] = 0.,
      dt: Union[float, int] = None,
      name: str = None,
      dtype=None,
  ):
    super(FixedLenDelay, self).__init__(name=name)

    # shape
    self.shape = to_size(shape)
    self.dtype = dtype

    # delay_len
    self.t0 = t0
    self.delay_len = delay_len
    check_float(delay_len, 'delay_len', allow_none=False, allow_int=True, min_bound=0.)
    self._dt = bm.get_dt() if dt is None else dt
    self.num_delay_steps = int(bm.ceil(delay_len / self._dt).value)

    # other variables
    self._idx = bm.Variable(bm.asarray([0]))
    check_float(t0, 't0', allow_none=False, allow_int=True,)
    self._current_time = bm.Variable(bm.asarray([t0]))

    # delay data
    self._data = bm.Variable(bm.zeros((self.num_delay_steps,) + self.shape, dtype=dtype))
    if before_t0 is None:
      self._before_type = _DATA_BEFORE
    elif callable(before_t0):
      self._before_t0 = lambda t: jnp.asarray(bm.broadcast_to(before_t0(t), self.shape).value,
                                              dtype=self.dtype)
      self._before_type = _FUNC_BEFORE
    elif isinstance(before_t0, (bm.ndarray, jnp.ndarray)):
      self._before_type = _DATA_BEFORE
      if before_t0.shape != ((self.num_delay_steps,) + self.shape):
        raise ValueError(f'"before_t0" should be a tensor with the shape of '
                         f'{((self.num_delay_steps,) + self.shape)}, while '
                         f'we got {before_t0.shape}')
      self._data[:] = before_t0
    else:
      raise ValueError(f'"before_t0" does not support {type(before_t0)}: before_t0')

  @property
  def idx(self):
    return self._idx

  @idx.setter
  def idx(self, value):
    raise ValueError('Cannot set "idx" by users.')

  @property
  def dt(self):
    return self._dt

  @dt.setter
  def dt(self, value):
    raise ValueError('Cannot set "dt" by users.')

  @property
  def data(self):
    return self._data

  @property
  def current_time(self):
    return self._current_time[0]

  def __call__(self, prev_time):
    # ## Cannot check ##
    # -----------------#
    # if prev_time > self.current_time:
    #   raise ValueError(f'The request time should be less than the '
    #                    f'current time {self.current_time}. But we '
    #                    f'got {prev_time} > {self.current_time}')
    # if prev_time < (self.current_time - self.delay_len):
    #   raise ValueError(f'The request time of the variable should be in '
    #                    f'[{self.current_time - self.delay_len}, '
    #                    f'{self.current_time}], but we got {prev_time}')
    if self._before_type == _FUNC_BEFORE:
      return cond(prev_time < self.t0,
                  self._before_t0,
                  self._fn1,
                  prev_time)
    else:
      return self._fn1(prev_time)

  def _fn1(self, prev_time):
    diff = self.delay_len - (self.current_time - prev_time)
    if isinstance(diff, bm.ndarray): diff = diff.value
    req_num_step = jnp.asarray(diff / self._dt, dtype=bm.int_)
    extra = diff - req_num_step * self._dt
    return cond(extra == 0., self._true_fn, self._false_fn, (req_num_step, extra))

  def _true_fn(self, div_mod):
    req_num_step, extra = div_mod
    return self._data[self.idx[0] + req_num_step]

  def _false_fn(self, div_mod):
    req_num_step, extra = div_mod
    f = jnp.interp
    for dim in range(1, len(self.shape) + 1, 1):
      f = vmap(f, in_axes=(None, None, dim), out_axes=dim - 1)
    idx = jnp.asarray([self.idx[0] + req_num_step,
                       self.idx[0] + req_num_step + 1])
    idx %= self.num_delay_steps
    return f(extra, jnp.asarray([0., self._dt]), self._data[idx])

  def update(self, time, value):
    self._data[self._idx[0]] = value
    # ## Cannot check due to XLA's error
    # check_float(time, 'time', allow_none=False, allow_int=True)
    self._current_time[0] = time
    self._idx.value = (self._idx + 1) % self.num_delay_steps

  def __add__(self, other):
    return self.data[self.idx[0]] + other


class VariedLenDelay(AbstractDelay):
  """Delay variable which has a functional delay

  """

  def update(self, time, value):
    pass

  def __init__(self):
    super(VariedLenDelay, self).__init__()


class NeutralDelay(FixedLenDelay):
  pass
