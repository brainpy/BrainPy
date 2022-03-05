# -*- coding: utf-8 -*-


from typing import Union, Callable

import jax.numpy as jnp
from jax import vmap
from jax.lax import cond

from brainpy import math as bm
from brainpy.base.base import Base
from brainpy.tools.checking import check_float

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
  >>> delay = bm.FixedLenDelay(bm.zeros(3), delay_len=1., dt=0.1, before_t0=lambda t: t)
  >>> delay(-0.2)
  [-0.2 -0.2 -0.2]

  2. the two-dimensional delay data
  >>> delay = bm.FixedLenDelay(bm.zeros([3, 2]), delay_len=1., dt=0.1, before_t0=lambda t: t)
  >>> delay(-0.6)
  [[-0.6 -0.6]
   [-0.6 -0.6]
   [-0.6 -0.6]]

  3. the three-dimensional delay data
  >>> delay = bm.FixedLenDelay(bm.zeros([3, 2, 1]), delay_len=1., dt=0.1, before_t0=lambda t: t)
  >>> delay(-0.6)
  [[[-0.8]
    [-0.8]]
   [[-0.8]
    [-0.8]]
   [[-0.8]
    [-0.8]]]

  Parameters
  ----------
  v0: bm.ndarray, jnp.ndarray, float, int
    The current state at the zero time :math:`t_0`.
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
      v0: Union[bm.ndarray, jnp.ndarray, float, int],
      delay_len: Union[float, int],
      before_t0: Union[Callable, bm.ndarray, jnp.ndarray] = None,
      t0: Union[float, int] = 0.,
      dt: Union[float, int] = None,
      name: str = None,
  ):
    super(FixedLenDelay, self).__init__(name=name)

    v0 = bm.asarray(v0)
    # shape
    self.shape = v0.shape
    self.dtype = v0.dtype

    # delay_len
    self.t0 = t0
    self.delay_len = delay_len
    self._dt = bm.get_dt() if dt is None else dt
    self.num_delay_steps = int(bm.ceil(delay_len / self._dt).value)

    # other variables
    self._id_in = bm.Variable(bm.asarray([self.num_delay_steps]))
    self._id_out = bm.Variable(bm.asarray([0]))
    check_float(t0, 't0', allow_none=False)
    self._current_time = bm.Variable(bm.asarray([t0]))

    # delay data
    self._data = bm.Variable(bm.zeros((self.num_delay_steps + 1,) + self.shape, dtype=v0.dtype))
    self._data[self._id_in[0]] = v0
    if before_t0 is None:
      self._before_type = _DATA_BEFORE
    elif callable(before_t0):
      self._before_t0 = lambda t: jnp.asarray(bm.broadcast_to(before_t0(t), self.shape).value, dtype=self.dtype)
      self._before_type = _FUNC_BEFORE
    elif isinstance(before_t0, (bm.ndarray, jnp.ndarray)):
      self._before_type = _DATA_BEFORE
      if before_t0.shape != ((self.num_delay_steps,) + self.shape):
        raise ValueError(f'"before_t0" should be a tensor with the shape of '
                         f'{((self.num_delay_steps,) + self.shape)}, while '
                         f'we got {before_t0.shape}')
      self._data[:-1] = before_t0
    else:
      raise ValueError(f'"before_t0" does not support {type(before_t0)}: before_t0')

  @property
  def idx_in(self):
    return self._id_in

  @idx_in.setter
  def idx_in(self, value):
    raise ValueError('Cannot set "idx_in" by users.')

  @property
  def idx_out(self):
    return self._id_out

  @idx_out.setter
  def idx_out(self, value):
    raise ValueError('Cannot set "idx_out" by users.')

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
    assert prev_time <= self.current_time, (f'The request time should be less than the '
                                            f'current time {self.current_time}. But we '
                                            f'got {prev_time} > {self.current_time}')
    assert prev_time >= (self.current_time - self.delay_len), (f'The request time of the variable should be in '
                                                               f'[{self.current_time - self.delay_len}, '
                                                               f'{self.current_time}], but we got {prev_time}')
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
    req_num_step, extra = jnp.divmod(diff, self._dt)
    req_num_step = req_num_step.astype(dtype=bm.int_)
    return cond(extra == 0., self._true_fn, self._false_fn, (req_num_step, extra))

  def _true_fn(self, div_mod):
    req_num_step, extra = div_mod
    return self._data[self._id_out[0] + req_num_step]

  def _false_fn(self, div_mod):
    req_num_step, extra = div_mod
    f = jnp.interp
    for dim in range(1, len(self.shape) + 1, 1):
      f = vmap(f, in_axes=(None, None, dim), out_axes=dim - 1)
    idx = jnp.asarray([self._id_out[0] + req_num_step,
                       self._id_out[0] + req_num_step + 1])
    idx %= self.num_delay_steps
    return f(extra, jnp.asarray([0., self._dt]), self._data[idx])

  def update(self, time, value):
    self._data[self._id_in[0]] = value
    check_float(time, 'time', allow_none=False, allow_int=True)
    self._current_time[0] = time
    self._id_in.value = (self._id_in + 1) % (self.num_delay_steps + 1)
    self._id_out.value = (self._id_out + 1) % (self.num_delay_steps + 1)


class VariedLenDelay(AbstractDelay):
  """Delay variable which has a functional delay

  """

  def update(self, time, value):
    pass

  def __init__(self):
    super(VariedLenDelay, self).__init__()


class NeutralDelay(FixedLenDelay):
  pass
