# -*- coding: utf-8 -*-

import warnings
from typing import Union, Callable, Tuple

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.experimental.host_callback import id_tap
from jax.lax import cond

from brainpy import check
from brainpy import math as bm
from brainpy.base.base import Base
from brainpy.errors import UnsupportedError
from brainpy.tools.checking import check_float
from brainpy.tools.others import to_size

__all__ = [
  'AbstractDelay',
  'TimeDelay',
  'FixedLenDelay',
  'NeutralDelay',
]


class AbstractDelay(Base):
  def update(self, time, value):
    raise NotImplementedError


_FUNC_BEFORE = 'function'
_DATA_BEFORE = 'data'
_INTERP_LINEAR = 'linear_interp'
_INTERP_ROUND = 'round'


class TimeDelay(AbstractDelay):
  """Delay variable which has a fixed delay time length.

  For example, we create a delay variable which has a maximum delay length of 1 ms

  >>> import brainpy.math as bm
  >>> delay = bm.TimeDelay(bm.zeros(3), delay_len=1., dt=0.1)
  >>> delay(-0.5)
  [-0. -0. -0.]

  This function supports multiple dimensions of the tensor. For example,

  1. the one-dimensional delay data

  >>> delay = bm.TimeDelay(3, delay_len=1., dt=0.1, before_t0=lambda t: t)
  >>> delay(-0.2)
  [-0.2 -0.2 -0.2]

  2. the two-dimensional delay data

  >>> delay = bm.TimeDelay((3, 2), delay_len=1., dt=0.1, before_t0=lambda t: t)
  >>> delay(-0.6)
  [[-0.6 -0.6]
   [-0.6 -0.6]
   [-0.6 -0.6]]

  3. the three-dimensional delay data

  >>> delay = bm.TimeDelay((3, 2, 1), delay_len=1., dt=0.1, before_t0=lambda t: t)
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
  before_t0: callable, bm.ndarray, jnp.ndarray, float, int
    The delay data before ::math`t_0`.
    - when `before_t0` is a function, it should receive an time argument `t`
    - when `before_to` is a tensor, it should be a tensor with shape
      of :math:`(num_delay, ...)`, where the longest delay data is aranged in
      the first index.
  name: str
    The delay instance name.
  interp_method: str
    The way to deal with the delay at the time which is not integer times of the time step.
    For exameple, if the time step ``dt=0.1``, the time delay length ``delay_len=1.``,
    when users require the delay data at ``t-0.53``, we can deal this situation with
    the following methods:

    - ``"linear_interp"``: using linear interpolation to get the delay value
      at the required time (default).
    - ``"round"``: round the time to make it is the integer times of the time step. For
      the above situation, we will use the time at ``t-0.5`` to approximate the delay data
      at ``t-0.53``.

    .. versionadded:: 2.1.1
  """

  def __init__(
      self,
      shape: Union[int, Tuple[int, ...]],
      delay_len: Union[float, int],
      before_t0: Union[Callable, bm.ndarray, jnp.ndarray, float, int] = None,
      t0: Union[float, int] = 0.,
      dt: Union[float, int] = None,
      name: str = None,
      dtype=None,
      interp_method='linear_interp',
  ):
    super(TimeDelay, self).__init__(name=name)

    # shape
    self.shape = to_size(shape)
    self.dtype = dtype

    # delay_len
    self.t0 = t0
    self._dt = bm.get_dt() if dt is None else dt
    check_float(delay_len, 'delay_len', allow_none=False, allow_int=True, min_bound=0.)
    self._delay_len = delay_len
    self.delay_len = delay_len + self._dt
    self.num_delay_step = int(bm.ceil(self.delay_len / self._dt).value)

    # interp method
    if interp_method not in [_INTERP_LINEAR, _INTERP_ROUND]:
      raise UnsupportedError(f'Un-supported interpolation method {interp_method}, '
                             f'we only support: {[_INTERP_LINEAR, _INTERP_ROUND]}')
    self.interp_method = interp_method

    # time variables
    self._idx = bm.Variable(bm.asarray([0]))
    check_float(t0, 't0', allow_none=False, allow_int=True, )
    self._current_time = bm.Variable(bm.asarray([t0]))

    # delay data
    self._data = bm.Variable(bm.zeros((self.num_delay_step,) + self.shape, dtype=dtype))
    if before_t0 is None:
      self._before_type = _DATA_BEFORE
    elif callable(before_t0):
      self._before_t0 = lambda t: jnp.asarray(bm.broadcast_to(before_t0(t), self.shape).value,
                                              dtype=self.dtype)
      self._before_type = _FUNC_BEFORE
    elif isinstance(before_t0, (bm.ndarray, jnp.ndarray, float, int)):
      self._before_type = _DATA_BEFORE
      try:
        self._data[:] = before_t0
      except:
        raise ValueError(f'Cannot set delay data by using "before_t0". '
                         f'The delay data has the shape of '
                         f'{((self.num_delay_step,) + self.shape)}, while '
                         f'we got "before_t0" of {bm.asarray(before_t0).shape}. '
                         f'They are not compatible. Note that the delay length '
                         f'{self._delay_len} will automatically add a dt {self.dt} '
                         f'to {self.delay_len}.')
    else:
      raise ValueError(f'"before_t0" does not support {type(before_t0)}: before_t0')

    self.f = jnp.interp
    for dim in range(1, len(self.shape) + 1, 1):
      self.f = vmap(self.f, in_axes=(None, None, dim), out_axes=dim - 1)

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

  @data.setter
  def data(self, value):
    self._data[:] = value

  @property
  def current_time(self):
    return self._current_time[0]

  def _check_time(self, times, transforms):
    prev_time, current_time = times
    current_time = np.asarray(current_time, dtype=bm.float_)
    prev_time = np.asarray(prev_time, dtype=bm.float_)
    if prev_time > current_time:
      raise ValueError(f'\n'
                       f'!!! Error in {self.__class__.__name__}: \n'
                       f'The request time should be less than the '
                       f'current time {current_time}. But we '
                       f'got {prev_time} > {current_time}')
    lower_time = np.asarray(current_time - self.delay_len)
    if prev_time < lower_time:
      raise ValueError(f'\n'
                       f'!!! Error in {self.__class__.__name__}: \n'
                       f'The request time of the variable should be in '
                       f'[{lower_time}, {current_time}], but we got {prev_time}')

  def __call__(self, time, indices=None):
    # check
    if check.is_checking():
      id_tap(self._check_time, (time, self.current_time))
    if self._before_type == _FUNC_BEFORE:
      return cond(time < self.t0,
                  self._before_t0,
                  self._after_t0,
                  time)
    else:
      return self._after_t0(time)

  def _after_t0(self, prev_time):
    diff = self.delay_len - (self.current_time - prev_time)
    if isinstance(diff, bm.ndarray):
      diff = diff.value
    if self.interp_method == _INTERP_LINEAR:
      req_num_step = jnp.asarray(diff / self._dt, dtype=bm.get_dint())
      extra = diff - req_num_step * self._dt
      return cond(extra == 0., self._true_fn, self._false_fn, (req_num_step, extra))
    elif self.interp_method == _INTERP_ROUND:
      req_num_step = jnp.asarray(jnp.round(diff / self._dt), dtype=bm.get_dint())
      return self._true_fn([req_num_step, 0.])
    else:
      raise UnsupportedError(f'Un-supported interpolation method {self.interp_method}, '
                             f'we only support: {[_INTERP_LINEAR, _INTERP_ROUND]}')

  def _true_fn(self, div_mod):
    req_num_step, extra = div_mod
    return self._data[self.idx[0] + req_num_step]

  def _false_fn(self, div_mod):
    req_num_step, extra = div_mod
    idx = jnp.asarray([self.idx[0] + req_num_step,
                       self.idx[0] + req_num_step + 1])
    idx %= self.num_delay_step
    return self.f(extra, jnp.asarray([0., self._dt]), self._data[idx])

  def update(self, time, value):
    self._data[self._idx[0]] = value
    self._current_time[0] = time
    self._idx.value = (self._idx + 1) % self.num_delay_step


def FixedLenDelay(shape: Union[int, Tuple[int, ...]],
                  delay_len: Union[float, int],
                  before_t0: Union[Callable, bm.ndarray, jnp.ndarray, float, int] = None,
                  t0: Union[float, int] = 0.,
                  dt: Union[float, int] = None,
                  name: str = None,
                  dtype=None,
                  interp_method='linear_interp', ):
  warnings.warn('Please use "brainpy.math.TimeDelay" instead. '
                '"brainpy.math.FixedLenDelay" is deprecated since version 2.1.2. ',
                DeprecationWarning)
  return TimeDelay(shape=shape,
                   delay_len=delay_len,
                   before_t0=before_t0,
                   t0=t0,
                   dt=dt,
                   name=name,
                   dtype=dtype,
                   interp_method=interp_method)


class NeutralDelay(TimeDelay):
  pass


class LengthDelay(AbstractDelay):
  pass


