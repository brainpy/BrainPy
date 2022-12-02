# -*- coding: utf-8 -*-

from typing import Union, Callable, Tuple

import jax.numpy as jnp
from jax import vmap
from jax.lax import cond, stop_gradient

from brainpy import check
from brainpy.base.base import Base
from brainpy.errors import UnsupportedError
from brainpy.math import numpy_ops as bm
from brainpy.math.jaxarray import ndarray, Variable, JaxArray
from brainpy.math.setting import get_dt
from brainpy.tools.checking import check_float, check_integer
from brainpy.tools.errors import check_error_in_jit

__all__ = [
  'AbstractDelay',
  'TimeDelay', 'LengthDelay',
  'NeuTimeDelay', 'NeuLenDelay',
]


class AbstractDelay(Base):
  def update(self, *args, **kwargs):
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

  >>> delay = bm.TimeDelay(bm.zeros(3), delay_len=1., dt=0.1, before_t0=lambda t: t)
  >>> delay(-0.2)
  [-0.2 -0.2 -0.2]

  2. the two-dimensional delay data

  >>> delay = bm.TimeDelay(bm.zeros((3, 2)), delay_len=1., dt=0.1, before_t0=lambda t: t)
  >>> delay(-0.6)
  [[-0.6 -0.6]
   [-0.6 -0.6]
   [-0.6 -0.6]]

  3. the three-dimensional delay data

  >>> delay = bm.TimeDelay(bm.zeros((3, 2, 1)), delay_len=1., dt=0.1, before_t0=lambda t: t)
  >>> delay(-0.8)
  [[[-0.8]
    [-0.8]]
   [[-0.8]
    [-0.8]]
   [[-0.8]
    [-0.8]]]

  Parameters
  ----------
  delay_target: JaxArray, ndarray, Variable
    The initial delay data.
  t0: float, int
    The zero time.
  delay_len: float, int
    The maximum delay length.
  dt: float, int
    The time precesion.
  before_t0: callable, bm.ndarray, jnp.ndarray, float, int
    The delay data before ::math`t_0`.
    - when `before_t0` is a function, it should receive a time argument `t`
    - when `before_to` is a tensor, it should be a tensor with shape
      of :math:`(num\_delay, ...)`, where the longest delay data is aranged in
      the first index.
  name: str
    The delay instance name.
  interp_method: str
    The way to deal with the delay at the time which is not integer times of the time step.
    For exameple, if the time step ``dt=0.1``, the time delay length ``delay\_len=1.``,
    when users require the delay data at ``t-0.53``, we can deal this situation with
    the following methods:

    - ``"linear_interp"``: using linear interpolation to get the delay value
      at the required time (default).
    - ``"round"``: round the time to make it is the integer times of the time step. For
      the above situation, we will use the time at ``t-0.5`` to approximate the delay data
      at ``t-0.53``.

    .. versionadded:: 2.1.1

  See Also
  --------
  LengthDelay
  """

  def __init__(
      self,
      delay_target: Union[ndarray, jnp.ndarray],
      delay_len: Union[float, int],
      before_t0: Union[Callable, ndarray, jnp.ndarray, float, int] = None,
      t0: Union[float, int] = 0.,
      dt: Union[float, int] = None,
      name: str = None,
      interp_method: str = 'linear_interp',
  ):
    super(TimeDelay, self).__init__(name=name)

    # shape
    if not isinstance(delay_target, (jnp.ndarray, JaxArray)):
      raise ValueError(f'Must be an instance of JaxArray or jax.numpy.ndarray. But we got {type(delay_target)}')

    # delay_len
    self.t0 = t0
    self.dt = get_dt() if dt is None else dt
    check_float(delay_len, 'delay_len', allow_none=False, allow_int=True, min_bound=0.)
    self.delay_len = delay_len
    self.num_delay_step = int(jnp.ceil(self.delay_len / self.dt)) + 1

    # interp method
    if interp_method not in [_INTERP_LINEAR, _INTERP_ROUND]:
      raise UnsupportedError(f'Un-supported interpolation method {interp_method}, '
                             f'we only support: {[_INTERP_LINEAR, _INTERP_ROUND]}')
    self.interp_method = interp_method

    # time variables
    self.idx = Variable(jnp.asarray([0]))
    check_float(t0, 't0', allow_none=False, allow_int=True, )
    self.current_time = Variable(jnp.asarray([t0]))

    # delay data
    batch_axis = None
    if hasattr(delay_target, 'batch_axis') and (delay_target.batch_axis is not None):
      batch_axis = delay_target.batch_axis + 1
    self.data = Variable(jnp.zeros((self.num_delay_step,) + delay_target.shape, dtype=delay_target.dtype),
                         batch_axis=batch_axis)
    if before_t0 is None:
      self._before_type = _DATA_BEFORE
    elif callable(before_t0):
      self._before_t0 = lambda t: bm.asarray(bm.broadcast_to(before_t0(t), delay_target.shape),
                                             dtype=delay_target.dtype).value
      self._before_type = _FUNC_BEFORE
    elif isinstance(before_t0, (ndarray, jnp.ndarray, float, int)):
      self._before_type = _DATA_BEFORE
      self.data[:-1] = before_t0
    else:
      raise ValueError(f'"before_t0" does not support {type(before_t0)}')
    # set initial data
    self.data[-1] = delay_target

    # interpolation function
    self._interp_fun = jnp.interp
    for dim in range(1, delay_target.ndim + 1, 1):
      self._interp_fun = vmap(self._interp_fun, in_axes=(None, None, dim), out_axes=dim - 1)

  def reset(self,
            delay_target,
            delay_len,
            t0: Union[float, int] = 0.,
            before_t0=None):
    """Reset the delay variable.

    Parameters
    ----------
    delay_target: JaxArray, ndarray, Variable
      The delay target.
    delay_len: float, int
      The maximum delay length. The unit is the time.
    t0: int, float
      The zero time.
    before_t0: int, float, ndarray, JaxArray
      The data before t0.
    """
    self.delay_len = delay_len
    self.num_delay_step = int(jnp.ceil(self.delay_len / self.dt)) + 1
    self.data.value = jnp.zeros((self.num_delay_step,) + delay_target.shape, dtype=delay_target.dtype)
    self.data[-1] = delay_target
    self.idx = Variable(jnp.asarray([0]))
    self.current_time = Variable(jnp.asarray([t0]))
    if before_t0 is not None:
      if not isinstance(before_t0, (ndarray, jnp.ndarray, float, int)):
        raise ValueError('Only support numerical values.')
      self.data[:-1] = before_t0
      self._before_type = _DATA_BEFORE

  def _check_time1(self, times):
    prev_time, current_time = times
    raise ValueError(f'The request time should be less than the '
                     f'current time {current_time}. But we '
                     f'got {prev_time} > {current_time}')

  def _check_time2(self, times):
    prev_time, current_time = times
    raise ValueError(f'The request time of the variable should be in '
                     f'[{current_time - self.delay_len}, {current_time}], '
                     f'but we got {prev_time}')

  def __call__(self, time, indices=None):
    # check
    if check.is_checking():
      current_time = self.current_time[0]
      check_error_in_jit(time > current_time + 1e-6, self._check_time1, (time, current_time))
      check_error_in_jit(time < current_time - self.delay_len - self.dt, self._check_time2, (time, current_time))
    if self._before_type == _FUNC_BEFORE:
      res = cond(time < self.t0,
                 self._before_t0,
                 self._after_t0,
                 time)
    else:
      res = self._after_t0(time)
    if indices is not None:  # TODO: indices is highly inefficient
      res = res[indices]
    return res

  def _after_t0(self, prev_time):
    diff = self.delay_len - (self.current_time[0] - prev_time)
    if isinstance(diff, ndarray):
      diff = diff.value
    if self.interp_method == _INTERP_LINEAR:
      req_num_step = jnp.asarray(diff / self.dt, dtype=jnp.int32)
      extra = diff - req_num_step * self.dt
      return cond(extra == 0., self._true_fn, self._false_fn, (req_num_step, extra))
    elif self.interp_method == _INTERP_ROUND:
      req_num_step = jnp.asarray(jnp.round(diff / self.dt), dtype=jnp.int32)
      return self._true_fn([req_num_step, 0.])
    else:
      raise UnsupportedError(f'Un-supported interpolation method {self.interp_method}, '
                             f'we only support: {[_INTERP_LINEAR, _INTERP_ROUND]}')

  def _true_fn(self, div_mod):
    req_num_step, extra = div_mod
    return self.data[self.idx[0] + req_num_step]

  def _false_fn(self, div_mod):
    req_num_step, extra = div_mod
    idx = jnp.asarray([self.idx[0] + req_num_step,
                       self.idx[0] + req_num_step + 1])
    idx %= self.num_delay_step
    return self._interp_fun(extra, jnp.asarray([0., self.dt]), self.data[idx])

  def update(self, time, value):
    self.data[self.idx[0]] = value
    self.current_time[0] = time
    self.idx.value = (self.idx + 1) % self.num_delay_step


class NeuTimeDelay(TimeDelay):
  """Neutral Time Delay. Alias of :py:class:`~.TimeDelay`."""
  pass


ROTATION_UPDATING = 'rotation'
CONCAT_UPDATING = 'concatenate'


class LengthDelay(AbstractDelay):
  """Delay variable which has a fixed delay length.

  Parameters
  ----------
  delay_target: int, sequence of int
    The initial delay data.
  delay_len: int
    The maximum delay length.
  initial_delay_data: Any
    The delay data. It can be a Python number, like float, int, boolean values.
    It can also be arrays. Or a callable function or instance of ``Connector``.
    Note that ``initial_delay_data`` should be arranged as the following way::

       delay = 1             [ data
       delay = 2               data
       ...                     ....
       ...                     ....
       delay = delay_len-1     data
       delay = delay_len       data ]

    .. versionchanged:: 2.2.3.2

       The data in the previous version of ``LengthDelay`` is::

         delay = delay_len     [ data
         delay = delay_len-1     data
         ...                     ....
         ...                     ....
         delay = 2               data
         delay = 1               data ]


  name: str
    The delay object name.
  batch_axis: int
    The batch axis. If not provided, it will be inferred from the `delay_target`.
  update_method: str
    The method used for updating delay.

  See Also
  --------
  TimeDelay
  """

  def __init__(
      self,
      delay_target: Union[ndarray, jnp.ndarray],
      delay_len: int,
      initial_delay_data: Union[float, int, bool, ndarray, jnp.ndarray, Callable] = None,
      name: str = None,
      batch_axis: int = None,
      update_method: str = ROTATION_UPDATING
  ):
    super(LengthDelay, self).__init__(name=name)

    assert update_method in [ROTATION_UPDATING, CONCAT_UPDATING]
    self.update_method = update_method
    # attributes and variables
    self.data: Variable = None
    self.num_delay_step: int = None
    self.idx: Variable = None

    # initialization
    self.reset(delay_target, delay_len, initial_delay_data, batch_axis)

  @property
  def delay_shape(self):
    """The data shape of this delay variable."""
    return self.data.shape

  @property
  def delay_target_shape(self):
    """The data shape of the delay target."""
    return self.data.shape[1:]

  def __repr__(self):
    name = self.__class__.__name__
    return (f'{name}(num_delay_step={self.num_delay_step}, '
            f'delay_target_shape={self.delay_target_shape}, '
            f'update_method={self.update_method})')

  def reset(
      self,
      delay_target,
      delay_len=None,
      initial_delay_data=None,
      batch_axis=None
  ):
    if not isinstance(delay_target, (ndarray, jnp.ndarray)):
      raise ValueError(f'Must be an instance of brainpy.math.ndarray '
                       f'or jax.numpy.ndarray. But we got {type(delay_target)}')

    # delay_len
    check_integer(delay_len, 'delay_len', allow_none=True, min_bound=0)
    if delay_len is None:
      if self.num_delay_step is None:
        raise ValueError('"delay_len" cannot be None.')
      delay_len = self.num_delay_step - 1
    self.num_delay_step = delay_len + 1

    # initialize delay data
    if self.data is None:
      if batch_axis is None:
        if isinstance(delay_target, Variable) and (delay_target.batch_axis is not None):
          batch_axis = delay_target.batch_axis + 1
      self.data = Variable(jnp.zeros((self.num_delay_step,) + delay_target.shape,
                                     dtype=delay_target.dtype),
                           batch_axis=batch_axis)
    else:
      self.data._value = jnp.zeros((self.num_delay_step,) + delay_target.shape,
                                   dtype=delay_target.dtype)

    # update delay data
    self.data[0] = delay_target
    if initial_delay_data is None:
      pass
    elif isinstance(initial_delay_data, (ndarray, jnp.ndarray, float, int, bool)):
      self.data[1:] = initial_delay_data
    elif callable(initial_delay_data):
      self.data[1:] = initial_delay_data((delay_len,) + delay_target.shape,
                                          dtype=delay_target.dtype)
    else:
      raise ValueError(f'"delay_data" does not support {type(initial_delay_data)}')

    # time variables
    if self.update_method == ROTATION_UPDATING:
      if self.idx is None:
        self.idx = Variable(stop_gradient(jnp.asarray([0], dtype=jnp.int32)))
      else:
        self.idx.value = stop_gradient(jnp.asarray([0], dtype=jnp.int32))

  def _check_delay(self, delay_len):
    raise ValueError(f'The request delay length should be less than the '
                     f'maximum delay {self.num_delay_step}. '
                     f'But we got {delay_len}')

  def __call__(self, delay_len, *indices):
    return self.retrieve(delay_len, *indices)

  def retrieve(self, delay_len, *indices):
    """Retrieve the delay data acoording to the delay length.

    Parameters
    ----------
    delay_len: int, Array
      The delay length used to retrieve the data.
    """
    if check.is_checking():
      check_error_in_jit(bm.any(delay_len >= self.num_delay_step), self._check_delay, delay_len)

    if self.update_method == ROTATION_UPDATING:
      delay_idx = (self.idx[0] + delay_len) % self.num_delay_step
      delay_idx = stop_gradient(delay_idx)

    elif self.update_method == CONCAT_UPDATING:
      delay_idx = delay_len

    else:
      raise ValueError(f'Unknown updating method "{self.update_method}"')

    # the delay index
    if isinstance(delay_idx, int):
      pass
    elif hasattr(delay_idx, 'dtype') and not jnp.issubdtype(delay_idx.dtype, jnp.integer):
      raise ValueError(f'"delay_len" must be integer, but we got {delay_idx}')
    indices = (delay_idx,) + tuple(indices)
    # the delay data
    return self.data[indices]

  def update(self, value: Union[float, int, bool, JaxArray, jnp.DeviceArray]):
    """Update delay variable with the new data.

    Parameters
    ----------
    value: Any
      The value of the latest data, used to update this delay variable.
    """
    if self.update_method == ROTATION_UPDATING:
      self.idx.value = stop_gradient((self.idx - 1) % self.num_delay_step)
      self.data[self.idx[0]] = value

    elif self.update_method == CONCAT_UPDATING:
      if self.num_delay_step >= 2:
        self.data.value = bm.vstack([bm.broadcast_to(value, self.data.shape[1:]), self.data[1:]])
      else:
        self.data[:] = value

    else:
      raise ValueError(f'Unknown updating method "{self.update_method}"')


class NeuLenDelay(LengthDelay):
  """Neutral Length Delay. Alias of :py:class:`~.LengthDelay`."""
  pass
