# -*- coding: utf-8 -*-

import functools
import inspect
import os
import re
import sys
from typing import Any, Callable, TypeVar, cast

from jax import dtypes, config, numpy as jnp, devices
from jax.lib import xla_bridge
from brainpy import check
from brainpy import tools
from . import modes

__all__ = [
  # set jax environments
  'enable_x64',
  'disable_x64',
  'set_platform',
  'get_platform',
  'set_host_device_count',

  # device memory
  'clear_buffer_memory',
  'enable_gpu_memory_preallocation',
  'disable_gpu_memory_preallocation',

  # default data types
  'bool_',
  'int_',
  'float_',
  'complex_',
  'ditype',
  'dftype',
  'set_float',
  'get_float',
  'set_int',
  'get_int',

  # default numerical integration step
  'dt',
  'set_dt',
  'get_dt',

  # default computation modes
  'mode',
  'set_mode',
  'get_mode',

  # context manage for environment setting
  'environment',

  # others
  'form_shared_args',

]

# default dtype
# --------------------------

bool_ = jnp.bool_
'''Default bool data type.'''

int_ = jnp.int32
'''Default integer data type.'''

float_ = jnp.float32
'''Default float data type.'''

complex_ = jnp.complex_
'''Default complex data type.'''


def ditype():
  """Default int type."""
  return jnp.int64 if config.read('jax_enable_x64') else jnp.int32


def dftype():
  """Default float type."""
  return jnp.float64 if config.read('jax_enable_x64') else jnp.float32


def set_float(dtype: type):
  """Set global default float type.

  Parameters
  ----------
  dtype: type
    The float type in JAX.
  """
  if dtype not in [jnp.float16, jnp.float32, jnp.float64, ]:
    raise TypeError
  global float_
  float_ = dtype


def get_float():
  """Get the default float data type.
  
  Returns
  -------
  dftype: type
    The default float data type.
  """
  return float_


def set_int(dtype: type):
  """Set global default integer type.

  Parameters
  ----------
  dtype: type
    The integer type in JAX.
  """
  if dtype not in [jnp.int8, jnp.int16, jnp.int32, jnp.int64,
                   jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64, ]:
    raise TypeError
  global int_
  int_ = dtype


def get_int():
  """Get the default int data type.

  Returns
  -------
  dftype: type
    The default int data type.
  """
  return int_


# numerical precision
# --------------------------

dt = 0.1
'''Default time step for numerical integration.'''


def set_dt(d):
  """Set the numerical integrator precision.

  Parameters
  ----------
  d : float
      Numerical integration precision.
  """
  _dt = jnp.asarray(d)
  if not dtypes.issubdtype(_dt.dtype, jnp.floating):
    raise ValueError(f'"dt" must a float, but we got {d}')
  if _dt.ndim != 0:
    raise ValueError(f'"dt" must be a scalar, but we got {d}')
  global dt
  dt = d


def get_dt():
  """Get the numerical integrator precision.

  Returns
  -------
  dt : float
      Numerical integration precision.
  """
  return dt


def enable_x64(mode=True):
  assert mode in [True, False]
  config.update("jax_enable_x64", mode)


def disable_x64():
  config.update("jax_enable_x64", False)


def set_platform(platform: str):
  """
  Changes platform to CPU, GPU, or TPU. This utility only takes
  effect at the beginning of your program.
  """
  assert platform in ['cpu', 'gpu', 'tpu']
  config.update("jax_platform_name", platform)


def get_platform() -> str:
  """Get the computing platform.

  Returns
  -------
  platform: str
    Either 'cpu', 'gpu' or 'tpu'.
  """
  return devices()[0].platform


def set_host_device_count(n):
  """
  By default, XLA considers all CPU cores as one device. This utility tells XLA
  that there are `n` host (CPU) devices available to use. As a consequence, this
  allows parallel mapping in JAX :func:`jax.pmap` to work in CPU platform.

  .. note:: This utility only takes effect at the beginning of your program.
      Under the hood, this sets the environment variable
      `XLA_FLAGS=--xla_force_host_platform_device_count=[num_devices]`, where
      `[num_device]` is the desired number of CPU devices `n`.

  .. warning:: Our understanding of the side effects of using the
      `xla_force_host_platform_device_count` flag in XLA is incomplete. If you
      observe some strange phenomenon when using this utility, please let us
      know through our issue or forum page. More information is available in this
      `JAX issue <https://github.com/google/jax/issues/1408>`_.

  :param int n: number of devices to use.
  """
  xla_flags = os.getenv("XLA_FLAGS", "")
  xla_flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "", xla_flags).split()
  os.environ["XLA_FLAGS"] = " ".join(["--xla_force_host_platform_device_count={}".format(n)] + xla_flags)


def clear_buffer_memory(platform=None):
  """Clear all on-device buffers.

  This function will be very useful when you call models in a Python loop,
  because it can clear all cached arrays, and clear device memory.

  .. warning::

     This operation may cause errors when you use a deleted buffer.
     Therefore, regenerate data always.

  Parameters
  ----------
  platform: str
    The device to clear its memory.
  """
  for buf in xla_bridge.get_backend(platform=platform).live_buffers():
    buf.delete()


def disable_gpu_memory_preallocation():
  """Disable pre-allocating the GPU memory."""
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


def enable_gpu_memory_preallocation():
  """Disable pre-allocating the GPU memory."""
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
  os.environ.pop('XLA_PYTHON_CLIENT_ALLOCATOR')


def form_shared_args(duration: float = None,
                     num_step: int = None,
                     dt: float = None,
                     t0: float = 0.):
  """Form a shared argument for the inference of a :py:class:`~.DynamicalSystem`.

  Parameters
  ----------
  duration: float
  num_step: int
  dt: float
  t0: float

  Returns
  -------
  shared: DotDict
    The shared arguments over the given time.
  """


  dt = get_dt() if dt is None else dt
  check.check_float(dt, 'dt', allow_none=False)
  if duration is None:
    check.check_integer(num_step, 'num_step', allow_none=False)
    duration = dt * num_step
  else:
    check.check_float(duration, 'duration', allow_none=False)

  r = tools.DotDict(t=jnp.arange(t0, duration + t0, dt))
  r['dt'] = jnp.ones_like(r['t']) * dt
  r['i'] = jnp.arange(r['t'].shape[0])
  return r


mode = modes.NonBatchingMode()
'''Default computation mode.'''


def set_mode(m: modes.CompMode):
  """Set the default computing mode.

  Parameters
  ----------
  m: CompMode
    The instance of :py:class:`~.CompMode`.
  """
  if not isinstance(m, modes.CompMode):
    raise TypeError(f'Must be instance of brainpy.math.CompMode. '
                    f'But we got {type(m)}: {m}')
  global mode
  mode = m


def get_mode() -> modes.CompMode:
  """Get the default computing mode.

  References
  ----------
  mode: CompMode
    The default computing mode.
  """
  return mode


# See https://mypy.readthedocs.io/en/latest/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)


class _DecoratorContextManager:
  """Allow a context manager to be used as a decorator"""

  def __call__(self, func: F) -> F:
    if inspect.isgeneratorfunction(func):
      return self._wrap_generator(func)

    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
      with self.clone():
        return func(*args, **kwargs)

    return cast(F, decorate_context)

  def _wrap_generator(self, func):
    """Wrap each generator invocation with the context manager"""

    @functools.wraps(func)
    def generator_context(*args, **kwargs):
      gen = func(*args, **kwargs)

      # Generators are suspended and unsuspended at `yield`, hence we
      # make sure the grad modes is properly set every time the execution
      # flow returns into the wrapped generator and restored when it
      # returns through our `yield` to our caller (see PR #49017).
      try:
        # Issuing `None` to a generator fires it up
        with self.clone():
          response = gen.send(None)

        while True:
          try:
            # Forward the response to our caller and get its next request
            request = yield response

          except GeneratorExit:
            # Inform the still active generator about its imminent closure
            with self.clone():
              gen.close()
            raise

          except BaseException:
            # Propagate the exception thrown at us by the caller
            with self.clone():
              response = gen.throw(*sys.exc_info())

          else:
            # Pass the last request to the generator and get its response
            with self.clone():
              response = gen.send(request)

      # We let the exceptions raised above by the generator's `.throw` or
      # `.send` methods bubble up to our caller, except for StopIteration
      except StopIteration as e:
        # The generator informed us that it is done: take whatever its
        # returned value (if any) was and indicate that we're done too
        # by returning it (see docs for python's return-statement).
        return e.value

    return generator_context

  def __enter__(self) -> None:
    raise NotImplementedError

  def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
    raise NotImplementedError

  def clone(self):
    # override this method if your children class takes __init__ parameters
    return self.__class__()


class environment(_DecoratorContextManager):
  r"""Context-manager that sets an environment for brain dynamics modeling.

  Disabling gradient calculation is useful for inference, when you are sure
  that you will not call :meth:`Tensor.backward()`. It will reduce memory
  consumption for computations that would otherwise have `requires_grad=True`.

  In this mode, the result of every computation will have
  `requires_grad=False`, even when the inputs have `requires_grad=True`.

  This context manager is thread local; it will not affect computation
  in other threads.

  Also functions as a decorator. (Make sure to instantiate with parenthesis.)

  .. note::
      No-grad is one of several mechanisms that can enable or
      disable gradients locally see :ref:`locally-disable-grad-doc` for
      more information on how they compare.

  .. note::
      This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.
      If you want to disable forward AD for a computation, you can unpack
      your dual tensors.

  Example::

      >>> x = torch.tensor([1.], requires_grad=True)
      >>> with torch.no_grad():
      ...   y = x * 2
      >>> y.requires_grad
      False
      >>> @torch.no_grad()
      ... def doubler(x):
      ...     return x * 2
      >>> z = doubler(x)
      >>> z.requires_grad
      False
  """

  def __init__(
      self,
      dt: float = None,
      mode: modes.CompMode = None,
      float_: type = None,
      int_: type = None,
  ) -> None:
    super().__init__()
    self.old_dt = get_dt()
    self.old_mode = get_mode()
    self.old_float = get_float()
    self.old_int = get_int()

    if dt is not None:
      assert isinstance(dt, float), '"dt" must a float.'
    if mode is not None:
      assert isinstance(mode, modes.CompMode), f'"mode" must a {modes.CompMode}.'
    if float_ is not None:
      assert isinstance(float_, type), '"float_" must a float.'
    if int_ is not None:
      assert isinstance(int_, type), '"int_" must a type.'
    self.dt = dt
    self.compmode = mode
    self.float_ = float_
    self.int_ = int_

  def __enter__(self) -> None:
    if self.dt is not None:
      set_dt(self.dt)
    if self.compmode is not None:
      set_mode(self.compmode)
    if self.float_ is not None:
      set_float(self.float_)
    if self.int_ is not None:
      set_int(self.int_)

  def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
    set_dt(self.old_dt)
    set_mode(self.old_mode)
    set_int(self.old_int)
    set_float(self.old_float)

  def clone(self):
    return self.__class__(dt=self.dt,
                          mode=self.compmode,
                          float_=self.float_,
                          int_=self.int_)
