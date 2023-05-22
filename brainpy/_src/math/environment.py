# -*- coding: utf-8 -*-


import functools
import inspect
import os
import re
import sys
from typing import Any, Callable, TypeVar, cast

from jax import config, numpy as jnp, devices
from jax.lib import xla_bridge

from . import modes

bm = None


__all__ = [
  # context manage for environment setting
  'environment',
  'batching_environment',
  'training_environment',
  'set_environment',
  'set',

  # default data types
  'set_float', 'get_float',
  'set_int', 'get_int',
  'set_bool', 'get_bool',
  'set_complex', 'get_complex',

  # default numerical integration step
  'set_dt', 'get_dt',

  # default computation modes
  'set_mode', 'get_mode',


  # set jax environments
  'enable_x64', 'disable_x64',
  'set_platform', 'get_platform',
  'set_host_device_count',

  # device memory
  'clear_buffer_memory',
  'enable_gpu_memory_preallocation',
  'disable_gpu_memory_preallocation',

  # deprecated
  'ditype',
  'dftype',

]


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
  r"""Context-manager that sets a computing environment for brain dynamics computation.

  In BrainPy, there are several basic computation settings when constructing models,
  including ``mode`` for controlling model computing behavior, ``dt`` for numerical
  integration, ``int_`` for integer precision, and ``float_`` for floating precision.
  :py:class:`~.environment`` provides a context for model construction and
  computation. In this temporal environment, models are constructed with the given
  ``mode``, ``dt``, ``int_``, etc., environment settings.

  For instance::

    >>> import brainpy as bp
    >>> import brainpy.math as bm
    >>>
    >>> with bm.environment(mode=bm.training_mode, dt=0.1):
    >>>   lif1 = bp.neurons.LIF(1)
    >>>
    >>> with bm.environment(mode=bm.nonbatching_mode, dt=0.05, float_=bm.float64):
    >>>   lif2 = bp.neurons.LIF(1)

  """

  def __init__(
      self,
      mode: modes.Mode = None,
      dt: float = None,
      x64: bool = None,
      complex_: type = None,
      float_: type = None,
      int_: type = None,
      bool_: type = None,
  ) -> None:
    super().__init__()

    if dt is not None:
      assert isinstance(dt, float), '"dt" must a float.'
      self.old_dt = get_dt()

    if mode is not None:
      assert isinstance(mode, modes.Mode), f'"mode" must a {modes.Mode}.'
      self.old_mode = get_mode()

    if x64 is not None:
      assert isinstance(x64, bool), f'"x64" must be a bool.'
      self.old_x64 = config.read("jax_enable_x64")

    if float_ is not None:
      assert isinstance(float_, type), '"float_" must a float.'
      self.old_float = get_float()

    if int_ is not None:
      assert isinstance(int_, type), '"int_" must a type.'
      self.old_int = get_int()

    if bool_ is not None:
      assert isinstance(bool_, type), '"bool_" must a type.'
      self.old_bool = get_bool()

    if complex_ is not None:
      assert isinstance(complex_, type), '"complex_" must a type.'
      self.old_complex = get_complex()

    self.dt = dt
    self.mode = mode
    self.x64 = x64
    self.complex_ = complex_
    self.float_ = float_
    self.int_ = int_
    self.bool_ = bool_

  def __enter__(self) -> 'environment':
    if self.dt is not None: set_dt(self.dt)
    if self.mode is not None: set_mode(self.mode)
    if self.x64 is not None: set_x64(self.x64)
    if self.float_ is not None: set_float(self.float_)
    if self.int_ is not None: set_int(self.int_)
    if self.complex_ is not None: set_complex(self.complex_)
    if self.bool_ is not None: set_bool(self.bool_)
    return self

  def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
    if self.dt is not None: set_dt(self.old_dt)
    if self.mode is not None: set_mode(self.old_mode)
    if self.x64 is not None: set_x64(self.old_x64)
    if self.int_ is not None: set_int(self.old_int)
    if self.float_ is not None:  set_float(self.old_float)
    if self.complex_ is not None:  set_complex(self.old_complex)
    if self.bool_ is not None:  set_bool(self.old_bool)

  def clone(self):
    return self.__class__(dt=self.dt,
                          mode=self.mode,
                          x64=self.x64,
                          bool_=self.bool_,
                          complex_=self.complex_,
                          float_=self.float_,
                          int_=self.int_)

  def __eq__(self, other):
    return id(self) == id(other)


class training_environment(environment):
  """Environment with the training mode.

  This is a short-cut context setting for an environment with the training mode.
  It is equivalent to::

    >>> import brainpy.math as bm
    >>> with bm.environment(mode=bm.training_mode):
    >>>   pass

  """

  def __init__(
      self,
      dt: float = None,
      x64: bool = None,
      complex_: type = None,
      float_: type = None,
      int_: type = None,
      bool_: type = None,
      batch_size: int = 1,
  ):
    super().__init__(dt=dt,
                     x64=x64,
                     complex_=complex_,
                     float_=float_,
                     int_=int_,
                     bool_=bool_,
                     mode=modes.TrainingMode(batch_size))


class batching_environment(environment):
  """Environment with the batching mode.

  This is a short-cut context setting for an environment with the batching mode.
  It is equivalent to::

    >>> import brainpy.math as bm
    >>> with bm.environment(mode=bm.batching_mode):
    >>>   pass


  """

  def __init__(
      self,
      dt: float = None,
      x64: bool = None,
      complex_: type = None,
      float_: type = None,
      int_: type = None,
      bool_: type = None,
      batch_size: int = 1,
  ):
    super().__init__(dt=dt,
                     x64=x64,
                     complex_=complex_,
                     float_=float_,
                     int_=int_,
                     bool_=bool_,
                     mode=modes.BatchingMode(batch_size))


def set(
    mode: modes.Mode = None,
    dt: float = None,
    x64: bool = None,
    complex_: type = None,
    float_: type = None,
    int_: type = None,
    bool_: type = None,
):
  """Set the default computation environment.

  Parameters
  ----------
  mode: Mode
    The computing mode.
  dt: float
    The numerical integration precision.
  x64: bool
    Enable x64 computation.
  complex_: type
    The complex data type.
  float_
    The floating data type.
  int_
    The integer data type.
  bool_
    The bool data type.
  """
  if dt is not None:
    assert isinstance(dt, float), '"dt" must a float.'
    set_dt(dt)

  if mode is not None:
    assert isinstance(mode, modes.Mode), f'"mode" must a {modes.Mode}.'
    set_mode(mode)

  if x64 is not None:
    assert isinstance(x64, bool), f'"x64" must be a bool.'
    set_x64(x64)

  if float_ is not None:
    assert isinstance(float_, type), '"float_" must a float.'
    set_float(float_)

  if int_ is not None:
    assert isinstance(int_, type), '"int_" must a type.'
    set_int(int_)

  if bool_ is not None:
    assert isinstance(bool_, type), '"bool_" must a type.'
    set_bool(bool_)

  if complex_ is not None:
    assert isinstance(complex_, type), '"complex_" must a type.'
    set_complex(complex_)


set_environment = set


# default dtype
# --------------------------


def ditype():
  """Default int type.

  .. deprecated:: 2.3.1
     Use `brainpy.math.int_` instead.
  """
  # raise errors.NoLongerSupportError('\nGet default integer data type through `ditype()` has been deprecated. \n'
  #                                   'Use `brainpy.math.int_` instead.')
  global bm
  if bm is None: from brainpy import math as bm
  return bm.int_


def dftype():
  """Default float type.

  .. deprecated:: 2.3.1
     Use `brainpy.math.float_` instead.
  """

  # raise errors.NoLongerSupportError('\nGet default floating data type through `dftype()` has been deprecated. \n'
  #                                   'Use `brainpy.math.float_` instead.')
  global bm
  if bm is None: from brainpy import math as bm
  return bm.float_


def set_float(dtype: type):
  """Set global default float type.

  Parameters
  ----------
  dtype: type
    The float type.
  """
  if dtype not in [jnp.float16, jnp.float32, jnp.float64, ]:
    raise TypeError(f'Float data type {dtype} is not supported.')
  global bm
  if bm is None: from brainpy import math as bm
  bm.__dict__['float_'] = dtype


def get_float():
  """Get the default float data type.
  
  Returns
  -------
  dftype: type
    The default float data type.
  """
  global bm
  if bm is None: from brainpy import math as bm
  return bm.float_


def set_int(dtype: type):
  """Set global default integer type.

  Parameters
  ----------
  dtype: type
    The integer type.
  """
  if dtype not in [jnp.int8, jnp.int16, jnp.int32, jnp.int64,
                   jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64, ]:
    raise TypeError(f'Integer data type {dtype} is not supported.')
  global bm
  if bm is None: from brainpy import math as bm
  bm.__dict__['int_'] = dtype


def get_int():
  """Get the default int data type.

  Returns
  -------
  dftype: type
    The default int data type.
  """
  global bm
  if bm is None: from brainpy import math as bm
  return bm.int_


def set_bool(dtype: type):
  """Set global default boolean type.

  Parameters
  ----------
  dtype: type
    The bool type.
  """
  global bm
  if bm is None: from brainpy import math as bm
  bm.__dict__['bool_'] = dtype


def get_bool():
  """Get the default boolean data type.

  Returns
  -------
  dftype: type
    The default bool data type.
  """
  global bm
  if bm is None: from brainpy import math as bm
  return bm.bool_


def set_complex(dtype: type):
  """Set global default complex type.

  Parameters
  ----------
  dtype: type
    The complex type.
  """
  global bm
  if bm is None: from brainpy import math as bm
  bm.__dict__['complex_'] = dtype


def get_complex():
  """Get the default complex data type.

  Returns
  -------
  dftype: type
    The default complex data type.
  """
  global bm
  if bm is None: from brainpy import math as bm
  return bm.complex_


# numerical precision
# --------------------------

def set_dt(dt):
  """Set the default numerical integrator precision.

  Parameters
  ----------
  dt : float
      Numerical integration precision.
  """
  assert isinstance(dt, float), f'"dt" must a float, but we got {dt}'
  global bm
  if bm is None: from brainpy import math as bm
  bm.__dict__['dt'] = dt


def get_dt():
  """Get the numerical integrator precision.

  Returns
  -------
  dt : float
      Numerical integration precision.
  """
  global bm
  if bm is None: from brainpy import math as bm
  return bm.dt


def set_mode(mode: modes.Mode):
  """Set the default computing mode.

  Parameters
  ----------
  mode: Mode
    The instance of :py:class:`~.Mode`.
  """
  if not isinstance(mode, modes.Mode):
    raise TypeError(f'Must be instance of brainpy.math.Mode. '
                    f'But we got {type(mode)}: {mode}')
  global bm
  if bm is None: from brainpy import math as bm
  bm.__dict__['mode'] = mode


def get_mode() -> modes.Mode:
  """Get the default computing mode.

  References
  ----------
  mode: Mode
    The default computing mode.
  """
  global bm
  if bm is None: from brainpy import math as bm
  return bm.mode


def enable_x64():
  config.update("jax_enable_x64", True)
  set_int(jnp.int64)
  set_float(jnp.float64)
  set_complex(jnp.complex128)


def disable_x64():
  config.update("jax_enable_x64", False)
  set_int(jnp.int32)
  set_float(jnp.float32)
  set_complex(jnp.complex64)


def set_x64(enable: bool):
  assert isinstance(enable, bool)
  if enable:
    enable_x64()
  else:
    disable_x64()


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

