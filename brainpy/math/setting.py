# -*- coding: utf-8 -*-

import os
import re

from jax import dtypes, config, numpy as jnp
from jax.lib import xla_bridge

__all__ = [
  'enable_x64',
  'disable_x64',
  'set_platform',
  'set_host_device_count',

  # device memory
  'clear_buffer_memory',
  'disable_gpu_memory_preallocation',
  'enable_gpu_memory_preallocation',

  # default data types
  'bool_',
  'int_',
  'float_',
  'complex_',
  'ditype',
  'dftype',

  # default numerical integration step
  'set_dt',
  'get_dt',
]

# default dtype
# --------------------------

bool_ = jnp.bool_
int_ = jnp.int32
float_ = jnp.float32
complex_ = jnp.complex_


def ditype():
  """Default int type."""
  return jnp.int64 if config.read('jax_enable_x64') else jnp.int32


def dftype():
  """Default float type."""
  return jnp.float64 if config.read('jax_enable_x64') else jnp.float32


# numerical precision
# --------------------------

__dt = 0.1


def set_dt(dt):
  """Set the numerical integrator precision.

  Parameters
  ----------
  dt : float
      Numerical integration precision.
  """
  _dt = jnp.asarray(dt)
  if not dtypes.issubdtype(_dt.dtype, jnp.floating):
    raise ValueError(f'"dt" must a float, but we got {dt}')
  if _dt.ndim != 0:
    raise ValueError(f'"dt" must be a scalar, but we got {dt}')
  global __dt
  __dt = dt


def get_dt():
  """Get the numerical integrator precision.

  Returns
  -------
  dt : float
      Numerical integration precision.
  """
  return __dt


def enable_x64(mode=True):
  assert mode in [True, False]
  config.update("jax_enable_x64", mode)


def disable_x64():
  config.update("jax_enable_x64", False)


def set_platform(platform):
  """
  Changes platform to CPU, GPU, or TPU. This utility only takes
  effect at the beginning of your program.
  """
  assert platform in ['cpu', 'gpu', 'tpu']
  config.update("jax_platform_name", platform)


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

