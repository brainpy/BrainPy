# -*- coding: utf-8 -*-

import os
import re

import jax.config
import jax.numpy as jnp

__all__ = [
  'enable_x64',
  'disable_x64',
  'set_platform',
  'set_host_device_count',

  # data types
  'bool_',
  'int_',
  'float_',
  'complex_',

  # change default data types
  'set_dt',
  'get_dt',
]

# default dtype
# --------------------------

bool_ = jnp.bool_
int_ = jnp.int32
float_ = jnp.float32
complex_ = jnp.complex_


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
  assert isinstance(dt, float), f'"dt" must a float, but we got {dt}'
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
  jax.config.update("jax_enable_x64", mode)


def disable_x64():
  jax.config.update("jax_enable_x64", False)


def set_platform(platform):
  """
  Changes platform to CPU, GPU, or TPU. This utility only takes
  effect at the beginning of your program.
  """
  assert platform in ['cpu', 'gpu', 'tpu']
  jax.config.update("jax_platform_name", platform)


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
