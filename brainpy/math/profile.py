# -*- coding: utf-8 -*-


import jax.numpy as jnp

__all__ = [
  'bool_',
  'int_',
  'float_',
  'complex_',

  'set_int_',
  'set_float_',
  'set_complex_',
  'set_dt',
  'get_dt',
]

# default dtype
# --------------------------


bool_ = jnp.bool_
int_ = jnp.int32
float_ = jnp.float32
complex_ = jnp.complex_


def set_int_(int_type):
  global int_
  assert isinstance(int_type, type)
  int_ = int_type


def set_float_(float_type):
  global float_
  assert isinstance(float_type, type)
  float_ = float_type


def set_complex_(complex_type):
  global complex_
  assert isinstance(complex_type, type)
  complex_ = complex_type


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
