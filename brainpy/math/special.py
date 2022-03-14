# -*- coding: utf-8 -*-

import jax.numpy as jnp

from brainpy.tools.checking import check_integer

__all__ = [
  'poch',
  'Gamma',
  'Beta',
]


def poch(a, n):
  """ Returns the Pochhammer symbol (a)_n. """
  # First, check if 'a' is a real number (this is currently only working for reals).
  assert not isinstance(a, complex), "a must be real: %r" % a
  check_integer(n, allow_none=False, min_bound=0)
  # Compute the Pochhammer symbol.
  return 1.0 if n == 0 else jnp.prod(jnp.arange(n) + a)


def Gamma(z):
  """ Paul Godfrey's Gamma function implementation valid for z complex.
      This is converted from Godfrey's Gamma.m Matlab file available at
      https://www.mathworks.com/matlabcentral/fileexchange/3572-gamma.
      15 significant digits of accuracy for real z and 13
      significant digits for other values.
  """
  zz = z

  # Find negative real parts of z and make them positive.
  if isinstance(z, (complex, jnp.complex64, jnp.complex128)):
    Z = [z.real, z.imag]
    if Z[0] < 0:
      Z[0] = -Z[0]
      z = jnp.asarray(Z)
      z = z.astype(complex)

  g = 607 / 128.
  c = jnp.asarray([0.99999999999999709182, 57.156235665862923517, -59.597960355475491248,
                   14.136097974741747174, -0.49191381609762019978, .33994649984811888699e-4,
                   .46523628927048575665e-4, -.98374475304879564677e-4, .15808870322491248884e-3,
                   -.21026444172410488319e-3, .21743961811521264320e-3, -.16431810653676389022e-3,
                   .84418223983852743293e-4, -.26190838401581408670e-4, .36899182659531622704e-5])
  if z == 0 or z == 1:
    return 1.
  if ((jnp.round(zz) == zz)
      and (zz.imag == 0)
      and (zz.real <= 0)):  # Adjust for negative poles.
    return jnp.inf
  z = z - 1
  zh = z + 0.5
  zgh = zh + g
  zp = zgh ** (zh * 0.5)  # Trick for avoiding floating-point overflow above z = 141.
  idx = jnp.arange(len(c) - 1, 0, -1)
  ss = jnp.sum(c[idx] / (z + idx))
  sq2pi = 2.5066282746310005024157652848110
  f = (sq2pi * (c[0] + ss)) * ((zp * jnp.exp(-zgh)) * zp)
  if isinstance(zz, (complex, jnp.complex64, jnp.complex128)):
    return f.astype(complex)
  elif isinstance(zz, int) and zz >= 0:
    f = jnp.round(f)
    return f.astype(int)
  else:
    return f


def Beta(x, y):
  """ Beta function using Godfrey's Gamma function. """
  return Gamma(x) * Gamma(y) / Gamma(x + y)
