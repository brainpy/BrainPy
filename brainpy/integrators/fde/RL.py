# -*- coding: utf-8 -*-

import jax.numpy as jnp
from jax import vmap
from jax.lax import cond

from brainpy.math.special import Gamma
from brainpy.tools.checking import check_float

__all__ = [
  'RL',
]


def RLcoeffs(index_k, index_j, alpha):
  """Calculates coefficients for the RL differintegral operator.

  see Baleanu, D., Diethelm, K., Scalas, E., and Trujillo, J.J. (2012). Fractional
      Calculus: Models and Numerical Methods. World Scientific.
  """

  def f1(x):
    k, j = x
    return ((k - 1) ** (1 - alpha) -
            (k + alpha - 1) * k ** -alpha)

  def f2(x):
    k, j = x
    return cond(k == j, lambda _: 1., f3, x)

  def f3(x):
    k, j = x
    return ((k - j + 1) ** (1 - alpha) +
            (k - j - 1) ** (1 - alpha) -
            2 * (k - j) ** (1 - alpha))

  return cond(index_j == 0, f1, f2, (index_k, index_j))


def RLmatrix(alpha, N):
  """ Define the coefficient matrix for the RL algorithm. """
  ij = jnp.tril_indices(N, -1)
  coeff = vmap(RLcoeffs, in_axes=(0, 0, None))(ij[0], ij[1], alpha)
  mat = jnp.zeros((N, N)).at[ij].set(coeff)
  diagonal = jnp.arange(N)
  mat = mat.at[diagonal, diagonal].set(1.)
  return mat / Gamma(2 - alpha)


def RL(alpha, f, domain_start=0.0, domain_end=1.0, dt=0.01):
  """ Calculate the RL algorithm using a trapezoid rule over
      an array of function values.

  Examples
  --------

  >>> RL_sqrt = RL(0.5, lambda x: x ** 0.5)
  >>> RL_poly = RL(0.5, lambda x: x**2 - 1, 0., 1., 100)

  Parameters
  ----------
  alpha : float
    The order of the differintegral to be computed.
  f : function
    This is the function that is to be differintegrated.
  domain_start : float, int
    The left-endpoint of the function domain. Default value is 0.
  domain_end : float, int
    The right-endpoint of the function domain; the point at which the
    differintegral is being evaluated. Default value is 1.
  dt : float, int
    The number of points in the domain. Default value is 100.

  Returns
  -------
  RL : float 1d-array
      Each element of the array is the RL differintegral evaluated at the
      corresponding function array index.
  """
  # checking
  assert domain_start < domain_end, ('"domain_start" should be lower than "domain_end", ' 
                                     f'while we got {domain_start} >= {domain_end}')
  check_float(alpha, 'alpha', allow_none=False)
  check_float(domain_start, 'domain_start', allow_none=False)
  check_float(domain_end, 'domain_start', allow_none=False)
  check_float(dt, 'dt', allow_none=False)
  # computing
  points = jnp.arange(domain_start, domain_end, dt)
  f_values = vmap(f)(points)
  # Calculate the RL differintegral.
  D = RLmatrix(alpha, points.shape[0])
  RL = dt ** -alpha * jnp.dot(D, f_values)
  return RL


