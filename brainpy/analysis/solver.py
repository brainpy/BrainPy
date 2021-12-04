# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize

from brainpy import tools

__all__ = [
  'brentq',
  'find_root_of_1d',
  'find_root_of_2d'
]

_ECONVERGED = 0
_ECONVERR = -1


@tools.numba_jit
def brentq(f, a, b, args=(), xtol=2e-14, maxiter=200, rtol=4 * np.finfo(float).eps):
  """
  Find a root of a function in a bracketing interval using Brent's method
  adapted from Scipy's brentq.

  Uses the classic Brent's method to find a zero of the function `f` on
  the sign changing interval [a , b].

  Parameters
  ----------
  f : callable
      Python function returning a number.  `f` must be continuous.
  a : number
      One end of the bracketing interval [a,b].
  b : number
      The other end of the bracketing interval [a,b].
  args : tuple, optional(default=())
      Extra arguments to be used in the function call.
  xtol : number, optional(default=2e-12)
      The computed root ``x0`` will satisfy ``np.allclose(x, x0,
      atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
      parameter must be nonnegative.
  rtol : number, optional(default=4*np.finfo(float).eps)
      The computed root ``x0`` will satisfy ``np.allclose(x, x0,
      atol=xtol, rtol=rtol)``, where ``x`` is the exact root.
  maxiter : number, optional(default=100)
      Maximum number of iterations.
  """
  if xtol <= 0:
    raise ValueError("xtol is too small (<= 0)")
  if maxiter < 1:
    raise ValueError("maxiter must be greater than 0")

  # Convert to float
  xpre = a * 1.0
  xcur = b * 1.0

  # Conditional checks for intervals in methods involving bisection
  fpre = f(xpre, *args)
  fcur = f(xcur, *args)
  funcalls = 2

  if fpre * fcur > 0:
    raise ValueError("f(a) and f(b) must have different signs")
  root = 0.0
  status = _ECONVERR

  # Root found at either end of [a,b]
  if fpre == 0:
    root = xpre
    status = _ECONVERGED
  if fcur == 0:
    root = xcur
    status = _ECONVERGED

  root, status = root, status

  # Check for sign error and early termination
  if status == _ECONVERGED:
    itr = 0
  else:
    # Perform Brent's method
    for itr in range(maxiter):
      if fpre * fcur < 0:
        xblk = xpre
        fblk = fpre
        spre = scur = xcur - xpre
      if abs(fblk) < abs(fcur):
        xpre = xcur
        xcur = xblk
        xblk = xpre

        fpre = fcur
        fcur = fblk
        fblk = fpre

      delta = (xtol + rtol * abs(xcur)) / 2
      sbis = (xblk - xcur) / 2

      # Root found
      if fcur == 0 or abs(sbis) < delta:
        status = _ECONVERGED
        root = xcur
        itr += 1
        break

      if abs(spre) > delta and abs(fcur) < abs(fpre):
        if xpre == xblk:
          # interpolate
          stry = -fcur * (xcur - xpre) / (fcur - fpre)
        else:
          # extrapolate
          dpre = (fpre - fcur) / (xpre - xcur)
          dblk = (fblk - fcur) / (xblk - xcur)
          stry = -fcur * (fblk * dblk - fpre * dpre) / \
                 (dblk * dpre * (fblk - fpre))

        if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta):
          # good short step
          spre = scur
          scur = stry
        else:
          # bisect
          spre = sbis
          scur = sbis
      else:
        # bisect
        spre = sbis
        scur = sbis

      xpre = xcur
      fpre = fcur
      if abs(scur) > delta:
        xcur += scur
      else:
        xcur += (delta if sbis > 0 else -delta)
      fcur = f(xcur, *args)
      funcalls += 1

  if status == _ECONVERR:
    raise RuntimeError("Failed to converge")

  # x, funcalls, iterations = root, funcalls, itr
  return root, funcalls, itr


@tools.numba_jit
def find_root_of_1d(f, f_points, args=(), tol=1e-8):
  """Find the roots of the given function by numerical methods.

  Parameters
  ----------
  f : callable
      The function.
  f_points : onp.ndarray, list, tuple
      The value points.

  Returns
  -------
  roots : list
      The roots.
  """
  vals = f(f_points, *args)
  fs_len = len(f_points)
  signs = np.sign(vals)

  roots = []
  sign_l = signs[0]
  point_l = f_points[0]
  idx = 1
  while idx < fs_len and sign_l == 0.:
    roots.append(f_points[idx - 1])
    sign_l = signs[idx]
    idx += 1
  while idx < fs_len:
    sign_r = signs[idx]
    point_r = f_points[idx]
    if sign_r == 0.:
      roots.append(point_r)
      if idx + 1 < fs_len:
        sign_l = sign_r
        point_l = point_r
      else:
        break
      idx += 1
    else:
      if not np.isnan(sign_r) and sign_l != sign_r:
        root, funcalls, itr = brentq(f, point_l, point_r, args)
        if abs(f(root, *args)) < tol: roots.append(root)
      sign_l = sign_r
      point_l = point_r
      idx += 1

  return roots


def find_root_of_2d(f, x_bound, y_bound, args=(), shgo_args=None,
                    fl_tol=1e-6, xl_tol=1e-4, verbose=False):
  """Find the root of a two dimensional function.

  This function is aimed to find the root of :backend:`f(x) = 0`, where :backend:`x`
  is a vector with the shape of `(2,)`.

  Parameters
  ----------
  f : callable
      he objective function to be minimized.  Must be in the form
      ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
      and ``args`` is a tuple of any additional fixed parameters needed to
      completely specify the function.
  args : tuple, optional
      Any additional fixed parameters needed to completely specify the
      objective function.
  x_bound : sequence
      Bound for the first variable. It must be a tuple/list with the format of
      ``(min, max)``, which define the lower and upper bounds for the optimizing
      argument of `f`.
  y_bound : sequence
      Bound for the second variable. It must be a tuple/list with the format of
      ``(min, max)``, which define the lower and upper bounds for the optimizing
      argument of `f`.
  shgo_args : dict
      A dictionary which contains the arguments or the parameters of `shgo` optimizer.
      It is defined in a dictionary with fields:

          - constraints
          - n
          - iters
          - callback
          - minimizer_kwargs
          - options
          - sampling_method
  fl_tol : float
      The tolerance of the function value to recognize it as a condidate of function root point.
  xl_tol : float
      The tolerance of the l2 norm distances between this point and previous points.
      If the norm distances are all bigger than `xl_tol` means this point belong to a
      new function root point.
  verbose : bool
      Whether show the shogo results.

  Returns
  -------
  res : tuple
      The roots.
  """
  print('Using scipy.optimize.shgo to solve fixed points.')

  # 1. shgo arguments
  if shgo_args is None:
    shgo_args = dict()
  if 'sampling_method' not in shgo_args:
    shgo_args['sampling_method'] = 'sobol'
  if 'n' not in shgo_args:
    shgo_args['n'] = 400

  # 2. shgo optimization
  ret = optimize.shgo(f, [x_bound, y_bound], args, **shgo_args)
  points = np.ascontiguousarray(ret.xl)
  values = np.ascontiguousarray(ret.funl)
  if verbose:
    print(ret.xl)
    print(ret.funl)

  # 3. points
  final_points = []
  for i in range(len(values)):
    if values[i] <= fl_tol:
      # first point which is less than "fl_tol"
      if len(final_points) == 0:
        final_points.append(points[i])
        continue
      # if the l2 norm distances between points[i] and
      # previous points are all bigger than "xl_tol"
      if np.alltrue(np.linalg.norm(np.array(final_points) - points[i], axis=1) > xl_tol):
        final_points.append(points[i])
    else:
      break

  # 4. x_values, y_values
  x_values, y_values = [], []
  for p in final_points:
    x_values.append(p[0])
    y_values.append(p[1])
  x_values = np.ascontiguousarray(x_values)
  y_values = np.ascontiguousarray(y_values)

  return x_values, y_values
