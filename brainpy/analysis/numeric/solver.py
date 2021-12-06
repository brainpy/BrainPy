# -*- coding: utf-8 -*-

import jax.lax
import jax.numpy as jnp
import numpy as np
from scipy import optimize

import brainpy.math as bm

__all__ = [
  'brentq',
  'find_root_of_2d'
]

_ECONVERGED = 0
_ECONVERR = -1


def _logical_or(a, b):
  a = a.value if isinstance(a, bm.JaxArray) else a
  b = b.value if isinstance(b, bm.JaxArray) else b
  return jnp.logical_or(a, b)


def _logical_and(a, b):
  a = a.value if isinstance(a, bm.JaxArray) else a
  b = b.value if isinstance(b, bm.JaxArray) else b
  return jnp.logical_and(a, b)


def _where(p, a, b):
  p = p.value if isinstance(p, bm.JaxArray) else p
  a = a.value if isinstance(a, bm.JaxArray) else a
  b = b.value if isinstance(b, bm.JaxArray) else b
  return jnp.where(p, a, b)


def f_without_jaxarray_return(f):
  def f2(*args, **kwargs):
    r = f(*args, **kwargs)
    return r.value if isinstance(r, bm.JaxArray) else r

  return f2


def brentq(fun):
  f = f_without_jaxarray_return(fun)
  if jax.config.read('jax_enable_x64'):
    rtol = 4 * jnp.finfo(jnp.float64).eps
  else:
    rtol = 2 * jnp.finfo(jnp.float32).eps

  def x(a, b, args=(), xtol=2e-14, maxiter=200):
    # Convert to float
    xpre = a * 1.0
    xcur = b * 1.0

    # Conditional checks for intervals in methods involving bisection
    fpre = f(xpre, *args)
    fcur = f(xcur, *args)

    # Root found at either end of [a,b]
    root = _where(fpre == 0, xpre, 0.)
    status = _where(fpre == 0, _ECONVERGED, _ECONVERR)
    root = _where(fcur == 0, xcur, root)
    status = _where(fcur == 0, _ECONVERGED, status)

    # Check for sign error and early termination
    # Perform Brent's method
    def _f1(x):
      x['xblk'] = x['xpre']
      x['fblk'] = x['fpre']
      x['spre'] = x['xcur'] - x['xpre']
      x['scur'] = x['xcur'] - x['xpre']
      return x

    def _f2(x):
      x['xpre'] = x['xcur']
      x['xcur'] = x['xblk']
      x['xblk'] = x['xpre']
      x['fpre'] = x['fcur']
      x['fcur'] = x['fblk']
      x['fblk'] = x['fpre']
      return x

    def _f5(x):
      x['stry'] = -x['fcur'] * (x['xcur'] - x['xpre']) / (x['fcur'] - x['fpre'])
      return x

    def _f6(x):
      x['dpre'] = (x['fpre'] - x['fcur']) / (x['xpre'] - x['xcur'])
      dblk = (x['fblk'] - x['fcur']) / (x['xblk'] - x['xcur'])
      x['stry'] = -x['fcur'] * (x['fblk'] * dblk - x['fpre'] * x['dpre']) / (dblk * x['dpre'] * (x['fblk'] - x['fpre']))
      return x

    def _f3(x):
      x = jax.lax.cond(x['xpre'] == x['xblk'], _f5, _f6, x)
      k = jnp.min(jnp.array([abs(x['spre']), 3 * abs(x['sbis']) - x['delta']]))
      j = 2 * abs(x['stry']) < k
      x['spre'] = _where(j, x['scur'], x['sbis'])
      x['scur'] = _where(j, x['stry'], x['sbis'])
      return x

    def _f4(x):  # bisect
      x['spre'] = x['sbis']
      x['scur'] = x['sbis']
      return x

    def body_fun(x):
      x['itr'] += 1
      x = jax.lax.cond(x['fpre'] * x['fcur'] < 0, _f1, lambda a: a, x)
      x = jax.lax.cond(abs(x['fblk']) < abs(x['fcur']), _f2, lambda a: a, x)
      x['delta'] = (xtol + rtol * abs(x['xcur'])) / 2
      x['sbis'] = (x['xblk'] - x['xcur']) / 2
      # Root found
      j = _logical_or(x['fcur'] == 0, abs(x['sbis']) < x['delta'])
      x['status'] = _where(j, _ECONVERGED, x['status'])
      x['root'] = _where(j, x['xcur'], x['root'])
      x = jax.lax.cond(_logical_and(abs(x['spre']) > x['delta'], abs(x['fcur']) < abs(x['fpre'])),
                       _f3, _f4, x)
      x['xpre'] = x['xcur']
      x['fpre'] = x['fcur']
      x['xcur'] += _where(abs(x['scur']) > x['delta'],
                          x['scur'], _where(x['sbis'] > 0, x['delta'], -x['delta']))
      x['fcur'] = f(x['xcur'], *args)
      x['funcalls'] += 1
      return x

    def cond_fun(R):
      return jnp.logical_and(R['status'] != _ECONVERGED, R['itr'] <= maxiter)

    R = dict(root=root, status=status, xpre=xpre, xcur=xcur, fpre=fpre, fcur=fcur,
             itr=0, funcalls=2, xblk=xpre, fblk=fpre,
             sbis=(xpre - xcur) / 2,
             delta=(xtol + rtol * abs(xcur)) / 2,
             stry=-fcur * (xcur - xpre) / (fcur - fpre),
             scur=xcur - xpre, spre=xcur - xpre,
             dpre=(fpre - fcur) / (xpre - xcur))
    R = jax.lax.cond(status == _ECONVERGED,
                     lambda x: x,
                     lambda x: jax.lax.while_loop(cond_fun, body_fun, x),
                     R)
    return dict(root=R['root'], funcalls=R['funcalls'], itr=R['itr'], status=R['status'])

  return x


def roots_of_1d_by_x(f, candidates, args=()):
  """Find the roots of the given function by numerical methods.
  """
  f = f_without_jaxarray_return(f)
  candidates = candidates.value if isinstance(candidates, bm.JaxArray) else candidates
  args = tuple(a.value if isinstance(candidates, bm.JaxArray) else a for a in args)
  vals = f(candidates, *args)
  signs = jnp.sign(vals)
  zero_sign_idx = jnp.where(signs == 0)[0]
  fps = candidates[zero_sign_idx]
  candidate_ids = jnp.where(signs[:-1] * signs[1:] < 0)[0]
  if len(candidate_ids) <= 0:
    return fps
  starts = candidates[candidate_ids]
  ends = candidates[candidate_ids + 1]
  f_opt = bm.jit(bm.vmap(brentq(f), in_axes=(0, 0, None)))
  res = f_opt(starts, ends, args)
  valid_idx = jnp.where(res['status'] == _ECONVERGED)[0]
  fps2 = res['root'][valid_idx]
  return jnp.concatenate([fps, fps2])


def get_brentq_candidates(f, xs, ys):
  f = f_without_jaxarray_return(f)
  xs = xs.value if isinstance(xs, bm.JaxArray) else xs
  ys = ys.value if isinstance(ys, bm.JaxArray) else ys
  Y, X  = jnp.meshgrid(ys, xs)
  vals = f(X, Y)
  signs = jnp.sign(vals)
  x_ids, y_ids = jnp.where(signs[:-1] * signs[1:] <= 0)
  starts = xs[x_ids]
  ends = xs[x_ids + 1]
  args = ys[y_ids]
  return starts, ends, args


def roots_of_1d_by_xy(f, starts, ends, args):
  f = f_without_jaxarray_return(f)
  f_opt = bm.jit(bm.vmap(brentq(f)))
  res = f_opt(starts, ends, (args, ))
  valid_idx = jnp.where(res['status'] == _ECONVERGED)[0]
  xs = res['root'][valid_idx]
  ys = args[valid_idx]
  return xs, ys


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
