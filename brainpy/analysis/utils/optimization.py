# -*- coding: utf-8 -*-


import jax.lax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax.flatten_util import ravel_pytree

import brainpy.math as bm
from brainpy import tools, errors
from . import f_without_jaxarray_return

try:
  import scipy.optimize as soptimize
except (ModuleNotFoundError, ImportError):
  soptimize = None


__all__ = [
  'ECONVERGED', 'ECONVERR',

  'jax_brentq',
  'get_brentq_candidates',
  'brentq_candidates',
  'brentq_roots',
  'brentq_roots2',
  'scipy_minimize_with_jax',
  'roots_of_1d_by_x',
  'roots_of_1d_by_xy',
]

ECONVERGED = 0
ECONVERR = -1


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


def jax_brentq(fun):
  f = f_without_jaxarray_return(fun)
  assert jax.config.read('jax_enable_x64'), ('Brentq optimization need x64 support. '
                                             'Please enable x64 with "brainpy.math.enable_x64()"')
  rtol = 4 * jnp.finfo(jnp.float64).eps
  # if jax.config.read('jax_enable_x64'):
  #   rtol = 4 * jnp.finfo(jnp.float64).eps
  # else:
  #   rtol = 1.5 * jnp.finfo(jnp.float32).eps

  def x(a, b, args=(), xtol=2e-14, maxiter=200):
    # Convert to float
    xpre = a * 1.0
    xcur = b * 1.0

    # Conditional checks for intervals in methods involving bisection
    fpre = f(xpre, *args)
    fcur = f(xcur, *args)

    # Root found at either end of [a,b]
    root = _where(fpre == 0, xpre, 0.)
    status = _where(fpre == 0, ECONVERGED, ECONVERR)
    root = _where(fcur == 0, xcur, root)
    status = _where(fcur == 0, ECONVERGED, status)

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
      _tmp = dblk * x['dpre'] * (x['fblk'] - x['fpre'])
      x['stry'] = -x['fcur'] * (x['fblk'] * dblk - x['fpre'] * x['dpre']) / _tmp
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
      x['status'] = _where(j, ECONVERGED, x['status'])
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
      return jnp.logical_and(R['status'] != ECONVERGED, R['itr'] <= maxiter)

    R = dict(root=root, status=status, xpre=xpre, xcur=xcur, fpre=fpre, fcur=fcur,
             itr=0, funcalls=2, xblk=xpre, fblk=fpre,
             sbis=(xpre - xcur) / 2,
             delta=(xtol + rtol * abs(xcur)) / 2,
             stry=-fcur * (xcur - xpre) / (fcur - fpre),
             scur=xcur - xpre, spre=xcur - xpre,
             dpre=(fpre - fcur) / (xpre - xcur))
    R = jax.lax.cond(status == ECONVERGED,
                     lambda x: x,
                     lambda x: jax.lax.while_loop(cond_fun, body_fun, x),
                     R)
    return dict(root=R['root'], funcalls=R['funcalls'], itr=R['itr'], status=R['status'])

  return x


def get_brentq_candidates(f, xs, ys):
  f = f_without_jaxarray_return(f)
  xs = xs.value if isinstance(xs, bm.JaxArray) else xs
  ys = ys.value if isinstance(ys, bm.JaxArray) else ys
  Y, X = jnp.meshgrid(ys, xs)
  vals = f(X, Y)
  signs = jnp.sign(vals)
  x_ids, y_ids = jnp.where(signs[:-1] * signs[1:] <= 0)
  starts = xs[x_ids]
  ends = xs[x_ids + 1]
  args = ys[y_ids]
  return starts, ends, args


def brentq_candidates(vmap_f, *values, args=()):
  # change the position of meshgrid values
  values = tuple((v.value if isinstance(v, bm.JaxArray) else v) for v in values)
  xs = values[0]
  mesh_values = jnp.meshgrid(*values)
  if bm.ndim(mesh_values[0]) > 1:
    mesh_values = tuple(jnp.moveaxis(m, 0, 1) for m in mesh_values)
  mesh_values = tuple(m.flatten() for m in mesh_values)
  # function outputs
  signs = jnp.sign(vmap_f(*(mesh_values + args)))
  # compute the selected values
  signs = signs.reshape((xs.shape[0], -1))
  par_len = signs.shape[1]
  signs1 = signs.at[-1].set(1)  # discard the final row
  signs2 = jnp.vstack((signs[1:], signs[:1])).at[-1].set(1)  # discard the first row
  ids = jnp.where((signs1 * signs2).flatten() <= 0)[0]
  x_starts = mesh_values[0][ids]
  x_ends = mesh_values[0][ids + par_len]
  other_vals = tuple(v[ids] for v in mesh_values[1:])
  return x_starts, x_ends, other_vals


def brentq_roots(f, starts, ends, *vmap_args, args=()):
  in_axes = (0, 0, tuple([0] * len(vmap_args)) + tuple([None] * len(args)))
  vmap_f_opt = bm.jit(vmap(jax_brentq(f), in_axes=in_axes))
  all_args = vmap_args + args
  if len(all_args):
    res = vmap_f_opt(starts, ends, all_args)
  else:
    res = vmap_f_opt(starts, ends, )
  valid_idx = jnp.where(res['status'] == ECONVERGED)[0]
  roots = res['root'][valid_idx]
  vmap_args = tuple(a[valid_idx] for a in vmap_args)
  return roots, vmap_args


def brentq_roots2(vmap_f, starts, ends, *vmap_args, args=()):
  all_args = vmap_args + args
  res = vmap_f(starts, ends, all_args)
  valid_idx = jnp.where(res['status'] == ECONVERGED)[0]
  roots = res['root'][valid_idx]
  vmap_args = tuple(a[valid_idx] for a in vmap_args)
  return roots, vmap_args


def scipy_minimize_with_jax(fun, x0,
                            method=None,
                            args=(),
                            bounds=None,
                            constraints=(),
                            tol=None,
                            callback=None,
                            options=None):
  """
  A simple wrapper for scipy.optimize.minimize using JAX.

  Parameters
  ----------
  fun: function
    The objective function to be minimized, written in JAX code
    so that it is automatically differentiable.  It is of type,
    ```fun: x, *args -> float``` where `x` is a PyTree and args
    is a tuple of the fixed parameters needed to completely specify the function.

  x0: jnp.ndarray
    Initial guess represented as a JAX PyTree.

  args: tuple, optional.
    Extra arguments passed to the objective function
    and its derivative.  Must consist of valid JAX types; e.g. the leaves
    of the PyTree must be floats.

  method : str or callable, optional
    Type of solver.  Should be one of
        - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
        - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
        - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
        - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
        - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
        - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
        - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
        - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
        - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
        - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
        - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
        - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
        - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
        - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
        - custom - a callable object (added in version 0.14.0),
          see below for description.
    If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
    depending on if the problem has constraints or bounds.

  bounds : sequence or `Bounds`, optional
    Bounds on variables for L-BFGS-B, TNC, SLSQP, Powell, and
    trust-constr methods. There are two ways to specify the bounds:
        1. Instance of `Bounds` class.
        2. Sequence of ``(min, max)`` pairs for each element in `x`. None
        is used to specify no bound.
    Note that in order to use `bounds` you will need to manually flatten
    them in the same order as your inputs `x0`.

  constraints : {Constraint, dict} or List of {Constraint, dict}, optional
    Constraints definition (only for COBYLA, SLSQP and trust-constr).
    Constraints for 'trust-constr' are defined as a single object or a
    list of objects specifying constraints to the optimization problem.
    Available constraints are:
        - `LinearConstraint`
        - `NonlinearConstraint`
    Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
    Each dictionary with fields:
        type : str
            Constraint type: 'eq' for equality, 'ineq' for inequality.
        fun : callable
            The function defining the constraint.
        jac : callable, optional
            The Jacobian of `fun` (only for SLSQP).
        args : sequence, optional
            Extra arguments to be passed to the function and Jacobian.
    Equality constraint means that the constraint function result is to
    be zero whereas inequality means that it is to be non-negative.
    Note that COBYLA only supports inequality constraints.

    Note that in order to use `constraints` you will need to manually flatten
    them in the same order as your inputs `x0`.

  tol : float, optional
    Tolerance for termination. For detailed control, use solver-specific
    options.

  options : dict, optional
      A dictionary of solver options. All methods accept the following
      generic options:
          maxiter : int
              Maximum number of iterations to perform. Depending on the
              method each iteration may use several function evaluations.
          disp : bool
              Set to True to print convergence messages.
      For method-specific options, see :func:`show_options()`.

  callback : callable, optional
      Called after each iteration. For 'trust-constr' it is a callable with
      the signature:
          ``callback(xk, OptimizeResult state) -> bool``
      where ``xk`` is the current parameter vector represented as a PyTree,
       and ``state`` is an `OptimizeResult` object, with the same fields
      as the ones from the return. If callback returns True the algorithm
      execution is terminated.

      For all the other methods, the signature is:
          ```callback(xk)```
      where `xk` is the current parameter vector, represented as a PyTree.

  Returns
  -------
  res : The optimization result represented as a ``OptimizeResult`` object.
    Important attributes are:
        ``x``: the solution array, represented as a JAX PyTree
        ``success``: a Boolean flag indicating if the optimizer exited successfully
        ``message``: describes the cause of the termination.
    See `scipy.optimize.OptimizeResult` for a description of other attributes.

  """
  if soptimize is None:
    raise errors.PackageMissingError(f'"scipy" must be installed when user want to use '
                                     f'function: {scipy_minimize_with_jax}')

  # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
  x0_flat, unravel = ravel_pytree(x0)

  # Wrap the objective function to consume flat _original_
  # numpy arrays and produce scalar outputs.
  def fun_wrapper(x_flat, *args):
    x = unravel(x_flat)
    r = fun(x, *args)
    r = r.value if isinstance(r, bm.JaxArray) else r
    return float(r)

  # Wrap the gradient in a similar manner
  jac = jit(grad(fun))

  def jac_wrapper(x_flat, *args):
    x = unravel(x_flat)
    g_flat, _ = ravel_pytree(jac(x, *args))
    return np.array(g_flat)

  # Wrap the callback to consume a pytree
  def callback_wrapper(x_flat, *args):
    if callback is not None:
      x = unravel(x_flat)
      return callback(x, *args)

  # Minimize with scipy
  results = soptimize.minimize(fun_wrapper,
                                    x0_flat,
                                    args=args,
                                    method=method,
                                    jac=jac_wrapper,
                                    callback=callback_wrapper,
                                    bounds=bounds,
                                    constraints=constraints,
                                    tol=tol,
                                    options=options)

  # pack the output back into a PyTree
  results["x"] = unravel(results["x"])
  return results


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
  f_opt = bm.jit(vmap(jax_brentq(f), in_axes=(0, 0, None)))
  res = f_opt(starts, ends, args)
  valid_idx = jnp.where(res['status'] == ECONVERGED)[0]
  fps2 = res['root'][valid_idx]
  return jnp.concatenate([fps, fps2])


def roots_of_1d_by_xy(f, starts, ends, args):
  f = f_without_jaxarray_return(f)
  f_opt = bm.jit(vmap(jax_brentq(f)))
  res = f_opt(starts, ends, (args,))
  valid_idx = jnp.where(res['status'] == ECONVERGED)[0]
  xs = res['root'][valid_idx]
  ys = args[valid_idx]
  return xs, ys


# @tools.numba_jit
def numpy_brentq(f, a, b, args=(), xtol=2e-14, maxiter=200, rtol=4 * np.finfo(float).eps):
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
  status = ECONVERR

  # Root found at either end of [a,b]
  if fpre == 0:
    root = xpre
    status = ECONVERGED
  if fcur == 0:
    root = xcur
    status = ECONVERGED

  root, status = root, status

  # Check for sign error and early termination
  if status == ECONVERGED:
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
        status = ECONVERGED
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

  if status == ECONVERR:
    raise RuntimeError("Failed to converge")

  # x, funcalls, iterations = root, funcalls, itr
  return root, funcalls, itr


# @tools.numba_jit
def find_root_of_1d_numpy(f, f_points, args=(), tol=1e-8):
  """Find the roots of the given function by numerical methods.

  Parameters
  ----------
  f : callable
      The function.
  f_points : np.ndarray, list, tuple
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
        root, funcalls, itr = numpy_brentq(f, point_l, point_r, args)
        if abs(f(root, *args)) < tol: roots.append(root)
      sign_l = sign_r
      point_l = point_r
      idx += 1

  return roots
