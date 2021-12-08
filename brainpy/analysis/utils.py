# -*- coding: utf-8 -*-

import inspect
from pprint import pprint

import jax.lax
import jax.numpy as jnp
import numpy as np
import numpy as onp
import scipy.optimize
from jax import grad, jit
from jax.flatten_util import ravel_pytree
from jax.scipy import optimize
from scipy import optimize
from scipy.spatial.distance import squareform, pdist

import brainpy.math as bm
from brainpy import errors, tools
from brainpy.integrators import analysis_by_ast
from brainpy.integrators import analysis_by_sympy
from brainpy.integrators.ode.base import ODEIntegrator
from brainpy.simulation.brainobjects.base import DynamicalSystem
from brainpy.simulation.utils import run_model

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


@tools.numba_jit
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
def find_root_of_1d_numpy(f, f_points, args=(), tol=1e-8):
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
        root, funcalls, itr = numpy_brentq(f, point_l, point_r, args)
        if abs(f(root, *args)) < tol: roots.append(root)
      sign_l = sign_r
      point_l = point_r
      idx += 1

  return roots


def f_without_jaxarray_return(f):
  def f2(*args, **kwargs):
    r = f(*args, **kwargs)
    return r.value if isinstance(r, bm.JaxArray) else r

  return f2


def jax_brentq(fun):
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
  f_opt = bm.jit(bm.vmap(jax_brentq(f), in_axes=(0, 0, None)))
  res = f_opt(starts, ends, args)
  valid_idx = jnp.where(res['status'] == _ECONVERGED)[0]
  fps2 = res['root'][valid_idx]
  return jnp.concatenate([fps, fps2])


def get_sign(f, xs, ys):
  f = f_without_jaxarray_return(f)
  xs = xs.value if isinstance(xs, bm.JaxArray) else xs
  ys = ys.value if isinstance(ys, bm.JaxArray) else ys
  Y, X = jnp.meshgrid(ys, xs)
  return jnp.sign(f(X, Y))


def get_sign2(f, *xyz, args=()):
  in_axes = tuple(range(len(xyz))) + tuple([None] * len(args))
  f = bm.jit(bm.vmap(f_without_jaxarray_return(f), in_axes=in_axes))
  xyz = tuple((v.value if isinstance(v, bm.JaxArray) else v) for v in xyz)
  XYZ = jnp.meshgrid(*xyz)
  XYZ = tuple(jnp.moveaxis(v, 1, 0).flatten() for v in XYZ)
  shape = (len(v) for v in xyz)
  return jnp.sign(f(*(XYZ + args))).reshape(shape)


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
  vmap_f_opt = bm.jit(bm.vmap(jax_brentq(f), in_axes=in_axes))
  all_args = vmap_args + args
  if len(all_args):
    res = vmap_f_opt(starts, ends, all_args)
  else:
    res = vmap_f_opt(starts, ends, )
  valid_idx = jnp.where(res['status'] == _ECONVERGED)[0]
  roots = res['root'][valid_idx]
  vmap_args = tuple(a[valid_idx] for a in vmap_args)
  return roots, vmap_args


def brentq_roots2(vmap_f, starts, ends, *vmap_args, args=()):
  all_args = vmap_args + args
  res = vmap_f(starts, ends, all_args)
  valid_idx = jnp.where(res['status'] == _ECONVERGED)[0]
  roots = res['root'][valid_idx]
  vmap_args = tuple(a[valid_idx] for a in vmap_args)
  return roots, vmap_args


def roots_of_1d_by_xy(f, starts, ends, args):
  f = f_without_jaxarray_return(f)
  f_opt = bm.jit(bm.vmap(jax_brentq(f)))
  res = f_opt(starts, ends, (args,))
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
    return onp.array(g_flat)

  # Wrap the callback to consume a pytree
  def callback_wrapper(x_flat, *args):
    if callback is not None:
      x = unravel(x_flat)
      return callback(x, *args)

  # Minimize with scipy
  results = scipy.optimize.minimize(fun_wrapper,
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


def keep_unique(candidates, tol=2.5e-2, verbose=False):
  """Filter unique fixed points by choosing a representative within tolerance.

  Parameters
  ----------
  candidates: np.ndarray
    The fixed points with the shape of (num_point, num_dim).

  Returns
  -------
  fps_and_ids : tuple
    A 2-tuple of (kept fixed points, ids of kept fixed points).
  """
  keep_ids = np.arange(candidates.shape[0])
  if tol <= 0.0:
    return candidates, keep_ids
  if candidates.shape[0] <= 1:
    return candidates, keep_ids

  nfps = candidates.shape[0]
  all_drop_idxs = []

  # If point a and point b are within identical_tol of each other, and the
  # a is first in the list, we keep a.
  example_idxs = np.arange(nfps)
  distances = squareform(pdist(candidates, metric="euclidean"))
  for fidx in range(nfps - 1):
    distances_f = distances[fidx, fidx + 1:]
    drop_idxs = example_idxs[fidx + 1:][distances_f <= tol]
    all_drop_idxs += list(drop_idxs)

  unique_dropidxs = np.unique(all_drop_idxs)
  keep_ids = np.setdiff1d(example_idxs, unique_dropidxs)
  if keep_ids.shape[0] > 0:
    unique_fps = candidates[keep_ids, :]
  else:
    unique_fps = np.array([], dtype=np.int64)

  if verbose:
    print(f"    Kept {unique_fps.shape[0]}/{nfps} unique fixed points "
          f"with uniqueness tolerance {tol}.")

  return unique_fps, keep_ids


def find_root_of_1d(f, *candidates, args=(), tol=1e-2):
  f2 = lambda x: f(*x)
  vals = f(*candidates, *args)
  signs = bm.sign(vals)
  signs = np.asarray(signs)  # on CPU
  print(candidates)
  print(signs)
  zero_sign_idx = np.where(signs == 0)[0]
  if len(zero_sign_idx):
    fps = np.stack(tuple(c[zero_sign_idx] for c in candidates))
  else:
    fps = np.array([])
  candidate_ids = np.where(signs[:-1] * signs[1:] < 0)[0]
  print(candidate_ids)
  if len(candidate_ids) <= 0:
    print('Find no fixed points.')
    return fps

  inits = bm.stack(tuple(c[candidate_ids] for c in candidates)).T
  f_opt = lambda x0: optimize.minimize(lambda x: (f2(x) ** 2).sum(), x0, *args, method='BFGS')
  f_opt = bm.jit(bm.vmap(f_opt))
  res = f_opt(inits.value)
  fps2, _ = keep_unique(np.asarray(res.x), tol=tol)
  if len(fps):
    return np.concatenate([fps, fps2])
  else:
    return fps2


def model_transform(model):
  # check integrals
  if isinstance(model, NumDSWrapper):
    return model
  elif isinstance(model, ODEIntegrator):  #
    model = [model]
  if isinstance(model, (list, tuple)):
    if len(model) == 0:
      raise errors.AnalyzerError(f'Found no integrators: {model}')
    model = tuple(model)
    for intg in model:
      if not isinstance(intg, ODEIntegrator):
        raise errors.AnalyzerError(f'Must be the instance of {ODEIntegrator}, but got {intg}.')
  elif isinstance(model, dict):
    if len(model) == 0:
      raise errors.AnalyzerError(f'Found no integrators: {model}')
    model = tuple(model.values())
    for intg in model:
      if not isinstance(intg, ODEIntegrator):
        raise errors.AnalyzerError(f'Must be the instance of {ODEIntegrator}, but got {intg}')
  elif isinstance(model, DynamicalSystem):
    model = tuple(model.ints().subset(ODEIntegrator).unique().values())
  else:
    raise errors.UnsupportedError(f'Dynamics analysis by symbolic approach only supports '
                                  f'list/tuple/dict of {ODEIntegrator} or {DynamicalSystem}, '
                                  f'but we got: {type(model)}: {str(model)}')

  # pars to update
  pars_update = set()
  for intg in model:
    pars_update.update(intg.parameters[1:])

  all_variables = set()
  all_parameters = set()
  for integral in model:
    if len(integral.variables) != 1:
      raise errors.AnalyzerError(f'Only supports one {ODEIntegrator.__name__} one variable, '
                                 f'but we got {len(integral.variables)} variables in {integral}.')
    var = integral.variables[0]
    if var in all_variables:
      raise errors.AnalyzerError(f'Variable name {var} has been defined before. '
                                 f'Please change another name.')
    all_variables.add(var)
    # parameters
    all_parameters.update(integral.parameters[1:])

  # form a dynamic model
  return NumDSWrapper(integrals=model,
                      variables=list(all_variables),
                      parameters=list(all_parameters),
                      pars_update=pars_update)


class NumDSWrapper(object):
  """The wrapper of a dynamical model."""

  def __init__(self,
               integrals,
               variables,
               parameters,
               pars_update=None):
    self.INTG = integrals  # all integrators
    self.F = {intg.variables[0]: intg.f for intg in integrals}  # all integrators
    self.variables = variables  # all variables
    self.parameters = parameters  # all parameters
    self.pars_update = pars_update  # the parameters to update


def num2sym(model):
  assert isinstance(model, NumDSWrapper)
  all_scope = dict(math=bm)
  analyzers = []
  for integral in model.INTG:
    assert isinstance(integral, ODEIntegrator)

    # code scope
    code_scope = dict()
    closure_vars = inspect.getclosurevars(integral.f)
    code_scope.update(closure_vars.nonlocals)
    code_scope.update(closure_vars.globals)
    if hasattr(integral.f, '__self__'):
      code_scope['self'] = integral.f.__self__
    # separate variables
    code = tools.deindent(inspect.getsource(integral.f))
    analysis = analysis_by_ast.separate_variables(code)
    variables_for_returns = analysis['variables_for_returns']
    expressions_for_returns = analysis['expressions_for_returns']
    for vi, (key, vars) in enumerate(variables_for_returns.items()):
      variables = []
      for v in vars:
        if len(v) > 1:
          raise ValueError(f'Cannot analyze multi-assignment code line: {vars}.')
        variables.append(v[0])
      expressions = expressions_for_returns[key]
      var_name = integral.variables[vi]
      DE = analysis_by_sympy.SingleDiffEq(var_name=var_name,
                                          variables=variables,
                                          expressions=expressions,
                                          derivative_expr=key,
                                          scope={k: v for k, v in code_scope.items()},
                                          func_name=integral.func_name)
      analyzers.append(DE)
    all_scope.update(code_scope)
  return SymDSWrapper(analyzers=analyzers, scopes=all_scope,
                      integrals=model.INTG,
                      variables=model.variables,
                      parameters=model.parameters,
                      pars_update=model.pars_update)


class SymDSWrapper(NumDSWrapper):
  def __init__(self,
               analyzers,
               scopes,

               integrals,
               variables,
               parameters,
               pars_update=None):
    super(SymDSWrapper, self).__init__(integrals=integrals,
                                       variables=variables,
                                       parameters=parameters,
                                       pars_update=pars_update)
    self.analyzers = analyzers
    self.scopes = scopes


class Trajectory(object):
  """Trajectory Class.

  Parameters
  ----------
  model : NumDSWrapper
    The instance of DynamicModel.
  size : int, tuple, list
    The network size.
  target_vars : dict
    The target variables, with the format of "{key: initial_v}".
  fixed_vars : dict
    The fixed variables, with the format of "{key: fixed_v}".
  pars_update : dict
    The parameters to update.
  """

  def __init__(self, model, size, target_vars, fixed_vars, pars_update, show_code=False):
    assert isinstance(model, NumDSWrapper), f'"model" must be an instance of {NumDSWrapper}, ' \
                                            f'while we got {model}'
    self.model = model
    self.target_vars = target_vars
    self.fixed_vars = fixed_vars
    self.pars_update = pars_update
    self.show_code = show_code
    self.scope = {k: v for k, v in model.scopes.items()}

    # check network size
    if isinstance(size, int):
      size = (size,)
    elif isinstance(size, (tuple, list)):
      assert isinstance(size[0], int)
      size = tuple(size)
    else:
      raise ValueError

    # monitors, variables, parameters
    self.mon = tools.DictPlus()
    self.vars_and_pars = tools.DictPlus()
    for key, val in target_vars.items():
      self.vars_and_pars[key] = np.ones(size) * val
      self.mon[key] = []
    for key, val in fixed_vars.items():
      self.vars_and_pars[key] = np.ones(size) * val
    for key, val in pars_update.items():
      self.vars_and_pars[key] = val
    self.scope['VP'] = self.vars_and_pars
    self.scope['MON'] = self.mon
    self.scope['_fixed_vars'] = fixed_vars

    code_lines = ['def run_func(t_and_dt):']
    code_lines.append('  _t, _dt = t_and_dt')
    for integral in self.model.INTG:
      assert isinstance(integral, ODEIntegrator)
      func_name = integral.func_name
      self.scope[func_name] = integral
      # update the step function
      assigns = [f'VP["{var}"]' for var in integral.variables]
      calls = [f'VP["{var}"]' for var in integral.variables]
      calls.append('_t')
      calls.extend([f'VP["{var}"]' for var in integral.parameters[1:]])
      code_lines.append(f'  {", ".join(assigns)} = {func_name}({", ".join(calls)})')
      # reassign the fixed variables
      for key, val in fixed_vars.items():
        code_lines.append(f'  VP["{key}"][:] = _fixed_vars["{key}"]')
    # monitor the target variables
    for key in target_vars.keys():
      code_lines.append(f'  MON["{key}"].append(VP["{key}"])')
    # compile
    code = '\n'.join(code_lines)
    if show_code:
      print(code)
      print()
      pprint(self.scope)
      print()

    # recompile
    exec(compile(code, '', 'exec'), self.scope)
    self.run_func = self.scope['run_func']

  def run(self, duration, report=0.1):
    if isinstance(duration, (int, float)):
      duration = [0, duration]
    elif isinstance(duration, (tuple, list)):
      assert len(duration) == 2
      duration = tuple(duration)
    else:
      raise ValueError

    # get the times
    times = np.arange(duration[0], duration[1], bm.get_dt())
    # reshape the monitor
    for key in self.mon.keys():
      self.mon[key] = []
    # run the model
    run_model(run_func=self.run_func, times=times, report=report)
    # reshape the monitor
    for key in self.mon.keys():
      self.mon[key] = np.asarray(self.mon[key])


def rescale(min_max, scale=0.01):
  """Rescale lim."""
  min_, max_ = min_max
  length = max_ - min_
  min_ -= scale * length
  max_ += scale * length
  return min_, max_


def unknown_symbol(expr, scope):
  """Examine where the given expression ``expr`` has the unknown symbol in ``scope``.
  """
  ids = tools.get_identifiers(expr)
  ids = set([id_.split('.')[0].strip() for id_ in ids])
  return ids - scope
