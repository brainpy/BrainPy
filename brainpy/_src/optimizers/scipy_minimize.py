import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

import brainpy.math as bm
from brainpy import errors

soptimize = None


def scipy_minimize_with_jax(
    fun, x0,
    method=None,
    args=(),
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options=None
):
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
  global soptimize
  if soptimize is None:
    try:
      import scipy.optimize as soptimize
    except ImportError:
      raise errors.PackageMissingError(f'"scipy" must be installed when user want to use '
                                       f'function: {scipy_minimize_with_jax}')

  # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
  x0_flat, unravel = ravel_pytree(x0)

  # Wrap the objective function to consume flat _original_
  # numpy arrays and produce scalar outputs.
  def fun_wrapper(x_flat, *args):
    x = unravel(x_flat)
    r = fun(x, *args)
    r = r.value if isinstance(r, bm.Array) else r
    return float(r)

  # Wrap the gradient in a similar manner
  jac = jax.jit(jax.grad(fun))

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
  results = soptimize.minimize(
    fun_wrapper,
    x0_flat,
    args=args,
    method=method,
    jac=jac_wrapper,
    callback=callback_wrapper,
    bounds=bounds,
    constraints=constraints,
    tol=tol,
    options=options
  )

  # pack the output back into a PyTree
  results["x"] = unravel(results["x"])
  return results
