# -*- coding: utf-8 -*-


import inspect
from functools import partial
from typing import Union, Sequence

import jax
import jax.linear_util as lu
import jax.numpy
import numpy as np
from jax._src.api import _check_callable, _check_input_dtype_jacrev, _ensure_index
from jax._src.api import _check_input_dtype_jacfwd, _check_output_dtype_jacfwd, _std_basis
from jax._src.api import _check_output_dtype_jacrev, _unravel_array_into_pytree, _dtype, _vjp
from jax._src.api import core, safe_zip
from jax._src.api import flatten_fun_nokwargs, vmap
from jax.api_util import argnums_partial
from jax.interpreters import ad
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_transpose, tree_structure
try:
  from jax.errors import UnexpectedTracerError
except ImportError:
  from jax.core import UnexpectedTracerError

from brainpy import errors
from brainpy.base.base import Base
from brainpy.base.collector import TensorCollector
from brainpy.math.jax.jaxarray import JaxArray, TrainVar

__all__ = [
  'grad', 'jacobian', 'jacrev', 'jacfwd', 'hessian',
  'Grad', 'Jacobian',
]


class AutoGrad(Base):
  pass


def _check_vars(variables, prefix=''):
  if variables is None:
    _, tree = tree_flatten(variables)
    return None, tree
  if isinstance(variables, JaxArray):
    _, tree = tree_flatten(variables)
    variables = TensorCollector({f'_{prefix}_v{0}': variables})
  elif isinstance(variables, dict):
    _, tree = tree_flatten(dict(variables))
    variables = TensorCollector(variables)
  elif isinstance(variables, (list, tuple)):
    _, tree = tree_flatten(variables)
    variables = TensorCollector({f'_{prefix}_v{i}': v for i, v in enumerate(variables)})
  else:
    raise ValueError
  for v in variables.values():
    if not isinstance(v, JaxArray):
      raise ValueError(f'"vars" only supports dict of JaxArray, but got {type(v)}: {v}')
  return variables, tree


def _separate_vars_and_grad_vars(vars, grad_vars):
  if vars is None:
    if grad_vars is None:
      _, grad_tree = tree_flatten(None)
      return None, (None, grad_tree)
    else:
      return TensorCollector(), _check_vars(grad_vars, prefix='grad_var')
  else:
    if grad_vars is None:
      grad_vars = vars.subset(TrainVar).unique()
      _, grad_tree = tree_flatten(dict(grad_vars))
    else:
      grad_vars, grad_tree = _check_vars(grad_vars, prefix='grad_var')
    grad_var_ids = [id(v) for v in grad_vars.values()]
    dyn_vars = TensorCollector()
    for key, var in vars.items():
      if id(var) not in grad_var_ids:
        dyn_vars[key] = var
    return dyn_vars.unique(), (grad_vars, grad_tree)


def _grad_checking(func, vars, grad_vars):
  if not callable(func):
    raise ValueError(f'Must be a callable object. But we got {func}')

  # vars and grad_vars
  # if vars is None:
  #   if isinstance(func, Base):
  #     vars = func.vars()
  #   elif hasattr(func, '__self__') and isinstance(func.__self__, Base):
  #     vars = func.__self__.vars()
  vars, _ = _check_vars(vars, prefix='dyn_var')
  vars, (grad_vars, grad_tree) = _separate_vars_and_grad_vars(vars, grad_vars)
  return func, vars, grad_vars, grad_tree


def grad(func, dyn_vars=None, grad_vars=None, argnums=None, has_aux=None,
         holomorphic=False, allow_int=False, reduce_axes=(), return_value=False):
  """Automatic Gradient Computation in JAX backend.

  Creates a function which evaluates the gradient of ``fun``.

  Parameters
  ----------
  func : function, Base
    Function to be differentiated. Its arguments at positions specified by
    ``argnums`` should be arrays, scalars, or standard Python containers.
    Argument arrays in the positions specified by ``argnums`` must be of
    inexact (i.e., floating-point or complex) type. It should return a scalar
    (which includes arrays with shape ``()`` but not arrays with shape ``(1,)`` etc.)
  dyn_vars : optional, JaxArray, dict of str, sequence of JaxArray
  grad_vars : optional, JaxArray, dict of str, sequence of JaxArray
  argnums : optional, integer or sequence of integers
    Specifies which positional argument(s) to differentiate with respect to (default 0).
  has_aux: optional, bool
    Indicates whether ``fun`` returns a pair where the
    first element is considered the output of the mathematical function to be
    differentiated and the second element is auxiliary data. Default False.
  holomorphic: optional, bool
    Indicates whether ``fun`` is promised to be
    holomorphic. If True, inputs and outputs must be complex. Default False.
  allow_int: optional, bool
    Whether to allow differentiating with
    respect to integer valued inputs. The gradient of an integer input will
    have a trivial vector-space dtype (float0). Default False.
  reduce_axes: optional, tuple of int
    tuple of axis names. If an axis is listed here, and
    ``fun`` implicitly broadcasts a value over that axis, the backward pass
    will perform a ``psum`` of the corresponding gradient. Otherwise, the
    gradient will be per-example over named axes. For example, if ``'batch'``
    is a named batch axis, ``grad(f, reduce_axes=('batch',))`` will create a
    function that computes the total gradient while ``grad(f)`` will create
    one that computes the per-example gradient.
  return_value : bool
    Whether return the loss value.

  Returns
  -------
  func : function
    A function with the same arguments as ``fun``, that evaluates the gradient
    of ``fun``. If ``argnums`` is an integer then the gradient has the same
    shape and type as the positional argument indicated by that integer. If
    argnums is a tuple of integers, the gradient is a tuple of values with the
    same shapes and types as the corresponding arguments. If ``has_aux`` is True
    then a pair of (gradient, auxiliary_data) is returned.

  Examples
  --------

  >>> import brainpy as bp
  >>> grad_tanh = grad(bp.math.tanh)
  >>> print(grad_tanh(0.2))
  0.961043
  """
  func, dyn_vars, grad_vars, grad_tree = _grad_checking(func, dyn_vars, grad_vars)

  # gradient
  if dyn_vars is None and grad_vars is None:
    has_aux = False if has_aux is None else has_aux
    argnums = 0 if argnums is None else argnums
    if return_value:
      result = jax.value_and_grad(fun=func,
                                  argnums=argnums,
                                  has_aux=has_aux,
                                  holomorphic=holomorphic,
                                  allow_int=allow_int,
                                  reduce_axes=reduce_axes)
      if has_aux:
        (ans, aux), g = result
        return g, (ans, aux)
      else:
        ans, g = result
        return g, ans
    else:
      # has_aux = True: g, aux
      # has_aux = False: g
      return jax.grad(fun=func,
                      argnums=argnums,
                      has_aux=has_aux,
                      holomorphic=holomorphic,
                      allow_int=allow_int,
                      reduce_axes=reduce_axes)

  else:
    return Grad(fun=func,
                vars=dyn_vars,
                grad_vars=grad_vars,
                grad_tree=grad_tree,
                argnums=argnums,
                has_aux=has_aux,
                holomorphic=holomorphic,
                allow_int=allow_int,
                reduce_axes=reduce_axes,
                return_value=return_value)


class Grad(AutoGrad):
  """Compute the gradients of trainable variables for the given object.

  Examples
  --------

  This example is that we return two auxiliary data, i.e., ``has_aux=True``.

  >>> import brainpy as bp
  >>> import brainpy.math as bm
  >>>
  >>> class Test(bp.Base):
  >>>   def __init__(self):
  >>>     super(Test, self).__init__()
  >>>     self.a = bm.TrainVar(bp.math.ones(1))
  >>>     self.b = bm.TrainVar(bp.math.ones(1))
  >>>
  >>>   def __call__(self, c):
  >>>     ab = self.a * self.b
  >>>     ab2 = ab * 2
  >>>     vv = ab2 + c
  >>>     return vv, (ab, ab2)
  >>>
  >>> test = Test()
  >>> test_grad = Grad(test, test.vars(), argnums=0, has_aux=True)
  >>> grads, outputs = test_grad(10.)
  >>> grads
  (DeviceArray(1., dtype=float32),
   {'Test3.a': DeviceArray([2.], dtype=float32), 'Test3.b': DeviceArray([2.], dtype=float32)})
  >>> outputs
  (JaxArray(DeviceArray([1.], dtype=float32)),
   JaxArray(DeviceArray([2.], dtype=float32)))


  This example is that we return two auxiliary data, i.e., ``has_aux=True``.

  >>> import brainpy as bp
  >>>
  >>> class Test(bp.dnn.Module):
  >>>   def __init__(self):
  >>>     super(Test, self).__init__()
  >>>     self.a = bp.TrainVar(bp.math.ones(1))
  >>>     self.b = bp.TrainVar(bp.math.ones(1))
  >>>
  >>>   def __call__(self, c):
  >>>     ab = self.a * self.b
  >>>     ab2 = ab * 2
  >>>     vv = ab2 + c
  >>>     return vv, (ab, ab2)
  >>>
  >>> test = Test()
  >>> test_grad = ValueAndGrad(test, argnums=0, has_aux=True)
  >>> outputs, grads = test_grad(10.)
  >>> grads
  (DeviceArray(1., dtype=float32),
   {'Test3.a': DeviceArray([2.], dtype=float32), 'Test3.b': DeviceArray([2.], dtype=float32)})
  >>> outputs
  (JaxArray(DeviceArray(12., dtype=float32)),
   (JaxArray(DeviceArray([1.], dtype=float32)),
    JaxArray(DeviceArray([2.], dtype=float32))))
  """

  def __init__(self, fun, grad_vars, grad_tree, vars, argnums=None, has_aux=None,
               holomorphic=False, allow_int=False, reduce_axes=(), return_value=False,
               name=None):
    super(Grad, self).__init__(name=name)

    # 'raw'
    self.raw = fun
    self.has_aux = False if has_aux is None else True
    self.holomorphic = holomorphic
    self.allow_int = allow_int
    self.reduce_axes = reduce_axes
    self.return_value = return_value
    self.grad_tree = grad_tree

    # variables
    assert isinstance(vars, TensorCollector)
    assert isinstance(grad_vars, TensorCollector)
    self.dyn_vars = list(vars.values())
    self.grad_vars = list(grad_vars.values())

    # argnums
    if argnums is None:
      argnums = (0,)
    elif isinstance(argnums, int):
      argnums = (0, argnums + 2)
    else:
      argnums = (0,) + tuple(a + 2 for a in argnums)
    self.argnums = argnums

    # final functions
    if self.has_aux:
      # Users should return the auxiliary data like:
      # ------------
      # >>> # 1. example of return one data
      # >>> return scalar_loss, data
      # >>> # 2. example of return multiple data
      # >>> return scalar_loss, (data1, data2, ...)
      def func(grad_values, dyn_values, *args, **kwargs):
        for v, d in zip(self.dyn_vars, dyn_values): v.value = d
        for v, d in zip(self.grad_vars, grad_values): v.value = d
        # outputs: [0] is the value for gradient,
        #          [1] is other values for return
        outputs = self.raw(*args, **kwargs)
        output = outputs[0].value if isinstance(outputs[0], JaxArray) else outputs[0]
        return output, (outputs, [v.value for v in self.grad_vars], [v.value for v in self.dyn_vars])
    else:
      # Users should return the scalar value like this:
      # ------------
      # >>> return scalar_loss
      def func(grad_values, dyn_values, *args, **kwargs):
        for v, d in zip(self.dyn_vars, dyn_values): v.value = d
        for v, d in zip(self.grad_vars, grad_values): v.value = d
        output = self.raw(*args, **kwargs)
        output2 = output.value if isinstance(output, JaxArray) else output
        return output2, (output, [v.value for v in self.grad_vars], [v.value for v in self.dyn_vars])

    # function for gradient
    self._call = jax.grad(fun=func,
                          argnums=self.argnums,
                          has_aux=True,
                          holomorphic=self.holomorphic,
                          allow_int=self.allow_int,
                          reduce_axes=self.reduce_axes)

  def __call__(self, *args, **kwargs):
    old_grad_values = [v.value for v in self.grad_vars]
    old_dyn_values = [v.value for v in self.dyn_vars]
    try:
      grads, (outputs, new_grad_values, new_dyn_values) = self._call(
        old_grad_values, old_dyn_values, *args, **kwargs)
    except UnexpectedTracerError as e:
      for v, d in zip(self.grad_vars, old_grad_values): v.value = d
      for v, d in zip(self.dyn_vars, old_dyn_values): v.value = d
      raise errors.JaxTracerError() from e
    for v, d in zip(self.grad_vars, new_grad_values): v.value = d
    for v, d in zip(self.dyn_vars, new_dyn_values): v.value = d
    grads_of_grad_vars = tree_unflatten(self.grad_tree, grads[0])
    grads = grads_of_grad_vars if len(self.argnums) == 1 else (grads_of_grad_vars,) + grads[1:]
    if self.return_value:
      return grads, outputs
    else:
      return (grads, outputs[1]) if self.has_aux else grads


def _argnums_partial(f, dyn_argnums, args):
  if 'require_static_args_hashable' in inspect.signature(argnums_partial).parameters.keys():
    return argnums_partial(f, dyn_argnums, args, require_static_args_hashable=False)
  else:
    return argnums_partial(f, dyn_argnums, args)


def _jac_rev_aux(fun, argnums: Union[int, Sequence[int]] = 0, holomorphic=False,
                 allow_int=False, has_aux=False, return_value=False):
  _check_callable(fun)

  if has_aux:
    def jacfun(*args, **kwargs):
      f = lu.wrap_init(fun, kwargs)
      f_partial, dyn_args = _argnums_partial(f, argnums, args)
      tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
      y, pullback, aux = _vjp(f_partial, *dyn_args, has_aux=True)
      tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
      jac = vmap(pullback)(_std_basis(y))
      jac = jac[0] if isinstance(argnums, int) else jac
      example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
      jac = tree_map(partial(_unravel_array_into_pytree, y, 0), jac)
      jac = tree_transpose(tree_structure(example_args), tree_structure(y), jac)
      return (jac, (y, aux)) if return_value else (jac, aux)

  else:
    def jacfun(*args, **kwargs):
      f = lu.wrap_init(fun, kwargs)
      f_partial, dyn_args = _argnums_partial(f, argnums, args)
      tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
      y, pullback = _vjp(f_partial, *dyn_args, has_aux=False)
      tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
      jac = vmap(pullback)(_std_basis(y))
      jac = jac[0] if isinstance(argnums, int) else jac
      example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
      jac = tree_map(partial(_unravel_array_into_pytree, y, 0), jac)
      jac = tree_transpose(tree_structure(example_args), tree_structure(y), jac)
      return (jac, y) if return_value else jac

  return jacfun


def _jvp(fun, primals, tangents, has_aux=False):
  if (not isinstance(primals, (tuple, list))) or (not isinstance(tangents, (tuple, list))):
    raise TypeError("primal and tangent arguments to jax.jvp must be tuples or lists; "
                    f"found {type(primals).__name__} and {type(tangents).__name__}.")

  ps_flat, tree_def = tree_flatten(primals)
  ts_flat, tree_def_2 = tree_flatten(tangents)
  if tree_def != tree_def_2:
    raise TypeError("primal and tangent arguments to jax.jvp must have the same tree "
                    f"structure; primals have tree structure {tree_def} whereas tangents have "
                    f"tree structure {tree_def_2}.")
  for p, t in safe_zip(ps_flat, ts_flat):
    if core.primal_dtype_to_tangent_dtype(_dtype(p)) != _dtype(t):
      raise TypeError("primal and tangent arguments to jax.jvp do not match; "
                      "dtypes must be equal, or in case of int/bool primal dtype "
                      "the tangent dtype must be float0."
                      f"Got primal dtype {_dtype(p)} and so expected tangent dtype "
                      f"{core.primal_dtype_to_tangent_dtype(_dtype(p))}, but got "
                      f"tangent dtype {_dtype(t)} instead.")
    if np.shape(p) != np.shape(t):
      raise ValueError("jvp called with different primal and tangent shapes;"
                       f"Got primal shape {np.shape(p)} and tangent shape as {np.shape(t)}")

  flat_fun, out_tree = flatten_fun_nokwargs(fun, tree_def)

  if not has_aux:
    out_primals, out_tangents = ad.jvp(flat_fun).call_wrapped(ps_flat, ts_flat)
    a = tree_unflatten(out_tree(), out_primals)
    b = tree_unflatten(out_tree(), out_tangents)
    return a, b
  else:
    jvp_fun, aux = ad.jvp(flat_fun, has_aux=True)
    out_primals, out_tangents = jvp_fun.call_wrapped(ps_flat, ts_flat)
    out_tree, aux_tree = out_tree().children()
    a = tree_unflatten(out_tree, out_primals)
    b = tree_unflatten(out_tree, out_tangents)
    # a = tree_unflatten(out_tree, (jax.numpy.array(out_primals),))  # TODO
    # b = tree_unflatten(out_tree, (jax.numpy.array(out_tangents),))  # TODO
    c = tree_unflatten(aux_tree, aux())
    return (a, b), c


def _jac_fwd_aux(fun, argnums: Union[int, Sequence[int]] = 0, holomorphic: bool = False,
                 has_aux=False, return_value=False):
  _check_callable(fun)
  argnums = _ensure_index(argnums)

  if has_aux:
    raise NotImplementedError

    def jacfun(*args, **kwargs):
      f = lu.wrap_init(fun, kwargs)
      f_partial, dyn_args = _argnums_partial(f, argnums, args)
      tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
      pushfwd = partial(_jvp, f_partial, dyn_args, has_aux=True)
      (y, jac), aux = vmap(pushfwd, out_axes=((None, -1), None))(_std_basis(dyn_args))
      tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
      example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
      jac = tree_map(partial(_unravel_array_into_pytree, example_args, -1), jac)
      return (jac, (y, aux)) if return_value else (jac, aux)
  else:
    def jacfun(*args, **kwargs):
      f = lu.wrap_init(fun, kwargs)
      f_partial, dyn_args = _argnums_partial(f, argnums, args)
      tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
      pushfwd = partial(_jvp, f_partial, dyn_args)
      y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
      tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
      example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
      jac = tree_map(partial(_unravel_array_into_pytree, example_args, -1), jac)
      return (jac, y) if return_value else jac

  return jacfun


def jacrev(func, dyn_vars=None, grad_vars=None, argnums=None, holomorphic=False,
           allow_int=False, has_aux=None, return_value=False):
  """Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.
  """
  func, dyn_vars, grad_vars, grad_tree = _grad_checking(func, dyn_vars, grad_vars)

  if dyn_vars is None and grad_vars is None:
    has_aux = False if has_aux is None else has_aux
    argnums = 0 if argnums is None else argnums
    return _jac_rev_aux(fun=func, argnums=argnums,
                        holomorphic=holomorphic, allow_int=allow_int,
                        has_aux=has_aux, return_value=return_value)
  else:
    return Jacobian(fun=func,
                    vars=dyn_vars,
                    grad_vars=grad_vars,
                    grad_tree=grad_tree,
                    holomorphic=holomorphic,
                    allow_int=allow_int,
                    has_aux=has_aux,
                    return_value=return_value,
                    method='rev')


jacobian = jacrev


def jacfwd(func, dyn_vars=None, grad_vars=None, argnums=None, holomorphic=False,
           has_aux=None, return_value=False):
  """Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

  """
  func, dyn_vars, grad_vars, grad_tree = _grad_checking(func, dyn_vars, grad_vars)

  if dyn_vars is None and grad_vars is None:
    has_aux = False if has_aux is None else has_aux
    argnums = 0 if argnums is None else argnums
    return _jac_fwd_aux(fun=func, argnums=argnums, holomorphic=holomorphic,
                        has_aux=has_aux, return_value=return_value)
  else:
    return Jacobian(fun=func,
                    vars=dyn_vars,
                    grad_vars=grad_vars,
                    grad_tree=grad_tree,
                    holomorphic=holomorphic,
                    argnums=argnums,
                    has_aux=has_aux,
                    return_value=return_value,
                    method='fwd')


class Jacobian(AutoGrad):
  """Base Class to Compute Jacobian Matrix."""

  def __init__(self, fun, vars, grad_vars, grad_tree, argnums=None, holomorphic=False, name=None,
               allow_int=False, has_aux=None, return_value=False, method='rev'):
    super(Jacobian, self).__init__(name=name)

    # 'raw'
    self.raw = fun
    self.has_aux = False if has_aux is None else True
    self.holomorphic = holomorphic
    self.allow_int = allow_int
    self.return_value = return_value
    self.method = method
    assert method in ['rev', 'fwd'], 'Only support reverse-mode "rev" and feedforward-mode "fwd"'
    self.grad_tree = grad_tree

    # variables
    assert isinstance(vars, TensorCollector)
    assert isinstance(grad_vars, TensorCollector)
    self.dyn_vars = list(vars.values())
    self.grad_vars = list(grad_vars.values())

    # argnums
    if argnums is None:
      argnums = (0,)
    elif isinstance(argnums, int):
      argnums = (0, argnums + 2)
    else:
      argnums = (0,) + tuple(a + 2 for a in argnums)
    self.argnums = argnums

    # final functions
    if has_aux:
      def func(grad_values, dyn_values, *args, **kwargs):
        for v, d in zip(self.dyn_vars, dyn_values): v.value = d
        for v, d in zip(self.grad_vars, grad_values): v.value = d
        # outputs: [0] is the value for gradient,
        #          [1] is other values for return
        outputs = self.raw(*args, **kwargs)
        output = outputs[0].value if isinstance(outputs[0], JaxArray) else outputs[0]
        return output, (outputs, [v.value for v in self.grad_vars], [v.value for v in self.dyn_vars])

    else:
      def func(grad_values, dyn_values, *args, **kwargs):
        for v, d in zip(self.dyn_vars, dyn_values): v.value = d
        for v, d in zip(self.grad_vars, grad_values): v.value = d
        outputs = self.raw(*args, **kwargs)
        output = outputs.value if isinstance(outputs, JaxArray) else outputs
        return output, (outputs, [v.value for v in self.grad_vars], [v.value for v in self.dyn_vars])

    if method == 'rev':
      self._call = _jac_rev_aux(fun=func,
                                argnums=self.argnums,
                                holomorphic=self.holomorphic,
                                allow_int=allow_int,
                                has_aux=True)
    elif method == 'fwd':
      self._call = _jac_fwd_aux(fun=func,
                                argnums=self.argnums,
                                holomorphic=self.holomorphic,
                                has_aux=True)
    else:
      raise ValueError

  def __call__(self, *args, **kwargs):
    old_grad_values = tuple([v.value for v in self.grad_vars])
    old_dyn_values = tuple([v.value for v in self.dyn_vars])
    try:
      grads, (outputs, new_grad_values, new_dyn_values) = self._call(
        old_grad_values, old_dyn_values, *args, **kwargs)
    except UnexpectedTracerError as e:
      for v, d in zip(self.grad_vars, old_grad_values): v.value = d
      for v, d in zip(self.dyn_vars, old_dyn_values): v.value = d
      raise errors.JaxTracerError() from e
    for v, d in zip(self.grad_vars, new_grad_values): v.value = d
    for v, d in zip(self.dyn_vars, new_dyn_values): v.value = d
    grads_of_grad_vars = tree_unflatten(self.grad_tree, grads[0])
    grads = grads_of_grad_vars if len(self.argnums) == 1 else (grads_of_grad_vars,) + grads[1:]
    if self.return_value:
      return grads, outputs
    else:
      return (grads, outputs[1]) if self.has_aux else grads


def hessian(fun, vars=None, grad_vars=None, argnums: Union[int, Sequence[int]] = 0,
            holomorphic=False, return_value=False):
  """Hessian of ``fun`` as a dense array.

  Parameters
  ----------
  fun : callable, function
    Function whose Hessian is to be computed.  Its arguments at positions
    specified by ``argnums`` should be arrays, scalars, or standard Python
    containers thereof. It should return arrays, scalars, or standard Python
    containers thereof.
  vars : optional, ArrayCollector, sequence of JaxArray
    The dynamical changed variables.
  grad_vars : optional, ArrayCollector, sequence of JaxArray
    The variables required to compute their gradients.
  argnums: Optional, integer or sequence of integers
    Specifies which positional argument(s) to differentiate with respect to (default ``0``).
  holomorphic : bool
    Indicates whether ``fun`` is promised to be holomorphic. Default False.
  return_value : bool
    Whether return the hessian values.
  """
  return jacfwd(jacrev(fun,
                       dyn_vars=vars,
                       grad_vars=grad_vars,
                       argnums=argnums,
                       holomorphic=holomorphic),
                dyn_vars=vars,
                grad_vars=grad_vars,
                argnums=argnums,
                holomorphic=holomorphic,
                return_value=return_value)
