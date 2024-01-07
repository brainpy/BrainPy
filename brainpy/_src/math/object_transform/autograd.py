# -*- coding: utf-8 -*-

from functools import wraps
from typing import Union, Callable, Dict, Sequence, Any, Optional, Tuple

import jax
from jax import numpy as jnp
from jax._src.api import _vjp
from jax.api_util import argnums_partial
from jax.tree_util import (tree_flatten, tree_unflatten, tree_map, tree_structure)

if jax.__version__ >= '0.4.16':
  from jax.extend import linear_util
else:
  from jax import linear_util

from brainpy import tools, check
from brainpy._src.math.ndarray import Array, _as_jax_array_
from brainpy._src.math.compat_numpy import zeros
from .base import (ObjectTransform)
from .tools import (get_stack_cache, cache_stack)
from .variables import (Variable, VariableStack, current_transform_number, new_transform)

__all__ = [
  'grad',  # gradient of scalar function
  'vector_grad',  # gradient of vector/matrix/...
  'functional_vector_grad',
  'jacobian', 'jacrev', 'jacfwd',  # gradient of jacobian
  'hessian',  # gradient of hessian
]


class GradientTransform(ObjectTransform):
  """Object-oriented Automatic Differentiation Transformation in BrainPy.
  """

  def __init__(
      self,
      fun: Callable,
      transform: Callable,

      # gradient setting
      grad_vars: Any,
      argnums: Union[int, Sequence[int]],
      return_value: bool,
      has_aux: bool,

      # other
      name: str = None,
      **transform_kwargs
  ):
    super().__init__(name=name)

    # gradient variables
    self._grad_vars, self._grad_tree = tree_flatten(grad_vars, is_leaf=_isleaf)

    # parameters
    if argnums is None and len(self._grad_vars) == 0:
      argnums = 0
    if argnums is None:
      assert len(self._grad_vars) > 0
      _argnums = 0
    elif isinstance(argnums, int):
      _argnums = (0, argnums + 2) if len(self._grad_vars) > 0 else (argnums + 2)
    else:
      _argnums = check.is_sequence(argnums, elem_type=int, allow_none=False)
      _argnums = tuple(a + 2 for a in _argnums)
      if len(self._grad_vars) > 0:
        _argnums = (0,) + _argnums
    self._nonvar_argnums = argnums
    self._argnums = _argnums
    self._return_value = return_value
    self._has_aux = has_aux

    # target function
    self.fun = fun

    # target transform
    self._dyn_vars = VariableStack()
    self._eval_dyn_vars = False
    self._transform = None
    if self._has_aux:
      self._transform = transform(
        self._f_grad_with_aux_to_transform,
        argnums=self._argnums,
        has_aux=True,
        **transform_kwargs
      )
    else:
      self._transform = transform(
        self._f_grad_without_aux_to_transform,
        argnums=self._argnums,
        has_aux=True,
        **transform_kwargs
      )

  def _f_grad_with_aux_to_transform(self, grad_vals: tuple, dyn_vals: dict, *args, **kwargs):
    for k in dyn_vals.keys():
      self._dyn_vars[k]._value = dyn_vals[k]
    for v, d in zip(self._grad_vars, grad_vals):
      v._value = d
    # Users should return the auxiliary data like::
    # >>> # 1. example of return one data
    # >>> return scalar_loss, data
    # >>> # 2. example of return multiple data
    # >>> return scalar_loss, (data1, data2, ...)
    outputs = self.fun(*args, **kwargs)
    # outputs: [0] is the value for gradient,
    #          [1] is other values for return
    output0 = tree_map(lambda a: (a.value if isinstance(a, Array) else a), outputs[0])
    return output0, (outputs, [v.value for v in self._grad_vars], self._dyn_vars.dict_data())

  def _f_grad_without_aux_to_transform(self, grad_vals: tuple, dyn_vals: dict, *args, **kwargs):
    for k in dyn_vals.keys():
      self._dyn_vars[k]._value = dyn_vals[k]
    for v, d in zip(self._grad_vars, grad_vals):
      v._value = d
    # Users should return the scalar value like this::
    # >>> return scalar_loss
    output = self.fun(*args, **kwargs)
    output0 = tree_map(lambda a: (a.value if isinstance(a, Array) else a), output)
    return output0, (output, [v.value for v in self._grad_vars], self._dyn_vars.dict_data())

  def __repr__(self):
    name = self.__class__.__name__
    f = tools.repr_object(self.fun)
    f = tools.repr_context(f, " " * (len(name) + 6))
    format_ref = (f'{name}({self.name}, target={f}, \n' +
                  f'{" " * len(name)} num_of_grad_vars={len(self._grad_vars)}, \n'
                  f'{" " * len(name)} num_of_dyn_vars={len(self._dyn_vars)})')
    return format_ref

  def _return(self, rets):
    grads, (outputs, new_grad_vs, new_dyn_vs) = rets
    for v, d in zip(self._grad_vars, new_grad_vs):
      v._value = d
    for k in new_dyn_vs.keys():
      self._dyn_vars[k]._value = new_dyn_vs[k]

    # check returned grads
    if len(self._grad_vars) > 0:
      if self._nonvar_argnums is None:
        grads = self._grad_tree.unflatten(grads)
      else:
        var_grads = self._grad_tree.unflatten(grads[0])
        arg_grads = grads[1] if isinstance(self._nonvar_argnums, int) else grads[1:]
        grads = (var_grads, arg_grads)

    # check returned value
    if self._return_value:
      # check aux
      if self._has_aux:
        return grads, outputs[0], outputs[1]
      else:
        return grads, outputs
    else:
      # check aux
      if self._has_aux:
        return grads, outputs[1]
      else:
        return grads

  def __call__(self, *args, **kwargs):
    if jax.config.jax_disable_jit:  # disable JIT
      rets = self._transform(
        [v.value for v in self._grad_vars],  # variables for gradients
        self._dyn_vars.dict_data(),  # dynamical variables
        *args,
        **kwargs
      )
      return self._return(rets)

    elif not self._eval_dyn_vars:  # evaluate dynamical variables
      stack = get_stack_cache(self.fun)
      if stack is None:
        with new_transform(self):
          with VariableStack() as stack:
            if current_transform_number() > 1:
              rets = self._transform(
                [v.value for v in self._grad_vars],  # variables for gradients
                {},  # dynamical variables
                *args,
                **kwargs
              )
            else:
              rets = jax.eval_shape(
                self._transform,
                [v.value for v in self._grad_vars],  # variables for gradients
                {},  # dynamical variables
                *args,
                **kwargs
              )
          cache_stack(self.fun, stack)

        self._dyn_vars = stack
        self._dyn_vars.remove_by_id(*[id(v) for v in self._grad_vars])
        self._eval_dyn_vars = True

        # if not the outermost transformation
        if current_transform_number():
          return self._return(rets)
      else:
        self._dyn_vars = stack
        self._dyn_vars.remove_by_id(*[id(v) for v in self._grad_vars])
        self._eval_dyn_vars = True

    rets = self._transform(
      [v.value for v in self._grad_vars],  # variables for gradients
      self._dyn_vars.dict_data(),  # dynamical variables
      *args,
      **kwargs
    )
    return self._return(rets)


def _make_grad(
    func: Callable,
    grad_vars: Optional[Union[Variable, Sequence[Variable], Dict[str, Variable]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    holomorphic: Optional[bool] = False,
    allow_int: Optional[bool] = False,
    reduce_axes: Optional[Sequence[str]] = (),
    has_aux: Optional[bool] = None,
    return_value: Optional[bool] = False,
):
  return GradientTransform(fun=func,
                           transform=jax.grad,
                           grad_vars=grad_vars,
                           argnums=argnums,
                           return_value=return_value,
                           has_aux=False if has_aux is None else has_aux,
                           holomorphic=holomorphic,
                           allow_int=allow_int,
                           reduce_axes=reduce_axes)


def grad(
    func: Optional[Callable] = None,
    grad_vars: Optional[Union[Variable, Sequence[Variable], Dict[str, Variable]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    holomorphic: Optional[bool] = False,
    allow_int: Optional[bool] = False,
    reduce_axes: Optional[Sequence[str]] = (),
    has_aux: Optional[bool] = None,
    return_value: Optional[bool] = False,
) -> Union[Callable, GradientTransform]:
  """Automatic gradient computation for functions or class objects.

  This gradient function only support scalar return. It creates a function
  which evaluates the gradient of ``func``.

  It's worthy to note that the returns are different for different argument settings (where ``arg_grads`` refers
  to the gradients of "argnums", and ``var_grads`` refers to the gradients of "grad_vars").

  1. When "grad_vars" is None
    - "has_aux=False" + "return_value=False" => ``arg_grads``.
    - "has_aux=True" + "return_value=False" => ``(arg_grads, aux_data)``.
    - "has_aux=False" + "return_value=True" => ``(arg_grads, loss_value)``.
    - "has_aux=True" + "return_value=True" => ``(arg_grads, loss_value, aux_data)``.
  2. When "grad_vars" is not None and "argnums" is None
    - "has_aux=False" + "return_value=False" => ``var_grads``.
    - "has_aux=True" + "return_value=False" => ``(var_grads, aux_data)``.
    - "has_aux=False" + "return_value=True" => ``(var_grads, loss_value)``.
    - "has_aux=True" + "return_value=True" => ``(var_grads, loss_value, aux_data)``.
  3. When "grad_vars" is not None and "argnums" is not None
    - "has_aux=False" + "return_value=False" => ``(var_grads, arg_grads)``.
    - "has_aux=True" + "return_value=False" => ``((var_grads, arg_grads), aux_data)``.
    - "has_aux=False" + "return_value=True" => ``((var_grads, arg_grads), loss_value)``.
    - "has_aux=True" + "return_value=True" => ``((var_grads, arg_grads), loss_value, aux_data)``.

  Let's see some examples below.

  Before start, let's figure out what should be provided as ``grad_vars``?
  And, what should be labeled in ``argnums``?
  Take the following codes as example:

  >>> import brainpy as bp
  >>> import brainpy.math as bm
  >>>
  >>> class Example(bp.BrainPyObject):
  >>>   def __init__(self):
  >>>     super(Example, self).__init__()
  >>>     self.x = bm.TrainVar(bm.zeros(1))
  >>>     self.y = bm.random.rand(10)
  >>>   def __call__(self, z, v):
  >>>     t1 = self.x * self.y.sum()
  >>>     t2 = bm.tanh(z * v + t1)
  >>>     return t2.mean()
  >>>
  >>> # This code is equivalent to the following function:
  >>>
  >>> x = bm.TrainVar(bm.zeros(1))
  >>> y = bm.random.rand(10)
  >>> def f(z, v):
  >>>   t1 = x * y.sum()
  >>>   t2 = bm.tanh(z * v + t1)
  >>>   return t2.mean()

  Generally speaking, all gradient variables which not provided in arguments should be
  labeled as ``grad_vars``, while all gradient variables provided in the function arguments
  should be declared in ``argnums``.
  In above codes, we try to take gradients of ``self.x`` and arguments ``z`` and ``v``, we should
  call ``brainpy.math.grad`` as:

  >>> f = Example()
  >>> f_grad = bm.grad(f, grad_vars=f.x, argnums=(0, 1))


  Examples
  --------

  Grad for a pure function:

  >>> import brainpy as bp
  >>> grad_tanh = grad(bp.math.tanh)
  >>> print(grad_tanh(0.2))
  0.961043

  Parameters
  ----------
  func : callable, function, BrainPyObject
    Function to be differentiated. Its arguments at positions specified by
    ``argnums`` should be arrays, scalars, or standard Python containers.
    Argument arrays in the positions specified by ``argnums`` must be of
    inexact (i.e., floating-point or complex) type. It should return a scalar
    (which includes arrays with shape ``()`` but not arrays with shape ``(1,)`` etc.)
  grad_vars : optional, ArrayType, sequence of ArrayType, dict
    The variables in ``func`` to take their gradients.
  argnums : optional, integer or sequence of integers
    Specifies which positional argument(s) to differentiate with respect to (default 0).
  has_aux: optional, bool
    Indicates whether ``fun`` returns a pair where the
    first element is considered the output of the mathematical function to be
    differentiated and the second element is auxiliary data. Default False.
  return_value : bool
    Whether return the loss value.
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

  Returns
  -------
  func : GradientTransform
    A function with the same arguments as ``fun``, that evaluates the gradient
    of ``fun``. If ``argnums`` is an integer then the gradient has the same
    shape and type as the positional argument indicated by that integer. If
    argnums is a tuple of integers, the gradient is a tuple of values with the
    same shapes and types as the corresponding arguments. If ``has_aux`` is True
    then a pair of (gradient, auxiliary_data) is returned.
  """

  if func is None:
    return lambda f: _make_grad(f,
                                grad_vars=grad_vars,
                                argnums=argnums,
                                holomorphic=holomorphic,
                                allow_int=allow_int,
                                reduce_axes=reduce_axes,
                                has_aux=has_aux,
                                return_value=return_value)
  else:
    return _make_grad(func=func,
                      grad_vars=grad_vars,
                      argnums=argnums,
                      holomorphic=holomorphic,
                      allow_int=allow_int,
                      reduce_axes=reduce_axes,
                      has_aux=has_aux,
                      return_value=return_value)


def _isleaf(x):
  return isinstance(x, Array)


def tree_as_jax(x):
  return tree_map(_as_jax_array_, x, is_leaf=_isleaf)


def _warp_fun_force_aux(fun: Callable, has_aux: bool):
  @wraps(fun)
  def new_fun(*args, **kwargs):
    if has_aux:
      y, aux = fun(*args, **kwargs)
      y, aux = tree_as_jax((y, aux))
      return y, (y, aux)
    else:
      y = fun(*args, **kwargs)
      y = tree_as_jax(y)
      return y, y

  return new_fun


def _warp_fun_force_return_jax(fun: Callable):
  @wraps(fun)
  def new_fun(*args, **kwargs):
    return tree_as_jax(fun(*args, **kwargs))

  return new_fun


def _jacrev(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    holomorphic: bool = False,
    allow_int: bool = False,
    has_aux: bool = False,
    return_value: bool = False
):
  """
  Jacobian of ``fun`` using reverse-mode autodiff.

  Compared to ``jax.jacrev``, this function supports returning value ("return_value").
  """
  fun = _warp_fun_force_aux(fun, has_aux)
  fun_jac = jax.jacrev(fun, argnums=argnums, holomorphic=holomorphic, allow_int=allow_int, has_aux=True)

  @wraps(fun_jac)
  def jacfun(*args, **kwargs):
    args, kwargs = tree_as_jax((args, kwargs))
    if has_aux:
      jac, (y, aux) = fun_jac(*args, **kwargs)
      return (jac, y, aux) if return_value else (jac, aux)
    else:
      jac, y = fun_jac(*args, **kwargs)
      return (jac, y) if return_value else jac

  return jacfun


def jacrev(
    func: Callable,
    grad_vars: Optional[Union[Variable, Sequence[Variable], Dict[str, Variable]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    has_aux: Optional[bool] = None,
    return_value: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> ObjectTransform:
  """Extending automatic Jacobian (reverse-mode) of ``func`` to classes.

  This function extends the JAX official ``jacrev`` to make automatic jacobian
  computation on functions and class functions. Moreover, it supports returning
  value ("return_value") and returning auxiliary data ("has_aux").

  Same as `brainpy.math.grad <./brainpy.math.autograd.grad.html>`_, the returns are
  different for different argument settings in ``brainpy.math.jacrev``.

  1. When "grad_vars" is None
    - "has_aux=False" + "return_value=False" => ``arg_grads``.
    - "has_aux=True" + "return_value=False" => ``(arg_grads, aux_data)``.
    - "has_aux=False" + "return_value=True" => ``(arg_grads, loss_value)``.
    - "has_aux=True" + "return_value=True" => ``(arg_grads, loss_value, aux_data)``.
  2. When "grad_vars" is not None and "argnums" is None
    - "has_aux=False" + "return_value=False" => ``var_grads``.
    - "has_aux=True" + "return_value=False" => ``(var_grads, aux_data)``.
    - "has_aux=False" + "return_value=True" => ``(var_grads, loss_value)``.
    - "has_aux=True" + "return_value=True" => ``(var_grads, loss_value, aux_data)``.
  3. When "grad_vars" is not None and "argnums" is not None
    - "has_aux=False" + "return_value=False" => ``(var_grads, arg_grads)``.
    - "has_aux=True" + "return_value=False" => ``((var_grads, arg_grads), aux_data)``.
    - "has_aux=False" + "return_value=True" => ``((var_grads, arg_grads), loss_value)``.
    - "has_aux=True" + "return_value=True" => ``((var_grads, arg_grads), loss_value, aux_data)``.


  Parameters
  ----------
  func: Function whose Jacobian is to be computed.
  grad_vars : optional, ArrayType, sequence of ArrayType, dict
    The variables in ``func`` to take their gradients.
  has_aux: optional, bool
    Indicates whether ``fun`` returns a pair where the
    first element is considered the output of the mathematical function to be
    differentiated and the second element is auxiliary data. Default False.
  return_value : bool
    Whether return the loss value.
  argnums: Optional, integer or sequence of integers.
    Specifies which
    positional argument(s) to differentiate with respect to (default ``0``).
  holomorphic: Optional, bool.
    Indicates whether ``fun`` is promised to be
    holomorphic. Default False.
  allow_int: Optional, bool.
    Whether to allow differentiating with
    respect to integer valued inputs. The gradient of an integer input will
    have a trivial vector-space dtype (float0). Default False.

  Returns
  -------
  fun: GradientTransform
    The transformed object.
  """
  return GradientTransform(fun=func,
                           transform=_jacrev,
                           grad_vars=grad_vars,
                           argnums=argnums,
                           return_value=return_value,
                           has_aux=False if has_aux is None else has_aux,
                           holomorphic=holomorphic,
                           allow_int=allow_int)


jacobian = jacrev


def _jacfwd(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    holomorphic: bool = False,
    has_aux: bool = False,
    return_value: bool = False
):
  """
  Jacobian of ``fun`` using forward-mode autodiff.

  Compared to ``jax.jacfwd``, this function supports returning value ("return_value").
  """
  fun = _warp_fun_force_aux(fun, has_aux)
  fun_jac = jax.jacfwd(fun, argnums=argnums, holomorphic=holomorphic, has_aux=True)

  @wraps(fun_jac)
  def jacfun(*args, **kwargs):
    args, kwargs = tree_as_jax((args, kwargs))
    if has_aux:
      jac, (y, aux) = fun_jac(*args, **kwargs)
      return (jac, y, aux) if return_value else (jac, aux)
    else:
      jac, y = fun_jac(*args, **kwargs)
      return (jac, y) if return_value else jac

  return jacfun


def jacfwd(
    func: Callable,
    grad_vars: Optional[Union[Variable, Sequence[Variable], Dict[str, Variable]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    has_aux: Optional[bool] = None,
    return_value: bool = False,
    holomorphic: bool = False,
) -> ObjectTransform:
  """Extending automatic Jacobian (forward-mode) of ``func`` to classes.

  This function extends the JAX official ``jacfwd`` to make automatic jacobian
  computation on functions and class functions. Moreover, it supports returning
  value ("return_value") and returning auxiliary data ("has_aux").

  Same as `brainpy.math.grad <./brainpy.math.autograd.grad.html>`_, the returns are
  different for different argument settings in ``brainpy.math.jacfwd``.

  1. When "grad_vars" is None
    - "has_aux=False" + "return_value=False" => ``arg_grads``.
    - "has_aux=True" + "return_value=False" => ``(arg_grads, aux_data)``.
    - "has_aux=False" + "return_value=True" => ``(arg_grads, loss_value)``.
    - "has_aux=True" + "return_value=True" => ``(arg_grads, loss_value, aux_data)``.
  2. When "grad_vars" is not None and "argnums" is None
    - "has_aux=False" + "return_value=False" => ``var_grads``.
    - "has_aux=True" + "return_value=False" => ``(var_grads, aux_data)``.
    - "has_aux=False" + "return_value=True" => ``(var_grads, loss_value)``.
    - "has_aux=True" + "return_value=True" => ``(var_grads, loss_value, aux_data)``.
  3. When "grad_vars" is not None and "argnums" is not None
    - "has_aux=False" + "return_value=False" => ``(var_grads, arg_grads)``.
    - "has_aux=True" + "return_value=False" => ``((var_grads, arg_grads), aux_data)``.
    - "has_aux=False" + "return_value=True" => ``((var_grads, arg_grads), loss_value)``.
    - "has_aux=True" + "return_value=True" => ``((var_grads, arg_grads), loss_value, aux_data)``.

  Parameters
  ----------
  func: Function whose Jacobian is to be computed.
  grad_vars : optional, ArrayType, sequence of ArrayType, dict
    The variables in ``func`` to take their gradients.
  has_aux: optional, bool
    Indicates whether ``fun`` returns a pair where the
    first element is considered the output of the mathematical function to be
    differentiated and the second element is auxiliary data. Default False.
  return_value : bool
    Whether return the loss value.
  argnums: Optional, integer or sequence of integers. Specifies which
    positional argument(s) to differentiate with respect to (default ``0``).
  holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
    holomorphic. Default False.

  Returns
  -------
  obj: GradientTransform
    The transformed object.
  """
  return GradientTransform(fun=func,
                           transform=_jacfwd,
                           grad_vars=grad_vars,
                           argnums=argnums,
                           return_value=return_value,
                           has_aux=False if has_aux is None else has_aux,
                           holomorphic=holomorphic)


def hessian(
    func: Callable,
    grad_vars: Optional[Union[Variable, Sequence[Variable], Dict[str, Variable]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    return_value: bool = False,
    holomorphic=False,
) -> ObjectTransform:
  """Hessian of ``func`` as a dense array.

  Parameters
  ----------
  func : callable, function
    Function whose Hessian is to be computed.  Its arguments at positions
    specified by ``argnums`` should be arrays, scalars, or standard Python
    containers thereof. It should return arrays, scalars, or standard Python
    containers thereof.
  grad_vars : optional, ArrayCollector, sequence of ArrayType
    The variables required to compute their gradients.
  argnums: Optional, integer or sequence of integers
    Specifies which positional argument(s) to differentiate with respect to (default ``0``).
  holomorphic : bool
    Indicates whether ``fun`` is promised to be holomorphic. Default False.
  return_value : bool
    Whether return the hessian values.

  Returns
  -------
  obj: ObjectTransform
    The transformed object.
  """

  return jacfwd(jacrev(func,
                       grad_vars=grad_vars,
                       argnums=argnums,
                       holomorphic=holomorphic),
                grad_vars=grad_vars,
                argnums=argnums,
                holomorphic=holomorphic,
                return_value=return_value)


def functional_vector_grad(
    func: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    return_value: bool = False,
    has_aux: bool = False,
    reduce_axes: Tuple = ()
):
  """
  Vector-Jacobian product of ``func`` using reverse-mode autodiff.
  """
  func = _warp_fun_force_return_jax(func)

  @wraps(func)
  def grad_fun(*args, **kwargs):
    f_partial, dyn_args = argnums_partial(linear_util.wrap_init(func, kwargs), argnums, args,
                                          require_static_args_hashable=False)
    if has_aux:
      y, vjp_fn, aux = _vjp(f_partial, *dyn_args, has_aux=True, reduce_axes=reduce_axes)
    else:
      y, vjp_fn = _vjp(f_partial, *dyn_args, has_aux=False, reduce_axes=reduce_axes)
    leaves, tree = tree_flatten(y)
    tangents = tree_unflatten(tree, [jnp.ones(l.shape, dtype=l.dtype) for l in leaves])
    grads = vjp_fn(tangents)
    if isinstance(argnums, int):
      grads = grads[0]
    if has_aux:
      return (grads, y, aux) if return_value else (grads, aux)
    else:
      return (grads, y) if return_value else grads

  return grad_fun


def vector_grad(
    func: Optional[Callable] = None,
    grad_vars: Optional[Union[Variable, Sequence[Variable], Dict[str, Variable]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    return_value: bool = False,
    has_aux: Optional[bool] = None,
) -> Union[Callable, ObjectTransform]:
  """Take vector-valued gradients for function ``func``.

  Same as `brainpy.math.grad <./brainpy.math.autograd.grad.html>`_,
  `brainpy.math.jacrev <./brainpy.math.autograd.jacrev.html>`_ and
  `brainpy.math.jacfwd <./brainpy.math.autograd.jacfwd.html>`_,
  the returns in this function are different for different argument settings.

  1. When "grad_vars" is None
    - "has_aux=False" + "return_value=False" => ``arg_grads``.
    - "has_aux=True" + "return_value=False" => ``(arg_grads, aux_data)``.
    - "has_aux=False" + "return_value=True" => ``(arg_grads, loss_value)``.
    - "has_aux=True" + "return_value=True" => ``(arg_grads, loss_value, aux_data)``.
  2. When "grad_vars" is not None and "argnums" is None
    - "has_aux=False" + "return_value=False" => ``var_grads``.
    - "has_aux=True" + "return_value=False" => ``(var_grads, aux_data)``.
    - "has_aux=False" + "return_value=True" => ``(var_grads, loss_value)``.
    - "has_aux=True" + "return_value=True" => ``(var_grads, loss_value, aux_data)``.
  3. When "grad_vars" is not None and "argnums" is not None
    - "has_aux=False" + "return_value=False" => ``(var_grads, arg_grads)``.
    - "has_aux=True" + "return_value=False" => ``((var_grads, arg_grads), aux_data)``.
    - "has_aux=False" + "return_value=True" => ``((var_grads, arg_grads), loss_value)``.
    - "has_aux=True" + "return_value=True" => ``((var_grads, arg_grads), loss_value, aux_data)``.

  Parameters
  ----------
  func: Callable
    Function whose gradient is to be computed.
  grad_vars : optional, ArrayType, sequence of ArrayType, dict
    The variables in ``func`` to take their gradients.
  has_aux: optional, bool
    Indicates whether ``fun`` returns a pair where the
    first element is considered the output of the mathematical function to be
    differentiated and the second element is auxiliary data. Default False.
  return_value : bool
    Whether return the loss value.
  argnums: Optional, integer or sequence of integers. Specifies which
    positional argument(s) to differentiate with respect to (default ``0``).

  Returns
  -------
  func : GradientTransform
    The vector gradient function.
  """

  if func is None:
    return lambda f: GradientTransform(fun=f,
                                       transform=functional_vector_grad,
                                       grad_vars=grad_vars,
                                       argnums=argnums,
                                       return_value=return_value,
                                       has_aux=False if has_aux is None else has_aux)
  else:
    return GradientTransform(fun=func,
                             transform=functional_vector_grad,
                             grad_vars=grad_vars,
                             argnums=argnums,
                             return_value=return_value,
                             has_aux=False if has_aux is None else has_aux)


def vjp_for_exp_euler(func: Callable):
  """
  Vector-Jacobian product of ``func`` using reverse-mode autodiff.
  """
  func = _warp_fun_force_return_jax(func)

  @wraps(func)
  def grad_fun(*dyn_args):
    ys, y_vjp = jax.vjp(func, *dyn_args)
    tree = tree_structure(ys)
    out_tangents = []
    for i in range(len(dyn_args)):
      raw_tangents = tuple([jnp.ones(l.shape, dtype=l.dtype) if j == i else jnp.zeros(l.shape, dtype=l.dtype)
                            for j, l in enumerate(dyn_args)])
      out_tangents.append(y_vjp(tree_unflatten(tree, raw_tangents))[i])
    return tree_unflatten(tree, tuple(out_tangents)), ys, tree_unflatten(tree, dyn_args)

  return grad_fun


def _init_tangents(leaf, n_copy, index):
  ret = zeros((n_copy,) + leaf.shape, dtype=leaf.dtype)
  ret[index] = 1.
  return ret.value
