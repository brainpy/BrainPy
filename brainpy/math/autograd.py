# -*- coding: utf-8 -*-

from functools import partial
from typing import Union, Callable, Dict, Sequence

import jax
import numpy as np
from jax import linear_util, dtypes, vmap, numpy as jnp
from jax._src.api import (_vjp, _jvp,
                          _check_callable,
                          _check_output_dtype_jacrev, _check_input_dtype_jacrev,
                          _check_output_dtype_jacfwd, _check_input_dtype_jacfwd, )
from jax.api_util import argnums_partial
from jax.errors import UnexpectedTracerError
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_transpose, tree_structure
from jax.util import safe_map

from brainpy import errors
from brainpy.base.naming import get_unique_name
from brainpy.math.jaxarray import JaxArray, add_context, del_context


__all__ = [
  'grad',  # gradient of scalar function
  'vector_grad',  # gradient of vector/matrix/...
  'jacobian', 'jacrev', 'jacfwd',  # gradient of jacobian
  'hessian',  # gradient of hessian
]


def _make_cls_call_func(grad_func, grad_tree, grad_vars, dyn_vars,
                        argnums, return_value, has_aux):
  name = get_unique_name('_brainpy_object_oriented_grad_')

  # outputs
  def call_func(*args, **kwargs):
    old_grad_vs = [v.value for v in grad_vars]
    old_dyn_vs = [v.value for v in dyn_vars]
    try:
      add_context(name)
      grads, (outputs, new_grad_vs, new_dyn_vs) = grad_func(old_grad_vs,
                                                            old_dyn_vs,
                                                            *args,
                                                            **kwargs)
      del_context(name)
    except UnexpectedTracerError as e:
      del_context(name)
      for v, d in zip(grad_vars, old_grad_vs): v._value = d
      for v, d in zip(dyn_vars, old_dyn_vs): v._value = d
      raise errors.JaxTracerError(variables=dyn_vars + grad_vars) from e
    except Exception as e:
      del_context(name)
      for v, d in zip(grad_vars, old_grad_vs): v._value = d
      for v, d in zip(dyn_vars, old_dyn_vs): v._value = d
      raise e
    for v, d in zip(grad_vars, new_grad_vs): v._value = d
    for v, d in zip(dyn_vars, new_dyn_vs): v._value = d

    # check returned grads
    if len(grad_vars) == 0:
      grads = grads[1] if isinstance(argnums, int) else grads[1:]
    else:
      var_grads = grad_tree.unflatten(grads[0])
      if argnums is None:
        grads = var_grads
      else:
        arg_grads = grads[1] if isinstance(argnums, int) else grads[1:]
        grads = (var_grads, arg_grads)

    # check returned value
    if return_value:
      # check aux
      if has_aux:
        return grads, outputs[0], outputs[1]
      else:
        return grads, outputs
    else:
      # check aux
      if has_aux:
        return grads, outputs[1]
      else:
        return grads

  return call_func


def _check_vars(variables):
  if variables is None:
    vars, tree = tree_flatten(variables, is_leaf=lambda a: isinstance(a, JaxArray))
    return vars, tree
  if isinstance(variables, dict):
    variables = dict(variables)
  elif isinstance(variables, (list, tuple)):
    variables = tuple(variables)
  elif isinstance(variables, JaxArray):
    pass
  else:
    raise ValueError
  vars, tree = tree_flatten(variables, is_leaf=lambda a: isinstance(a, JaxArray))
  for v in vars:
    if not isinstance(v, JaxArray):
      raise ValueError(f'"dyn_vars" and "grad_vars" only supports dict '
                       f'of JaxArray, but got {type(v)}: {v}')
  return vars, tree


def _grad_checking(func: Callable,
                   dyn_vars: Union[Dict, Sequence],
                   grad_vars: Union[Dict, Sequence]):
  # check function
  if not callable(func):
    raise ValueError(f'Must be a callable object. But we got {func}')

  # check "vars", make sure it is an instance of TensorCollector
  dyn_vars, _ = _check_vars(dyn_vars)
  grad_vars, grad_tree = _check_vars(grad_vars)

  # check the duplicate in "dyn_vars" and "grad_vars"
  new_dyn_vars = []
  _dyn_var_ids = set()
  for v in dyn_vars:
    if id(v) not in _dyn_var_ids:
      new_dyn_vars.append(v)
      _dyn_var_ids.add(id(v))
  for v in grad_vars:
    if id(v) not in _dyn_var_ids:
      new_dyn_vars.append(v)
      _dyn_var_ids.add(id(v))
  return new_dyn_vars, grad_vars, grad_tree


def _cls_grad(func, grad_vars, dyn_vars, argnums, has_aux=False,
              holomorphic=False, allow_int=False, reduce_axes=()):
  # parameters
  assert isinstance(dyn_vars, (tuple, list))  # tuple/list of JaxArray
  assert isinstance(grad_vars, (tuple, list))  # tuple/list of JaxArray
  assert isinstance(argnums, (tuple, list))  # tuple/list of int

  # gradient functions
  if has_aux:
    @partial(jax.grad, argnums=argnums, has_aux=True, holomorphic=holomorphic,
             allow_int=allow_int, reduce_axes=reduce_axes)
    def grad_func(grad_values, dyn_values, *args, **kwargs):
      for v, d in zip(dyn_vars, dyn_values): v._value = d
      for v, d in zip(grad_vars, grad_values): v._value = d
      # Users should return the auxiliary data like::
      # >>> # 1. example of return one data
      # >>> return scalar_loss, data
      # >>> # 2. example of return multiple data
      # >>> return scalar_loss, (data1, data2, ...)
      outputs = func(*args, **kwargs)
      # outputs: [0] is the value for gradient,
      #          [1] is other values for return
      output = outputs[0].value if isinstance(outputs[0], JaxArray) else outputs[0]
      return output, (outputs, [v.value for v in grad_vars], [v.value for v in dyn_vars])
  else:
    @partial(jax.grad,
             argnums=argnums, has_aux=True, holomorphic=holomorphic,
             allow_int=allow_int, reduce_axes=reduce_axes)
    def grad_func(grad_values, dyn_values, *args, **kwargs):
      for v, d in zip(dyn_vars, dyn_values): v._value = d
      for v, d in zip(grad_vars, grad_values): v._value = d
      # Users should return the scalar value like this::
      # >>> return scalar_loss
      output = func(*args, **kwargs)
      output2 = output.value if isinstance(output, JaxArray) else output
      return output2, (output, [v.value for v in grad_vars], [v.value for v in dyn_vars])

  return grad_func


def grad(func, grad_vars=None, dyn_vars=None, argnums=None, holomorphic=False,
         allow_int=False, reduce_axes=(), has_aux=None, return_value=False) -> callable:
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
  >>> class Example(bp.Base):
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
  func : callable, function, Base
    Function to be differentiated. Its arguments at positions specified by
    ``argnums`` should be arrays, scalars, or standard Python containers.
    Argument arrays in the positions specified by ``argnums`` must be of
    inexact (i.e., floating-point or complex) type. It should return a scalar
    (which includes arrays with shape ``()`` but not arrays with shape ``(1,)`` etc.)
  dyn_vars : optional, JaxArray, sequence of JaxArray, dict
    The dynamically changed variables used in ``func``.
  grad_vars : optional, JaxArray, sequence of JaxArray, dict
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
  func : function
    A function with the same arguments as ``fun``, that evaluates the gradient
    of ``fun``. If ``argnums`` is an integer then the gradient has the same
    shape and type as the positional argument indicated by that integer. If
    argnums is a tuple of integers, the gradient is a tuple of values with the
    same shapes and types as the corresponding arguments. If ``has_aux`` is True
    then a pair of (gradient, auxiliary_data) is returned.
  """

  dyn_vars, grad_vars, grad_tree = _grad_checking(func, dyn_vars, grad_vars)
  # dyn_vars -> TensorCollector
  # grad_vars -> TensorCollector
  has_aux = False if has_aux is None else has_aux

  # gradient
  if len(dyn_vars) == 0 and len(grad_vars) == 0:
    argnums = 0 if argnums is None else argnums
    if return_value:
      grad_func = jax.value_and_grad(fun=func,
                                     argnums=argnums,
                                     has_aux=has_aux,
                                     holomorphic=holomorphic,
                                     allow_int=allow_int,
                                     reduce_axes=reduce_axes)

      def call_func(*args, **kwargs):
        result = grad_func(*args, **kwargs)
        if has_aux:
          (ans, aux), g = result
          return g, ans, aux
        else:
          ans, g = result
          return g, ans

    else:
      # has_aux = True: g, aux
      # has_aux = False: g
      call_func = jax.grad(fun=func,
                           argnums=argnums,
                           has_aux=has_aux,
                           holomorphic=holomorphic,
                           allow_int=allow_int,
                           reduce_axes=reduce_axes)
  else:
    # argnums
    _argnums, _ = tree_flatten(argnums)
    _argnums = tuple(a + 2 for a in _argnums)
    if argnums is None and len(grad_vars) == 0:
      raise errors.MathError('We detect no require to compute gradients because '
                             '"grad_vars" is None and "argnums" is also None. '
                             'Please provide one of them.')
    # computation
    grad_func = _cls_grad(func=func,
                          grad_vars=grad_vars,
                          dyn_vars=dyn_vars,
                          argnums=(0,) + _argnums,
                          has_aux=has_aux,
                          holomorphic=holomorphic,
                          allow_int=allow_int,
                          reduce_axes=reduce_axes)

    call_func = _make_cls_call_func(grad_func=grad_func,
                                    grad_tree=grad_tree,
                                    grad_vars=grad_vars,
                                    dyn_vars=dyn_vars,
                                    argnums=argnums,
                                    return_value=return_value,
                                    has_aux=has_aux)

  return call_func  # Finally, return the callable function


def _unravel_array_into_pytree(pytree, axis, arr, is_leaf=None):
  leaves, treedef = tree_flatten(pytree, is_leaf=is_leaf)
  axis = axis % arr.ndim
  shapes = [arr.shape[:axis] + np.shape(l) + arr.shape[axis + 1:] for l in leaves]
  parts = arr.split(np.cumsum(safe_map(np.size, leaves[:-1])), axis)
  reshaped_parts = [x.reshape(shape) for x, shape in zip(parts, shapes)]
  return tree_unflatten(treedef, reshaped_parts, )


def _std_basis(pytree):
  leaves, _ = tree_flatten(pytree)
  ndim = sum(safe_map(np.size, leaves))
  dtype = dtypes.result_type(*leaves)
  flat_basis = jax.numpy.eye(ndim, dtype=dtype)
  return _unravel_array_into_pytree(pytree, 1, flat_basis)


_isleaf = lambda x: isinstance(x, JaxArray)


def _jacrev(fun, argnums=0, holomorphic=False, allow_int=False, has_aux=False, return_value=False):
  _check_callable(fun)

  def jacfun(*args, **kwargs):
    f = linear_util.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args, require_static_args_hashable=False)
    tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
    if has_aux:
      y, pullback, aux = _vjp(f_partial, *dyn_args, has_aux=True)
    else:
      y, pullback = _vjp(f_partial, *dyn_args, has_aux=False)
    tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
    jac = vmap(pullback)(_std_basis(y))
    jac = jac[0] if isinstance(argnums, int) else jac
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac_tree = tree_map(partial(_unravel_array_into_pytree, y, 0, is_leaf=_isleaf), jac, is_leaf=_isleaf)
    jac = tree_transpose(tree_structure(example_args), tree_flatten(y, is_leaf=_isleaf)[1], jac_tree)
    if return_value:
      return (jac, y, aux) if has_aux else (jac, y)
    else:
      return (jac, aux) if has_aux else jac

  return jacfun


def _cls_jacrev(func, grad_vars, dyn_vars, argnums,
                holomorphic=False, allow_int=False, has_aux=False):
  # parameters
  assert isinstance(dyn_vars, (tuple, list))  # tuple/list of JaxArray
  assert isinstance(grad_vars, (tuple, list))  # tuple/list of JaxArray
  assert isinstance(argnums, (tuple, list))  # tuple/list of int

  # final functions
  if has_aux:
    @partial(_jacrev, argnums=argnums, holomorphic=holomorphic,
             allow_int=allow_int, has_aux=True)
    def grad_func(grad_values, dyn_values, *args, **kwargs):
      for v, d in zip(dyn_vars, dyn_values): v._value = d
      for v, d in zip(grad_vars, grad_values): v._value = d
      # outputs: [0] is the value for gradient,
      #          [1] is other values for return
      outputs = func(*args, **kwargs)
      output = outputs[0].value if isinstance(outputs[0], JaxArray) else outputs[0]
      return output, (outputs, [v.value for v in grad_vars], [v.value for v in dyn_vars])

  else:
    @partial(_jacrev, argnums=argnums, holomorphic=holomorphic,
             allow_int=allow_int, has_aux=True)
    def grad_func(grad_values, dyn_values, *args, **kwargs):
      for v, d in zip(dyn_vars, dyn_values): v._value = d
      for v, d in zip(grad_vars, grad_values): v._value = d
      outputs = func(*args, **kwargs)
      output = outputs.value if isinstance(outputs, JaxArray) else outputs
      return output, (outputs, [v.value for v in grad_vars], [v.value for v in dyn_vars])

  return grad_func


def jacrev(func, grad_vars=None, dyn_vars=None, argnums=None, holomorphic=False,
           allow_int=False, has_aux=None, return_value=False):
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
  dyn_vars : optional, JaxArray, sequence of JaxArray, dict
    The dynamically changed variables used in ``func``.
  grad_vars : optional, JaxArray, sequence of JaxArray, dict
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
  allow_int: Optional, bool. Whether to allow differentiating with
    respect to integer valued inputs. The gradient of an integer input will
    have a trivial vector-space dtype (float0). Default False.

  """
  dyn_vars, grad_vars, grad_tree = _grad_checking(func, dyn_vars, grad_vars)
  has_aux = False if has_aux is None else has_aux

  if (len(dyn_vars) == 0) and (len(grad_vars) == 0):
    argnums = 0 if argnums is None else argnums
    return _jacrev(fun=func,
                   argnums=argnums,
                   holomorphic=holomorphic,
                   allow_int=allow_int,
                   has_aux=has_aux,
                   return_value=return_value)
  else:
    _argnums, _ = tree_flatten(argnums)
    _argnums = tuple(a + 2 for a in _argnums)
    if argnums is None and len(grad_vars) == 0:
      raise errors.MathError('We detect no require to compute gradients because '
                             '"grad_vars" is None and "argnums" is also None. '
                             'Please provide one of them.')
    # computation
    grad_func = _cls_jacrev(func=func,
                            grad_vars=grad_vars,
                            dyn_vars=dyn_vars,
                            argnums=(0,) + _argnums,
                            has_aux=has_aux,
                            holomorphic=holomorphic,
                            allow_int=allow_int)

    call_func = _make_cls_call_func(grad_func=grad_func,
                                    grad_tree=grad_tree,
                                    grad_vars=grad_vars,
                                    dyn_vars=dyn_vars,
                                    argnums=argnums,
                                    return_value=return_value,
                                    has_aux=has_aux)
    return call_func


jacobian = jacrev


def _jacfwd(fun, argnums=0, holomorphic=False, has_aux=False, return_value=False):
  _check_callable(fun)
  if has_aux and jax.__version__ < '0.2.28':
    raise NotImplementedError(f'"has_aux" only supported in jax>=0.2.28, but we detect '
                              f'the current jax version is {jax.__version__}')

  def jacfun(*args, **kwargs):
    f = linear_util.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args, require_static_args_hashable=False)
    tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
    if has_aux:
      pushfwd = partial(_jvp, f_partial, dyn_args, has_aux=True)
      y, jac, aux = vmap(pushfwd, out_axes=(None, -1, None))(_std_basis(dyn_args))
    else:
      pushfwd = partial(_jvp, f_partial, dyn_args)
      y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
    tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac = tree_map(partial(_unravel_array_into_pytree, example_args, -1, is_leaf=_isleaf), jac, is_leaf=_isleaf)
    if return_value:
      return (jac, y, aux) if has_aux else (jac, y)
    else:
      return (jac, aux) if has_aux else jac

  return jacfun


def _cls_jacfwd(func, grad_vars, dyn_vars, argnums, holomorphic=False, has_aux=False):
  # parameters
  assert isinstance(dyn_vars, (tuple, list))  # tuple/list of JaxArray
  assert isinstance(grad_vars, (tuple, list))  # tuple/list of JaxArray
  assert isinstance(argnums, (tuple, list))  # tuple/list of int

  # final functions
  if has_aux:
    @partial(_jacfwd,
             argnums=argnums, holomorphic=holomorphic, has_aux=True)
    def grad_func(grad_values, dyn_values, *args, **kwargs):
      for v, d in zip(dyn_vars, dyn_values): v._value = d
      for v, d in zip(grad_vars, grad_values): v._value = d
      # outputs: [0] is the value for gradient,
      #          [1] is other values for return
      outputs = func(*args, **kwargs)
      output = outputs[0].value if isinstance(outputs[0], JaxArray) else outputs[0]
      return output, (outputs, [v.value for v in grad_vars], [v.value for v in dyn_vars])

  else:
    @partial(_jacfwd,
             argnums=argnums, holomorphic=holomorphic, has_aux=True)
    def grad_func(grad_values, dyn_values, *args, **kwargs):
      for v, d in zip(dyn_vars, dyn_values): v._value = d
      for v, d in zip(grad_vars, grad_values): v._value = d
      outputs = func(*args, **kwargs)
      output = outputs.value if isinstance(outputs, JaxArray) else outputs
      return output, (outputs, [v.value for v in grad_vars], [v.value for v in dyn_vars])

  return grad_func


def jacfwd(func, grad_vars=None, dyn_vars=None, argnums=None, holomorphic=False,
           has_aux=None, return_value=False):
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
  dyn_vars : optional, JaxArray, sequence of JaxArray, dict
    The dynamically changed variables used in ``func``.
  grad_vars : optional, JaxArray, sequence of JaxArray, dict
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
  """
  dyn_vars, grad_vars, grad_tree = _grad_checking(func, dyn_vars, grad_vars)
  has_aux = False if has_aux is None else has_aux

  if (len(dyn_vars) == 0) and (len(grad_vars) == 0):
    argnums = 0 if argnums is None else argnums
    return _jacfwd(fun=func,
                   argnums=argnums,
                   holomorphic=holomorphic,
                   has_aux=has_aux,
                   return_value=return_value)
  else:
    _argnums, _ = tree_flatten(argnums)
    _argnums = tuple(a + 2 for a in _argnums)
    if argnums is None and len(grad_vars) == 0:
      raise errors.MathError('We detect no require to compute gradients because '
                             '"grad_vars" is None and "argnums" is also None. '
                             'Please provide one of them.')
    # computation
    grad_func = _cls_jacfwd(func=func,
                            grad_vars=grad_vars,
                            dyn_vars=dyn_vars,
                            argnums=(0,) + _argnums,
                            has_aux=has_aux,
                            holomorphic=holomorphic)

    call_func = _make_cls_call_func(grad_func=grad_func,
                                    grad_tree=grad_tree,
                                    grad_vars=grad_vars,
                                    dyn_vars=dyn_vars,
                                    argnums=argnums,
                                    return_value=return_value,
                                    has_aux=has_aux)

    return call_func


def hessian(func, dyn_vars=None, grad_vars=None, argnums=None, holomorphic=False, return_value=False):
  """Hessian of ``func`` as a dense array.

  Parameters
  ----------
  func : callable, function
    Function whose Hessian is to be computed.  Its arguments at positions
    specified by ``argnums`` should be arrays, scalars, or standard Python
    containers thereof. It should return arrays, scalars, or standard Python
    containers thereof.
  dyn_vars : optional, ArrayCollector, sequence of JaxArray
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
  dyn_vars, grad_vars, grad_tree = _grad_checking(func, dyn_vars, grad_vars)
  argnums = 0 if argnums is None else argnums

  if (len(dyn_vars) == 0) and (len(grad_vars) == 0) and (not return_value):
    return jax.hessian(func, argnums=argnums, holomorphic=holomorphic)
  else:
    return jacfwd(jacrev(func,
                         dyn_vars=dyn_vars,
                         grad_vars=grad_vars,
                         argnums=argnums,
                         holomorphic=holomorphic),
                  dyn_vars=dyn_vars,
                  grad_vars=grad_vars,
                  argnums=argnums,
                  holomorphic=holomorphic,
                  return_value=return_value)


def _vector_grad(func, argnums=0, return_value=False, has_aux=False):
  _check_callable(func)

  def grad_fun(*args, **kwargs):
    f = linear_util.wrap_init(func, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args, require_static_args_hashable=False)
    if has_aux:
      y, vjp_fn, aux = _vjp(f_partial, *dyn_args, has_aux=True)
    else:
      y, vjp_fn = _vjp(f_partial, *dyn_args, has_aux=False)
    leaves, tree = tree_flatten(y)
    tangents = tree_unflatten(tree, [jnp.ones_like(l) for l in leaves])
    grads = vjp_fn(tangents)
    if isinstance(argnums, int):
      grads = grads[0]
    if has_aux:
      return (grads, y, aux) if return_value else (grads, aux)
    else:
      return (grads, y) if return_value else grads

  return grad_fun


def _cls_vector_grad(func, grad_vars, dyn_vars, argnums, has_aux=False):
  # parameters
  assert isinstance(dyn_vars, (tuple, list))  # tuple/list of JaxArray
  assert isinstance(grad_vars, (tuple, list))  # tuple/list of JaxArray
  assert isinstance(argnums, (tuple, list))  # tuple/list of int

  # final functions
  if has_aux:
    @partial(_vector_grad, argnums=argnums, has_aux=True)
    def grad_func(grad_values, dyn_values, *args, **kwargs):
      for v, d in zip(dyn_vars, dyn_values): v._value = d
      for v, d in zip(grad_vars, grad_values): v._value = d
      outputs = func(*args, **kwargs)
      output = outputs[0].value if isinstance(outputs[0], JaxArray) else outputs[0]
      return output, (outputs, [v.value for v in grad_vars], [v.value for v in dyn_vars])

  else:
    @partial(_vector_grad, argnums=argnums, has_aux=True)
    def grad_func(grad_values, dyn_values, *args, **kwargs):
      for v, d in zip(dyn_vars, dyn_values): v._value = d
      for v, d in zip(grad_vars, grad_values): v._value = d
      outputs = func(*args, **kwargs)
      output = outputs.value if isinstance(outputs, JaxArray) else outputs
      return output, (outputs, [v.value for v in grad_vars], [v.value for v in dyn_vars])

  return grad_func


def vector_grad(func, dyn_vars=None, grad_vars=None, argnums=None, return_value=False, has_aux=None):
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
  func: Function whose Jacobian is to be computed.
  dyn_vars : optional, JaxArray, sequence of JaxArray, dict
    The dynamically changed variables used in ``func``.
  grad_vars : optional, JaxArray, sequence of JaxArray, dict
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
  func : callable
    The vector gradient function.
  """
  dyn_vars, grad_vars, grad_tree = _grad_checking(func, dyn_vars, grad_vars)
  has_aux = False if has_aux is None else has_aux

  if (len(dyn_vars) == 0) and (len(grad_vars) == 0):
    argnums = 0 if argnums is None else argnums
    call_func = _vector_grad(func=func,
                             argnums=argnums,
                             return_value=return_value,
                             has_aux=has_aux)

  else:
    _argnums, _ = tree_flatten(argnums)
    _argnums = tuple(a + 2 for a in _argnums)
    if argnums is None and len(grad_vars) == 0:
      raise errors.MathError('We detect no require to compute gradients because '
                             '"grad_vars" is None and "argnums" is also None. '
                             'Please provide one of them.')
    # computation
    grad_func = _cls_vector_grad(func=func,
                                 grad_vars=grad_vars,
                                 dyn_vars=dyn_vars,
                                 argnums=(0,) + _argnums,
                                 has_aux=has_aux)

    call_func = _make_cls_call_func(grad_func=grad_func,
                                    grad_tree=grad_tree,
                                    grad_vars=grad_vars,
                                    dyn_vars=dyn_vars,
                                    argnums=argnums,
                                    return_value=return_value,
                                    has_aux=has_aux)

  return call_func
