# -*- coding: utf-8 -*-


import inspect
from typing import List, Tuple

import jax

from brainpy.math.jax.ndarray import TrainVar, ndarray
from brainpy.primary.base import Primary

__all__ = [
  'grad', 'Grad',
  'value_and_grad', 'ValueAndGrad',
]


def grad(fun_or_obj, vars=None, argnums=None, has_aux=None,
         holomorphic=False, allow_int=False, reduce_axes=()):
  """Creates a function which evaluates the gradient of ``fun``.

  Parameters
  ----------
  fun_or_obj : function, Primary
    Function to be differentiated. Its arguments at positions specified by
    ``argnums`` should be arrays, scalars, or standard Python containers.
    Argument arrays in the positions specified by ``argnums`` must be of
    inexact (i.e., floating-point or complex) type. It
    should return a scalar (which includes arrays with shape ``()`` but not
    arrays with shape ``(1,)`` etc.)
  argnums : optional, integer or sequence of integers
    Specifies which positional argument(s) to differentiate with respect to (default 0).
  has_aux: optional, bool.
    Indicates whether ``fun`` returns a pair where the
    first element is considered the output of the mathematical function to be
    differentiated and the second element is auxiliary data. Default False.
  holomorphic: optional, bool.
    Indicates whether ``fun`` is promised to be
    holomorphic. If True, inputs and outputs must be complex. Default False.
  allow_int: optional, bool.
    Whether to allow differentiating with
    respect to integer valued inputs. The gradient of an integer input will
    have a trivial vector-space dtype (float0). Default False.
  reduce_axes: optional, tuple of axis names.
    If an axis is listed here, and
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

  Examples
  --------

  >>> import brainpy as bp
  >>> grad_tanh = grad(bp.math.tanh)
  >>> print(grad_tanh(0.2))
  0.961043

  >>>
  """
  # vars
  if vars is None:
    if isinstance(fun_or_obj, Primary):
      vars = fun_or_obj.vars().subset(TrainVar)

  # function
  if not callable(fun_or_obj):
    raise ValueError('Must be a callable object.')
  # gradient
  if vars is None:
    has_aux = False if has_aux is None else has_aux
    argnums = 0 if argnums is None else argnums
    return jax.grad(fun=fun_or_obj, argnums=argnums, has_aux=has_aux,
                    holomorphic=holomorphic, allow_int=allow_int,
                    reduce_axes=reduce_axes)
  else:
    has_aux = True if has_aux is None else has_aux
    if not has_aux:
      raise ValueError('"has_aux" must be True if provide "vars" and "obj"')
    return Grad(fun=fun_or_obj, vars=vars, argnums=argnums, has_aux=True,
                holomorphic=holomorphic, allow_int=allow_int,
                reduce_axes=reduce_axes)


def value_and_grad(f_or_ds, vars=None, argnums=None, has_aux=None,
                   holomorphic=False, allow_int=False, reduce_axes=()):
  """Create a function which evaluates both ``fun`` and the gradient of ``fun``.

  Parameters
  ----------
  f_or_ds : function, Primary
    Function to be differentiated. Its arguments at positions specified by
    ``argnums`` should be arrays, scalars, or standard Python containers. It
    should return a scalar (which includes arrays with shape ``()`` but not
    arrays with shape ``(1,)`` etc.)
  argnums: optional, integer or sequence of integers
    Specifies which
    positional argument(s) to differentiate with respect to (default 0).
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
  reduce_axes: optional, tuple of axis names
    If an axis is listed here, and
    ``fun`` implicitly broadcasts a value over that axis, the backward pass
    will perform a ``psum`` of the corresponding gradient. Otherwise, the
    gradient will be per-example over named axes. For example, if ``'batch'``
    is a named batch axis, ``value_and_grad(f, reduce_axes=('batch',))`` will
    create a function that computes the total gradient while
    ``value_and_grad(f)`` will create one that computes the per-example
    gradient.

  Returns
  -------
    A function with the same arguments as ``fun`` that evaluates both ``fun``
    and the gradient of ``fun`` and returns them as a pair (a two-element
    tuple). If ``argnums`` is an integer then the gradient has the same shape
    and type as the positional argument indicated by that integer. If argnums is
    a sequence of integers, the gradient is a tuple of values with the same
    shapes and types as the corresponding arguments.
  """
  # vars
  if vars is None:
    if isinstance(f_or_ds, Primary):
      vars = f_or_ds.vars()
  # function
  if not callable(f_or_ds):
    raise ValueError('Must be a callable object.')
  # jit compilation
  if vars is None:
    argnums = 0 if argnums is None else argnums
    return jax.value_and_grad(fun=f_or_ds, argnums=argnums, has_aux=has_aux,
                              holomorphic=holomorphic, allow_int=allow_int,
                              reduce_axes=reduce_axes)
  else:
    return ValueAndGrad(fun=f_or_ds, vars=vars)


class Gradient(Primary):
  def __init__(self, func, raw, vars, argnums=None, has_aux=True, holomorphic=False,
               allow_int=False, reduce_axes=()):
    super(Gradient, self).__init__()

    if argnums is None:
      argnums = (0,)
    elif isinstance(argnums, int):
      argnums = (0, argnums + 1)
    else:
      argnums = (0,) + tuple(a + 1 for a in argnums)
    self.argnums = argnums

    self._raw = raw
    self._vars = vars
    self._call = jax.grad(fun=func, argnums=argnums, has_aux=has_aux,
                          holomorphic=holomorphic, allow_int=allow_int,
                          reduce_axes=reduce_axes)
    signature = inspect.signature(raw)
    self.__signature__ = signature.replace(return_annotation=Tuple[List[ndarray], signature.return_annotation])

  def vars(self, method='absolute'):
    if isinstance(self._raw, Primary):
      return super(Gradient, self).vars(method=method)
    else:
      return self._vars


class Grad(Gradient):
  def __init__(self, fun, vars, argnums=None, has_aux=True, holomorphic=False,
               allow_int=False, reduce_axes=()):
    def func(train_vars, *args, **kwargs):
      vars.assign(train_vars)
      outputs = fun(*args, **kwargs)
      outputs2 = outputs.value if isinstance(outputs, ndarray) else outputs
      return outputs2, vars.dict()

    super(Grad, self).__init__(func=func, raw=fun, vars=vars, argnums=argnums,
                               has_aux=has_aux, holomorphic=holomorphic,
                               allow_int=allow_int, reduce_axes=reduce_axes)

  def __call__(self, *args, **kwargs):
    g, changes = self._call(self._vars.dict(), *args, **kwargs)
    self._vars.assign(changes)
    return g[0] if len(self.argnums) == 1 else g[1:] + g[:1]


class ValueAndGrad(Gradient):
  def __init__(self, fun, vars, argnums=None, has_aux=True, holomorphic=False,
               allow_int=False, reduce_axes=()):
    def func(train_vars, *args, **kwargs):
      vars.assign(train_vars)
      outputs = fun(*args, **kwargs)
      outputs2 = outputs.value if isinstance(outputs, ndarray) else outputs
      return outputs2, (outputs, vars.dict())

    super(ValueAndGrad, self).__init__(func=func, raw=fun, vars=vars, argnums=argnums,
                                       has_aux=has_aux, holomorphic=holomorphic,
                                       allow_int=allow_int, reduce_axes=reduce_axes)

  def __call__(self, *args, **kwargs):
    g, (outputs, changes) = self._call(self._vars.dict(), *args, **kwargs)
    self._vars.assign(changes)
    grads = g[0] if len(self.argnums) == 1 else g[1:] + g[:1]
    return outputs, grads
