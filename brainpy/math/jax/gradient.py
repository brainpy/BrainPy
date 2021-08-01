# -*- coding: utf-8 -*-


import jax

from brainpy.simulation.brainobjects.base import DynamicSystem

__all__ = [
  'grad', 'Grad',
  'value_and_grad', 'ValueAndGrad',
]


def grad(f_or_ds, vars=None, argnums=None, has_aux=False,
         holomorphic=False, allow_int=False, reduce_axes=()):
  """Creates a function which evaluates the gradient of ``fun``.

  Parameters
  ----------
  f_or_ds : function, DynamicSystem
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

  >>> grad_tanh = grad(jax.numpy.tanh)
  >>> print(grad_tanh(0.2))
  0.961043
  """
  # vars
  if vars is None:
    if isinstance(f_or_ds, DynamicSystem):
      vars = f_or_ds.vars()
  # argnums
  if argnums is None:
    if vars is None:
      argnums = (0,)
    else:
      argnums = ()
  # function
  if not callable(f_or_ds):
    raise ValueError
  # jit compilation
  if vars is None:
    return jax.grad(fun=f_or_ds, argnums=argnums, has_aux=has_aux,
                    holomorphic=holomorphic, allow_int=allow_int,
                    reduce_axes=reduce_axes)
  else:
    return Grad(fun=f_or_ds, vars=vars)


class Grad(DynamicSystem):
  def __init__(self, fun, vars):
    self._raw = fun
    self._vars = vars
    super(Grad, self).__init__()

    def func(inputs_and_train_tensors, state_tensors, list_args, kwargs):
      inputs = inputs_and_train_tensors[:len(self.input_argnums)]
      train_tensors = inputs_and_train_tensors[len(self.input_argnums):]
      self.vc.subset(TrainVar).assign(train_tensors)
      self.vc.subset(BaseState).assign(state_tensors)
      for i, arg in zip(self.input_argnums, inputs):
        list_args[i] = arg
      outputs = f(*list_args, **kwargs)
      if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
      return outputs[0], (outputs, variables.tensors())

  def vars(self, method='absolute'):
    if isinstance(self._raw, DynamicSystem):
      return super(Grad, self).vars(method=method)
    else:
      return self._vars

  def __call__(self, *args, **kwargs):
    return


def value_and_grad(f_or_ds, vars=None, argnums=0, has_aux=False,
                   holomorphic=False, allow_int=False, reduce_axes=()):
  """Create a function which evaluates both ``fun`` and the gradient of ``fun``.

  Parameters
  ----------
  f_or_ds : function, DynamicSystem
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
    if isinstance(f_or_ds, DynamicSystem):
      vars = f_or_ds.vars()
  # argnums
  if argnums is None:
    if vars is None:
      argnums = (0,)
    else:
      argnums = ()
  # function
  if not callable(f_or_ds):
    raise ValueError
  # jit compilation
  if vars is None:
    return jax.value_and_grad(fun=f_or_ds, argnums=argnums, has_aux=has_aux,
                              holomorphic=holomorphic, allow_int=allow_int,
                              reduce_axes=reduce_axes)
  else:
    return ValueAndGrad(fun=f_or_ds, vars=vars)


class ValueAndGrad(DynamicSystem):
  def __init__(self, fun, vars):
    self._raw = fun
    self._vars = vars
    super(ValueAndGrad, self).__init__()

  def vars(self, method='absolute'):
    return self._vars

  def __call__(self, *args, **kwargs):
    return
