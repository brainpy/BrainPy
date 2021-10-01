# -*- coding: utf-8 -*-


import jax

from brainpy import errors
from brainpy.base.base import Base
from brainpy.base.collector import ArrayCollector
from brainpy.math.jax.jaxarray import JaxArray, TrainVar

__all__ = [
  'grad', 'value_and_grad',
  'Grad', 'ValueAndGrad',
]


def grad(func, vars=None, argnums=None, has_aux=None,
         holomorphic=False, allow_int=False, reduce_axes=()):
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
  """
  # vars
  if vars is None:
    if isinstance(func, Base):
      vars = func.vars()

  # function
  if not callable(func):
    raise ValueError('Must be a callable object.')

  # gradient
  if vars is None:
    has_aux = False if has_aux is None else has_aux
    argnums = 0 if argnums is None else argnums
    return jax.grad(fun=func,
                    argnums=argnums,
                    has_aux=has_aux,
                    holomorphic=holomorphic,
                    allow_int=allow_int,
                    reduce_axes=reduce_axes)
  else:
    return Grad(fun=func,
                vars=vars,
                argnums=argnums,
                has_aux=has_aux,
                holomorphic=holomorphic,
                allow_int=allow_int,
                reduce_axes=reduce_axes)


def value_and_grad(func, vars=None, argnums=None, has_aux=None,
                   holomorphic=False, allow_int=False, reduce_axes=()):
  """Automatic Gradient Computation in JAX backend.

  Create a function which evaluates both ``fun`` and the gradient of ``fun``.

  Parameters
  ----------
  func : function, Base
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
    if isinstance(func, Base):
      vars = func.vars()

  # function
  if not callable(func):
    raise ValueError('Must be a callable object.')

  # jit compilation
  if vars is None:
    has_aux = False if has_aux is None else has_aux
    argnums = 0 if argnums is None else argnums
    return jax.value_and_grad(fun=func,
                              argnums=argnums,
                              has_aux=has_aux,
                              holomorphic=holomorphic,
                              allow_int=allow_int,
                              reduce_axes=reduce_axes)
  else:
    return ValueAndGrad(fun=func,
                        vars=vars,
                        argnums=argnums,
                        has_aux=has_aux,
                        holomorphic=holomorphic,
                        allow_int=allow_int,
                        reduce_axes=reduce_axes)


class Gradient(Base):
  """Base Class to Compute Gradients."""

  def __init__(self, raw, vars=None, argnums=None, has_aux=None,
               holomorphic=False, allow_int=False, reduce_axes=()):
    super(Gradient, self).__init__()

    # 'raw'
    self.raw = raw

    # trainable variables and dynamical variables
    if vars is None:
      if not isinstance(raw, Base):
        raise errors.BrainPyError(f'When "vars" is not provided, "raw" must be a {Base} object.')
      vars = raw.vars().unique()
    _train_vars = ArrayCollector()
    _dyn_vars = ArrayCollector()
    for key, var in vars.items():
      if isinstance(var, TrainVar):
        _train_vars[key] = var
      else:
        _dyn_vars[key] = var
    self._train_vars = _train_vars.unique()
    self._dyn_vars = _dyn_vars.unique()

    # 'argnums'
    if argnums is None:
      argnums = (0,)
    elif isinstance(argnums, int):
      argnums = (0, argnums + 2)
    else:
      argnums = (0,) + tuple(a + 2 for a in argnums)
    self.argnums = argnums

    # others
    self.has_aux = False if has_aux is None else True
    self.holomorphic = holomorphic
    self.allow_int = allow_int
    self.reduce_axes = reduce_axes

    # signature
    # signature = inspect.signature(raw)
    # self.__signature__ = signature.replace(return_annotation=Tuple[List[JaxArray], signature.return_annotation])

    # final functions
    if self.has_aux:
      # Users should return the auxiliary data like:
      # ------------
      # >>> # 1. example of return one data
      # >>> return scalar_loss, data
      # >>> # 2. example of return multiple data
      # >>> return scalar_loss, (data1, data2, ...)
      def func(train_vars, dyn_vars, *args, **kwargs):
        self._train_vars.assign(train_vars)
        self._dyn_vars.assign(dyn_vars)
        # outputs: [0] is the value for gradient,
        #          [1] is other values for return
        outputs = self.raw(*args, **kwargs)
        output = outputs[0].value if isinstance(outputs[0], JaxArray) else outputs[0]
        return output, (outputs, self._train_vars.dict(), self._dyn_vars.dict())
    else:
      # Users should return the scalar value like this:
      # ------------
      # >>> return scalar_loss
      def func(train_vars, dyn_vars, *args, **kwargs):
        self._train_vars.assign(train_vars)
        self._dyn_vars.assign(dyn_vars)
        output = self.raw(*args, **kwargs)
        output2 = output.value if isinstance(output, JaxArray) else output
        return output2, (output, self._train_vars.dict(), self._dyn_vars.dict())

    # function for gradient
    self._call = jax.grad(fun=func,
                          argnums=self.argnums,
                          has_aux=True,
                          holomorphic=self.holomorphic,
                          allow_int=self.allow_int,
                          reduce_axes=self.reduce_axes)


class Grad(Gradient):
  """Compute the gradients of trainable variables for the given object.

  Examples
  --------

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
  >>> test_grad = Grad(test, test.vars(), argnums=0, has_aux=True)
  >>> grads, outputs = test_grad(10.)
  >>> grads
  (DeviceArray(1., dtype=float32),
   {'Test3.a': DeviceArray([2.], dtype=float32), 'Test3.b': DeviceArray([2.], dtype=float32)})
  >>> outputs
  (JaxArray(DeviceArray([1.], dtype=float32)),
   JaxArray(DeviceArray([2.], dtype=float32)))
  """

  def __init__(self, fun, vars=None, argnums=None, has_aux=None,
               holomorphic=False, allow_int=False, reduce_axes=()):
    super(Grad, self).__init__(raw=fun,
                               vars=vars,
                               argnums=argnums,
                               has_aux=has_aux,
                               holomorphic=holomorphic,
                               allow_int=allow_int,
                               reduce_axes=reduce_axes)

  def __call__(self, *args, **kwargs):
    grads, (outputs, train_vars, dyn_vars) = self._call(self._train_vars.dict(),
                                                        self._dyn_vars.dict(),
                                                        *args,
                                                        **kwargs)
    self._train_vars.assign(train_vars)
    self._dyn_vars.assign(dyn_vars)
    grads = grads[0] if len(self.argnums) == 1 else grads[1:] + grads[:1]
    return (grads, outputs[1]) if self.has_aux else grads


class ValueAndGrad(Gradient):
  """Compute the results and the gradients of trainable variables for the given object.

  Examples
  --------

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

  def __init__(self, fun, vars=None, argnums=None, has_aux=None, holomorphic=False,
               allow_int=False, reduce_axes=()):
    super(ValueAndGrad, self).__init__(raw=fun,
                                       vars=vars,
                                       argnums=argnums,
                                       has_aux=has_aux,
                                       holomorphic=holomorphic,
                                       allow_int=allow_int,
                                       reduce_axes=reduce_axes)

  def __call__(self, *args, **kwargs):
    grads, (outputs, train_vars, dyn_vars) = self._call(self._train_vars.dict(),
                                                        self._dyn_vars.dict(),
                                                        *args,
                                                        **kwargs)
    self._train_vars.assign(train_vars)
    self._dyn_vars.assign(dyn_vars)
    grads = grads[0] if len(self.argnums) == 1 else grads[1:] + grads[:1]
    return outputs, grads
