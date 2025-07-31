# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
from jax import vmap

import brainpy.math as bm
from brainpy._src.math.object_transform.base import BrainPyObject

__all__ = [
  # brainpy_object class
  'OnlineAlgorithm',

  # online learning algorithms
  'RLS',
  'LMS',

  # generic methods
  'get_supported_online_methods',
  'register_online_method',
]

name2func = dict()


class OnlineAlgorithm(BrainPyObject):
  """Base class for online training algorithm."""

  def __init__(self, name=None):
    super(OnlineAlgorithm, self).__init__(name=name)

  def __call__(self, *args, **kwargs):
    """The training procedure.

    Parameters
    ----------
    identifier: str
      The variable name.
    target: ArrayType
      The 2d target data with the shape of `(num_batch, num_output)`.
    input: ArrayType
      The 2d input data with the shape of `(num_batch, num_input)`.
    output: ArrayType
      The 2d output data with the shape of `(num_batch, num_output)`.

    Returns
    -------
    weight: ArrayType
      The weights after fit.
    """
    return self.call(*args, **kwargs)

  def register_target(self, *args, **kwargs):
    pass

  def call(self, target, input, output, identifier: str=''):
    """The training procedure.

    Parameters
    ----------
    identifier: str
      The variable name.
    target: ArrayType
      The 2d target data with the shape of `(num_batch, num_output)`.
    input: ArrayType
      The 2d input data with the shape of `(num_batch, num_input)`.
    output: ArrayType
      The 2d output data with the shape of `(num_batch, num_output)`.

    Returns
    -------
    weight: ArrayType
      The weights after fit.
    """
    raise NotImplementedError('Must implement the call() function by the subclass itself.')

  def __repr__(self):
    return self.__class__.__name__


class RLS(OnlineAlgorithm):
  """The recursive least squares (RLS) algorithm.

  RLS is an adaptive filter algorithm that recursively finds the
  coefficients that minimize a weighted linear least squares cost
  function relating to the input signals. This approach is in
  contrast to other algorithms such as the least mean squares
  (LMS) that aim to reduce the mean square error.

  See Also
  --------
  LMS, ForceLearning

  Parameters
  ----------
  alpha: float
    The learning rate.
  name: str
    The algorithm name.

  """

  postfix = '.rls.P'

  def __init__(self, alpha=0.1, name=None):
    super(RLS, self).__init__(name=name)
    self.alpha = alpha

  def register_target(
      self,
      feature_in: int,
      identifier: str = '',
  ):
    identifier = identifier + self.postfix
    self.implicit_vars[identifier] = bm.Variable(jnp.eye(feature_in) * self.alpha)

  def call(
      self,
      target: jax.Array,
      input: jax.Array,
      output: jax.Array,
      identifier: str = '',
  ):
    identifier = identifier + self.postfix
    P = self.implicit_vars[identifier]
    input = bm.as_jax(input)
    output = bm.as_jax(output)
    target = bm.as_jax(target)
    if input.ndim == 1: input = jnp.expand_dims(input, 0)
    if target.ndim == 1: target = jnp.expand_dims(target, 0)
    if output.ndim == 1: output = jnp.expand_dims(output, 0)
    assert input.ndim == 2, f'should be a 2D array with shape of (batch, feature). Got {input.shape}'
    assert target.ndim == 2, f'should be a 2D array with shape of (batch, feature). Got {target.shape}'
    assert output.ndim == 2, f'should be a 2D array with shape of (batch, feature). Got {output.shape}'
    k = jnp.dot(P.value, input.T)  # (num_input, num_batch)
    hPh = jnp.dot(input, k)  # (num_batch, num_batch)
    c = jnp.sum(1.0 / (1.0 + hPh))  # ()
    P -= c * jnp.dot(k, k.T)  # (num_input, num_input)
    e = output - target  # (num_batch, num_output)
    dw = -c * jnp.dot(k, e)  # (num_input, num_output)
    return dw


name2func['rls'] = RLS


class LMS(OnlineAlgorithm):
  """The least mean squares (LMS).

  LMS algorithms are a class of adaptive filter used to mimic a desired filter
  by finding the filter coefficients that relate to producing the least mean
  square of the error signal (difference between the desired and the actual signal).
  It is a stochastic gradient descent method in that the filter is only adapted
  based on the error at the current time. It was invented in 1960 by
  Stanford University professor Bernard Widrow and his first Ph.D. student, Ted Hoff.

  Parameters
  ----------
  alpha: float
    The learning rate.
  name: str
    The target name.
  """

  def __init__(self, alpha=0.1, name=None):
    super(LMS, self).__init__(name=name)
    self.alpha = alpha

  def call(self, target, input, output, identifier: str=''):
    if input.ndim == 1: input = jnp.expand_dims(input, 0)
    if target.ndim == 1: target = jnp.expand_dims(target, 0)
    if output.ndim == 1: output = jnp.expand_dims(output, 0)
    assert input.ndim == 2, f'should be a 2D array with shape of (batch, feature). Got {input.shape}'
    assert target.ndim == 2, f'should be a 2D array with shape of (batch, feature). Got {target.shape}'
    assert output.ndim == 2, f'should be a 2D array with shape of (batch, feature). Got {output.shape}'
    error = bm.as_jax(output - target)
    input = bm.as_jax(input)
    return -self.alpha * jnp.sum(vmap(jnp.outer)(input, error), axis=0)


name2func['lms'] = LMS


def get_supported_online_methods():
  """Get all supported online training methods."""
  return tuple(name2func.keys())


def register_online_method(name: str, method: OnlineAlgorithm):
  """Register a new oneline learning method.

  Parameters
  ----------
  name: str
    The method name.
  method: callable
    The function method.
  """
  if name in name2func:
    raise ValueError(f'"{name}" has been registered in online training methods. Please change another name.')
  if not isinstance(method, OnlineAlgorithm):
    raise ValueError(f'"method" must be an instance of {OnlineAlgorithm.__name__}, but we got {type(method)}')
  name2func[name] = method


def get(name: str):
  """Get the training function according to the training method name."""
  if name not in name2func:
    raise ValueError(f'All online methods are: {get_supported_online_methods()}.\n'
                     f'But we got {name}.')
  return name2func[name]
