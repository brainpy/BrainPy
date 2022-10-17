# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.base import Base
from jax import vmap
import jax.numpy as jnp

__all__ = [
  # base class
  'OnlineAlgorithm',

  # online learning algorithms
  'RLS',
  'LMS',

  # generic methods
  'get_supported_online_methods',
  'register_online_method',
]

name2func = dict()


class OnlineAlgorithm(Base):
  """Base class for online training algorithm."""

  def __init__(self, name=None):
    super(OnlineAlgorithm, self).__init__(name=name)

  def __call__(self, identifier, target, input, output):
    """The training procedure.

    Parameters
    ----------
    identifier: str
      The variable name.
    target: JaxArray, ndarray
      The 2d target data with the shape of `(num_batch, num_output)`.
    input: JaxArray, ndarray
      The 2d input data with the shape of `(num_batch, num_input)`.
    output: JaxArray, ndarray
      The 2d output data with the shape of `(num_batch, num_output)`.

    Returns
    -------
    weight: JaxArray
      The weights after fit.
    """
    return self.call(identifier, target, input, output)

  def initialize(self, identifier, *args, **kwargs):
    pass

  def call(self, identifier, target, input, output):
    """The training procedure.

    Parameters
    ----------
    identifier: str
      The variable name.
    target: JaxArray, ndarray
      The 2d target data with the shape of `(num_batch, num_output)`.
    input: JaxArray, ndarray
      The 2d input data with the shape of `(num_batch, num_input)`.
    output: JaxArray, ndarray
      The 2d output data with the shape of `(num_batch, num_output)`.

    Returns
    -------
    weight: JaxArray
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

  def initialize(self, identifier, feature_in, feature_out=None):
    identifier = identifier + self.postfix
    self.implicit_vars[identifier] = bm.Variable(bm.eye(feature_in) * self.alpha)

  def call(self, identifier, target, input, output):
    identifier = identifier + self.postfix
    P = self.implicit_vars[identifier]
    # update the inverse correlation matrix
    k = bm.dot(P, input.T)  # (num_input, num_batch)
    hPh = bm.dot(input, k)  # (num_batch, num_batch)
    c = bm.sum(1.0 / (1.0 + hPh))  # ()
    P -= c * bm.dot(k, k.T)  # (num_input, num_input)
    # update weights
    e = output - target  # (num_batch, num_output)
    dw = -c * bm.dot(k, e)  # (num_input, num_output)
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

  def call(self, identifier, target, input, output):
    assert target.shape[0] == input.shape[0] == output.shape[0], 'Batch size should be consistent.'
    error = bm.as_jax(output - target)
    input = bm.as_jax(input)
    return -self.alpha * bm.sum(vmap(jnp.outer)(input, error), axis=0)


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
