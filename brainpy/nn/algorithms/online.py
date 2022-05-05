# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.base import Base

__all__ = [
  # base class
  'OnlineAlgorithm',

  # online learning algorithms
  'ForceLearning',
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

  def __call__(self, name, target, input, output):
    """The training procedure.

    Parameters
    ----------
    name: str
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
    return self.call(name, target, input, output)

  def initialize(self, name, *args, **kwargs):
    raise NotImplementedError('Must implement the initialize() '
                              'function by the subclass itself.')

  def call(self, name, target, input, output):
    """The training procedure.

    Parameters
    ----------
    name: str
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
  """The recursive least squares (RLS)."""

  postfix = '.rls.P'

  def __init__(self, alpha=0.1, name=None):
    super(RLS, self).__init__(name=name)
    self.alpha = alpha

  def initialize(self, name, feature_in, feature_out=None):
    name = name + self.postfix
    self.implicit_vars[name] = bm.Variable(bm.eye(feature_in) * self.alpha)

  def call(self, name, target, input, output):
    name = name + self.postfix
    P = self.implicit_vars[name]
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


class ForceLearning(RLS):
  postfix = '.force.P'


name2func['force'] = ForceLearning


class LMS(OnlineAlgorithm):
  """The least mean squares (LMS). """

  def __init__(self, alpha=0.1, name=None):
    super(LMS, self).__init__(name=name)
    self.alpha = alpha

  def initialize(self, name, *args, **kwargs):
    pass

  def call(self, name, target, input, output):
    return -self.alpha * bm.dot(output - target, output)


name2func['lms'] = LMS


def get_supported_online_methods():
  """Get all supported online training methods."""
  return tuple(name2func.keys())


def register_online_method(name, method):
  """Register a new oneline learning method.

  Parameters
  ----------
  name: str
    The method name.
  method: callable
    The function method.
  """
  if name in name2func:
    raise ValueError(f'"{name}" has been registered in offline training methods.')
  if not callable(method):
    raise ValueError(f'"method" must be an instance of callable '
                     f'function, but we got {type(method)}')
  name2func[name] = method


def get(name):
  """Get the training function according to the training method name."""
  if name not in name2func:
    raise ValueError(f'All online methods are: {get_supported_online_methods()}.\n'
                     f'But we got {name}.')
  return name2func[name]
