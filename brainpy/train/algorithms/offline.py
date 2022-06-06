# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.base import Base

__all__ = [
  # base class for offline training algorithm
  'OfflineAlgorithm',

  # training methods
  'RidgeRegression',
  'LinearRegression',

  # general supports
  'get_supported_offline_methods',
  'register_offline_method',
]

name2func = dict()


class OfflineAlgorithm(Base):
  """Base class for offline training algorithm."""

  def __init__(self, name=None):
    super(OfflineAlgorithm, self).__init__(name=name)

  def __call__(self, targets, inputs, outputs):
    """The training procedure.

    Parameters
    ----------
    inputs: JaxArray, jax.numpy.ndarray, numpy.ndarray
      The 3d input data with the shape of `(num_batch, num_time, num_input)`,
      or, the 2d input data with the shape of `(num_time, num_input)`.

    targets: JaxArray, jax.numpy.ndarray, numpy.ndarray
      The 3d target data with the shape of `(num_batch, num_time, num_output)`,
      or the 2d target data with the shape of `(num_time, num_output)`.

    outputs: JaxArray, jax.numpy.ndarray, numpy.ndarray
      The 3d output data with the shape of `(num_batch, num_time, num_output)`,
      or the 2d output data with the shape of `(num_time, num_output)`.

    Returns
    -------
    weight: JaxArray
      The weights after fit.
    """
    raise NotImplementedError('Must implement the __call__ function by the subclass itself.')

  def __repr__(self):
    return self.__class__.__name__

  def initialize(self, identifier, *args, **kwargs):
    raise NotImplementedError('Must implement the initialize() '
                              'function by the subclass itself.')


class RidgeRegression(OfflineAlgorithm):
  """Training algorithm of ridge regression.

  Parameters
  ----------
  beta: float
    The regularization coefficient.
  """

  def __init__(self, beta=1e-7, name=None):
    super(RidgeRegression, self).__init__(name=name)
    self.beta = beta

  def __call__(self, targets, inputs, outputs=None):
    # checking
    inputs = bm.asarray(inputs).reshape((-1, inputs.shape[2]))
    targets = bm.asarray(targets).reshape((-1, targets.shape[2]))
    # solving
    temp = inputs.T @ inputs
    if self.beta > 0.:
      temp += self.beta * bm.eye(inputs.shape[-1])
    weights = bm.linalg.pinv(temp) @ (inputs.T @ targets)
    return weights

  def __repr__(self):
    return f'{self.__class__.__name__}(beta={self.beta})'

  def initialize(self, identifier, *args, **kwargs):
    pass


name2func['ridge'] = RidgeRegression


class LinearRegression(OfflineAlgorithm):
  """Training algorithm of least-square regression."""

  def __init__(self, name=None):
    super(LinearRegression, self).__init__(name=name)

  def __call__(self, targets, inputs, outputs=None):
    inputs = bm.asarray(inputs).reshape((-1, inputs.shape[2]))
    targets = bm.asarray(targets).reshape((-1, targets.shape[2]))
    weights = bm.linalg.lstsq(inputs, targets)
    return weights[0]

  def initialize(self, identifier, *args, **kwargs):
    pass


name2func['linear'] = LinearRegression
name2func['lstsq'] = LinearRegression


class LassoRegression(OfflineAlgorithm):
  """Lasso regression method for offline training.

  Parameters
  ----------
  alpha: float
    Constant that multiplies the L1 term. Defaults to 1.0.
    `alpha = 0` is equivalent to an ordinary least square.
  max_iter: int
    The maximum number of iterations.
  """

  def __init__(self, alpha=1.0, max_iter=1000, name=None):
    super(LassoRegression, self).__init__(name=name)
    self.alpha = alpha
    self.max_iter = max_iter

  def __call__(self, *args, **kwargs):
    pass

  def initialize(self, identifier, *args, **kwargs):
    pass


# name2func['lasso'] = LassoRegression


def elastic_net_regression(x, y, train_pars):
  pass


# name2func['elastic_net'] = elastic_net_regression


def logistic_regression(x, y, train_pars):
  pass


# name2func['logistic'] = logistic_regression


def polynomial_regression(x, y, train_pars):
  pass


# name2func['polynomial'] = polynomial_regression


def stepwise_regression(x, y, train_pars):
  pass


# name2func['stepwise'] = stepwise_regression


def get_supported_offline_methods():
  """Get all supported offline training methods."""
  return tuple(name2func.keys())


def register_offline_method(name, method):
  """Register a new offline learning method.

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
    raise ValueError(f'All offline methods are: {get_supported_offline_methods()}.\n'
                     f'But we got {name}.')
  return name2func[name]
