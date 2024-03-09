from typing import Callable, Optional, Sequence

import numpy as np

from .base import Optimizer

__all__ = ['SkBayesianOptimizer']


class SkBayesianOptimizer(Optimizer):
  """
  SkoptOptimizer instance creates all the tools necessary for the user
  to use it with scikit-optimize library.

  Parameters
  ----------
  parameter_names: list[str]
      Parameters to be used as instruments.
  bounds : list
      List with appropiate bounds for each parameter.
  method : `str`, optional
      The optimization method. Possibilities: "GP", "RF", "ET", "GBRT" or
      sklearn regressor, default="GP"
  n_calls: int
      Number of calls to ``func``. Defaults to 100.
  n_jobs: int
      The number of jobs to run in parallel for ``fit``. If -1, then the
      number of jobs is set to the number of cores.

  """

  def __init__(
      self,
      loss_fun: Callable,
      n_sample: int,
      bounds: Optional[Sequence] = None,
      method: str = 'GP',
      **kwds
  ):
    super().__init__()

    try:
      from sklearn.base import RegressorMixin  # noqa
    except (ImportError, ModuleNotFoundError):
      raise ImportError("scikit-learn must be installed to use this class")

    # loss function
    assert callable(loss_fun), "'loss_fun' must be a callable function"
    self.loss_fun = loss_fun

    # method
    if not (method.upper() in ["GP", "RF", "ET", "GBRT"] or isinstance(method, RegressorMixin)):
      raise AssertionError(f"Provided method: {method} is not an skopt optimization or a regressor")
    self.method = method

    # population size
    assert n_sample > 0, "'n_sample' must be a positive integer"
    self.n_sample = n_sample

    # bounds
    if bounds is None:
      bounds = ()
    self.bounds = bounds

    # others
    self.kwds = kwds

  def initialize(self):
    try:
      from skopt.optimizer import Optimizer  # noqa
      from skopt.space import Real  # noqa
    except (ImportError, ModuleNotFoundError):
      raise ImportError("scikit-optimize must be installed to use this class")
    self.tested_parameters = []
    self.errors = []
    instruments = []
    for bound in self.bounds:
      instrumentation = Real(*np.asarray(bound), transform='normalize')
      instruments.append(instrumentation)
    self.optim = Optimizer(dimensions=instruments, base_estimator=self.method, **self.kwds)

  def one_trial(self, choice_best: bool = False):
    # draw parameters
    parameters = self.optim.ask(n_points=self.n_sample)
    self.tested_parameters.extend(parameters)

    # errors
    errors = self.loss_fun(*np.asarray(parameters).T)
    errors = np.asarray(errors).tolist()
    self.errors.extend(errors)

    # tell
    self.optim.tell(parameters, errors)

    if choice_best:
      xi = self.optim.Xi
      yii = np.array(self.optim.yi)
      return xi[yii.argmin()]
