import abc

from tqdm.auto import tqdm

__all__ = ['Optimizer']


class Optimizer(metaclass=abc.ABCMeta):
  """
  Optimizer class created as a base for optimization initialization and
  performance with different libraries. To be used with modelfitting
  Fitter.
  """

  @abc.abstractmethod
  def initialize(self, *args, **kwargs):
    """
    Initialize the instrumentation for the optimization, based on
    parameters, creates bounds for variables and attaches them to the
    optimizer
    """
    pass

  @abc.abstractmethod
  def one_trial(self, *args, **kwargs):
    """
    Returns the requested number of samples of parameter sets
    """
    pass

  def minimize(self, n_iter):
    results = []
    for i in tqdm(range(n_iter)):
      r = self.one_trial(choice_best=i + 1 == n_iter)
      results.append(r)
    return results[-1]
