from brainpy.dyn.base import DynamicalSystem
from typing import Optional
from brainpy.modes import Mode
from typing import Callable


class Activation(DynamicalSystem):
  r"""Applies a activation to the inputs

  Parameters:
  ----------
  activate_fun: Callable
    The function of Activation
  name: str, Optional
    The name of the object
  mode: Mode
    Enable training this node or not. (default True).
  """

  def __init__(self,
               activate_fun: Callable,
               name: Optional[str] = None,
               mode: Optional[Mode] = None,
               **kwargs,
      ):
    super().__init__(name, mode)
    self.activate_fun = activate_fun
    self.kwargs = kwargs

  def update(self, sha, x):
    return self.activate_fun(x, **self.kwargs)

  def reset_state(self, batch_size=None):
    pass
