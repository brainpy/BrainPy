# -*- coding: utf-8 -*-

from typing import Callable
from typing import Optional

import brainpy.math as bm
from brainpy import check
from .base import Layer

__all__ = [
  'Activation',
  'Flatten',
  'FunAsLayer',
]


class Activation(Layer):
  r"""Applies an activation function to the inputs

  Parameters:
  ----------
  activate_fun: Callable, function
    The function of Activation
  name: str, Optional
    The name of the object
  mode: Mode
    Enable training this node or not. (default True).
  """

  def __init__(
      self,
      activate_fun: Callable,
      name: Optional[str] = None,
      mode: bm.Mode = None,
      **kwargs,
  ):
    super().__init__(name, mode)
    self.activate_fun = activate_fun
    self.kwargs = kwargs

  def update(self, *args):
    x = args[0] if len(args) == 1 else args[1]
    return self.activate_fun(x, **self.kwargs)


class Flatten(Layer):
  r"""Flattens a contiguous range of dims into 2D or 1D.

  Parameters:
  ----------
  name: str, Optional
    The name of the object
  mode: Mode
    Enable training this node or not. (default True)
  """

  def __init__(
      self,
      name: Optional[str] = None,
      mode: bm.Mode = None,
  ):
    super().__init__(name, mode)
    check.is_subclass(self.mode, (bm.NonBatchingMode, bm.BatchingMode, bm.TrainingMode), self.name)

  def update(self, *args):
    x = args[0] if len(args) == 1 else args[1]
    if isinstance(self.mode, bm.BatchingMode):
      return x.reshape((x.shape[0], -1))
    else:
      return x.flatten()


class FunAsLayer(Layer):
  def __init__(
      self,
      fun: Callable,
      name: Optional[str] = None,
      mode: bm.Mode = None,
      has_shared: bool = False,
      **kwargs,
  ):
    super().__init__(name, mode)
    self._fun = fun
    self.kwargs = kwargs
    self.has_shared = has_shared

  def update(self, *args):
    x = args[0] if len(args) == 1 else args[1]
    if self.has_shared:
      assert len(args) > 1
      s = args[0]
      return self._fun(s, x, **self.kwargs)
    else:
      return self._fun(x, **self.kwargs)
