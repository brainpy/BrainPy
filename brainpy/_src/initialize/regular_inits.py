# -*- coding: utf-8 -*-

from brainpy import math as bm, tools
from .base import _InterLayerInitializer

__all__ = [
  'ZeroInit',
  'Constant',
  'OneInit',
  'Identity',
]


class ZeroInit(_InterLayerInitializer):
  """Zero initializer.

  Initialize the weights with zeros.
  """

  def __call__(self, shape, dtype=None):
    shape = [tools.size2num(d) for d in shape]
    return bm.zeros(shape, dtype=dtype)

  def __repr__(self):
    return self.__class__.__name__


class Constant(_InterLayerInitializer):
  """Constant initializer.

  Initialize the weights with the given values.

  Parameters
  ----------
  value : float, int, bm.ndarray
    The value to specify.
  """

  def __init__(self, value=1.):
    super(Constant, self).__init__()
    self.value = value

  def __call__(self, shape, dtype=None):
    shape = [tools.size2num(d) for d in shape]
    return bm.ones(shape, dtype=dtype) * self.value

  def __repr__(self):
    return f'{self.__class__.__name__}(value={self.value})'


class OneInit(Constant):
  """One initializer.
  """
  pass


class Identity(_InterLayerInitializer):
  """Returns the identity matrix.

  This initializer was proposed in (Le, et al., 2015) [1]_.

  Parameters
  ----------
  value : float
    The optional scaling factor.

  Returns
  -------
  shape: tuple of int
    The weight shape/size.

  References
  ----------
  .. [1] Le, Quoc V., Navdeep Jaitly, and Geoffrey E. Hinton. "A simple way to
         initialize recurrent networks of rectified linear units." arXiv preprint
         arXiv:1504.00941 (2015).
  """

  def __init__(self, value=1.):
    super(Identity, self).__init__()
    self.value = value

  def __call__(self, shape, dtype=None):
    if isinstance(shape, int):
      shape = (shape,)
    elif isinstance(shape, (tuple, list)):
      if len(shape) > 2:
        raise ValueError(f'Only support initialize 2D weights for {self.__class__.__name__}.')
    else:
      raise ValueError(f'Only support shape of int, or tuple/list of int '
                       f'in {self.__class__.__name__}, but we got {shape}.')
    shape = [tools.size2num(d) for d in shape]
    return bm.eye(*shape, dtype=dtype) * self.value

  def __repr__(self):
    return f'{self.__class__.__name__}(value={self.value})'
