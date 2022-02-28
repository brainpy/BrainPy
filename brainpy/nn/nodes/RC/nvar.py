# -*- coding: utf-8 -*-

from brainpy.nn.base import Node

__all__ = [
  'NVAR'
]


class NVAR(Node):
  """Nonlinear vector autoregression (NVAR) node.

  Parameters
  ----------


  References
  ----------
  .. [1] Gauthier, D.J., Bollt, E., Griffith, A. et al. Next generation
         reservoir computing. Nat Commun 12, 5564 (2021).
         https://doi.org/10.1038/s41467-021-25801-2

  """
  def __init__(self, **kwargs):
    super(NVAR, self).__init__(**kwargs)
