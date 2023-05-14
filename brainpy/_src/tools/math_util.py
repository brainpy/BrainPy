# -*- coding: utf-8 -*-

import numpy as np


__all__ = [
  'format_seed'
]


def format_seed(seed=None):
  """Get the random sed.
  """
  if seed is None:
    return np.random.randint(0, int(1e7))
  else:
    return seed

