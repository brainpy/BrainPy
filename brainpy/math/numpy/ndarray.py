# -*- coding: utf-8 -*-

import numpy as np


__all__ = [
  'ndarray',
  'Variable',
  'TrainVar',
]

ndarray = np.ndarray


class Variable(np.ndarray):
  pass


class TrainVar(Variable):
  pass
