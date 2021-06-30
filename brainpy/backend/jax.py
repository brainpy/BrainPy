# -*- coding: utf-8 -*-

from brainpy.backend import numpy as np_backend

__all__ = [
  'JaxDSDriver',
  'JaxDiffIntDriver',
]


class JaxDiffIntDriver(np_backend.NumpyDiffIntDriver):
  pass


class JaxDSDriver(np_backend.NumpyDSDriver):
  pass
