# -*- coding: utf-8 -*-

from brainpy.backend.numpy import NumpyDiffIntDriver
from brainpy.backend.numpy import NumpyDSDriver

__all__ = [
  'JaxDSDriver',
  'JaxDiffIntDriver',
]


class JaxDiffIntDriver(NumpyDiffIntDriver):
  pass


class JaxDSDriver(NumpyDSDriver):
  pass
