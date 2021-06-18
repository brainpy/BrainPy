# -*- coding: utf-8 -*-

from brainpy.backend.driver.numpy import NumpyDSDriver
from brainpy.backend.driver.numpy import NumpyDiffIntDriver

__all__ = [
  'JaxDSDriver',
  'JaxDiffIntDriver',
]


class JaxDiffIntDriver(NumpyDiffIntDriver):
  pass


class JaxDSDriver(NumpyDSDriver):
  pass
