# -*- coding: utf-8 -*-

from brainpy import math
from brainpy.integrators.driver.base import *
from brainpy.integrators.driver.jax_driver import *
from brainpy.integrators.driver.numpy_driver import *


def get_driver():
  backend = math.get_backend_name()
  if backend == 'numpy':
    return NumpyDiffIntDriver
  elif backend == 'jax':
    return JaxDiffIntDriver
  else:
    raise NotImplementedError(f'Unknown backend "{backend}"')
