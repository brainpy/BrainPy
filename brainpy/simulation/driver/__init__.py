# -*- coding: utf-8 -*-

from brainpy import math
from brainpy.simulation.driver.base import *
from brainpy.simulation.driver.jax_driver import *
from brainpy.simulation.driver.numpy_driver import *


def get_driver():
  backend = math.get_backend_name()
  if backend == 'numpy':
    return NumpyDSDriver
  elif backend == 'jax':
    return JaxDSDriver
  else:
    raise NotImplementedError(f'Unknown backend "{backend}"')
