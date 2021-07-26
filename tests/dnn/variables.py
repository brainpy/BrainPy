# -*- coding: utf-8 -*-

import jax

from brainpy.dnn.variables import TrainVar
from brainpy.math.jax import ones


@jax.jit
def f(a, b):
  return a + b


a1 = ones(100)
b1 = ones(100)
a2 = TrainVar(a1)
b2 = TrainVar(b1)
f(a1, b1)
f(a2, b2)
