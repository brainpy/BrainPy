# -*- coding: utf-8 -*-

from brainpy.analysis.numeric.solver import jax_brentq


def f(x):
  return (x ** 2 - 1)


print(jax_brentq(f, -2, 0))
