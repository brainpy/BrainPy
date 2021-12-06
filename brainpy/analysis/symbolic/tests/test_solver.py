# -*- coding: utf-8 -*-

from brainpy.analysis.numeric.solver import brentq


def f(x):
  return (x ** 2 - 1)


print(brentq(f, -2, 0))
