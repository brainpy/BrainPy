# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pytest

import brainpy as bp
from brainpy.integrators import sde

plt = None

sigma = 10
beta = 8 / 3
rho = 28
p = 0.1


def lorenz_f(x, y, z, t):
  dx = sigma * (y - x)
  dy = x * (rho - z) - y
  dz = x * y - beta * z
  return dx, dy, dz


def lorenz_g(x, y, z, t):
  return p * x, p * y, p * z


def lorenz_system(method, **kwargs):
  integral = bp.math.jit(method(f=lorenz_f,
                                g=lorenz_g,
                                show_code=True,
                                dt=0.005,
                                **kwargs))

  times = np.arange(0, 10, 0.01)
  mon1 = []
  mon2 = []
  mon3 = []
  x, y, z = 1, 1, 1
  for t in times:
    x, y, z = integral(x, y, z, t)
    mon1.append(x)
    mon2.append(y)
    mon3.append(z)
  mon1 = bp.math.array(mon1).to_numpy()
  mon2 = bp.math.array(mon2).to_numpy()
  mon3 = bp.math.array(mon3).to_numpy()

  global plt
  if plt is None:
    import matplotlib.pyplot as plt

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  plt.plot(mon1, mon2, mon3)
  ax.set_xlabel('x')
  ax.set_xlabel('y')
  ax.set_xlabel('z')
  plt.show()


class TestScalarWienerIntegral(unittest.TestCase):
  def test_srk1w1_try1(self):
    lorenz_system(sde.SRK1W1)

  def test_srk1w1_try2(self):
    with pytest.raises(AssertionError):
      lorenz_system(sde.SRK1W1, wiener_type=bp.integrators.VECTOR_WIENER)

  def test_srk2w1(self):
    lorenz_system(sde.SRK2W1)

  def test_euler(self):
    lorenz_system(sde.Euler, intg_type=bp.integrators.ITO_SDE)
    lorenz_system(sde.Euler, intg_type=bp.integrators.STRA_SDE)

  def test_milstein(self):
    lorenz_system(sde.Milstein, intg_type=bp.integrators.ITO_SDE)
    lorenz_system(sde.Milstein, intg_type=bp.integrators.STRA_SDE)
