# -*- coding: utf-8 -*-


import unittest
from functools import partial

from jax import jit

import brainpy as bp
import brainpy.math as bm

bm.set_platform('cpu')

class TestCrossCorrelation(unittest.TestCase):
  def test_c(self):
    bm.random.seed()
    spikes = bm.asarray([[1, 0, 1, 0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]]).T
    cc1 = bp.measure.cross_correlation(spikes, 1., dt=1.)
    f_cc = jit(partial(bp.measure.cross_correlation, numpy=False, bin=1, dt=1.))
    cc2 = f_cc(spikes)
    print(cc1, cc2)
    self.assertTrue(cc1 == cc2)
    bm.clear_buffer_memory()

  def test_cc(self):
    bm.random.seed()
    spikes = bm.ones((1000, 10))
    cc1 = bp.measure.cross_correlation(spikes, 1.)
    self.assertTrue(cc1 == 1.)

    spikes = bm.zeros((1000, 10))
    cc2 = bp.measure.cross_correlation(spikes, 1.)
    self.assertTrue(cc2 == 0.)

    bm.clear_buffer_memory()

  def test_cc2(self):
    bm.random.seed()
    spikes = bm.random.randint(0, 2, (1000, 10))
    print(bp.measure.cross_correlation(spikes, 1.))
    print(bp.measure.cross_correlation(spikes, 0.5))
    bm.clear_buffer_memory()

  def test_cc3(self):
    bm.random.seed()
    spikes = bm.random.random((1000, 100)) < 0.8
    print(bp.measure.cross_correlation(spikes, 1.))
    print(bp.measure.cross_correlation(spikes, 0.5))
    bm.clear_buffer_memory()

  def test_cc4(self):
    bm.random.seed()
    spikes = bm.random.random((1000, 100)) < 0.2
    print(bp.measure.cross_correlation(spikes, 1.))
    print(bp.measure.cross_correlation(spikes, 0.5))
    bm.clear_buffer_memory()

  def test_cc5(self):
    bm.random.seed()
    spikes = bm.random.random((1000, 100)) < 0.05
    print(bp.measure.cross_correlation(spikes, 1.))
    print(bp.measure.cross_correlation(spikes, 0.5))
    bm.clear_buffer_memory()


class TestVoltageFluctuation(unittest.TestCase):
  def test_vf1(self):
    bm.random.seed()
    voltages = bm.random.normal(0, 10, size=(100, 10))
    print(bp.measure.voltage_fluctuation(voltages))

    bm.enable_x64()
    voltages = bm.ones((100, 10))
    r1 = bp.measure.voltage_fluctuation(voltages)

    jit_f = jit(partial(bp.measure.voltage_fluctuation, numpy=False))
    jit_f = jit(lambda a: bp.measure.voltage_fluctuation(a, numpy=False))
    r2 = jit_f(voltages)
    print(r1, r2)  # TODO: JIT results are different?
    # self.assertTrue(r1 == r2)

    bm.disable_x64()
    bm.clear_buffer_memory()


class TestFunctionalConnectivity(unittest.TestCase):
  def test_cf1(self):
    bm.random.seed()
    act = bm.random.random((10000, 3))
    r1 = bp.measure.functional_connectivity(act)

    jit_f = jit(partial(bp.measure.functional_connectivity, numpy=False))
    r2 = jit_f(act)

    self.assertTrue(bm.allclose(r1, r2))
    bm.clear_buffer_memory()


class TestMatrixCorrelation(unittest.TestCase):
  def test_mc(self):
    bm.random.seed()
    A = bm.random.random((100, 100))
    B = bm.random.random((100, 100))
    r1 = (bp.measure.matrix_correlation(A, B))

    jit_f = jit(partial(bp.measure.matrix_correlation, numpy=False))
    r2 = jit_f(A, B)
    self.assertTrue(bm.allclose(r1, r2))
    bm.clear_buffer_memory()


