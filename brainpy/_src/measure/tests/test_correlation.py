# -*- coding: utf-8 -*-


import unittest
import brainpy as bp
import brainpy.math as bm
from jax import jit
from functools import partial


class TestCrossCorrelation(unittest.TestCase):
  def test_c(self):
    spikes = bp.math.asarray([[1, 0, 1, 0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]]).T
    cc1 = bp.measure.cross_correlation(spikes, 1., dt=1.)
    f_cc = jit(partial(bp.measure.cross_correlation, numpy=False, bin=1, dt=1.))
    cc2 = f_cc(spikes)
    print(cc1, cc2)
    self.assertTrue(cc1 == cc2)

  def test_cc(self):
    spikes = bp.math.ones((1000, 10))
    cc1 = bp.measure.cross_correlation(spikes, 1.)
    self.assertTrue(cc1 == 1.)

    spikes = bp.math.zeros((1000, 10))
    cc2 = bp.measure.cross_correlation(spikes, 1.)
    self.assertTrue(cc2 == 0.)

  def test_cc2(self):
    bp.math.random.seed()
    spikes = bp.math.random.randint(0, 2, (1000, 10))
    print(bp.measure.cross_correlation(spikes, 1.))
    print(bp.measure.cross_correlation(spikes, 0.5))

  def test_cc3(self):
    bp.math.random.seed()
    spikes = bp.math.random.random((1000, 100)) < 0.8
    print(bp.measure.cross_correlation(spikes, 1.))
    print(bp.measure.cross_correlation(spikes, 0.5))

  def test_cc4(self):
    bp.math.random.seed()
    spikes = bp.math.random.random((1000, 100)) < 0.2
    print(bp.measure.cross_correlation(spikes, 1.))
    print(bp.measure.cross_correlation(spikes, 0.5))

  def test_cc5(self):
    bp.math.random.seed()
    spikes = bp.math.random.random((1000, 100)) < 0.05
    print(bp.measure.cross_correlation(spikes, 1.))
    print(bp.measure.cross_correlation(spikes, 0.5))


class TestVoltageFluctuation(unittest.TestCase):
  def test_vf1(self):
    rng = bp.math.random.RandomState(122)
    voltages = rng.normal(0, 10, size=(1000, 100))
    print(bp.measure.voltage_fluctuation(voltages))

    bm.enable_x64()
    voltages = bp.math.ones((1000, 100)).value
    r1 = bp.measure.voltage_fluctuation(voltages)

    jit_f = jit(partial(bp.measure.voltage_fluctuation, numpy=False))
    jit_f = jit(lambda a: bp.measure.voltage_fluctuation(a, numpy=False))
    r2 = jit_f(voltages)
    print(r1, r2)  # TODO: JIT results are different?
    # self.assertTrue(r1 == r2)

    bm.disable_x64()


class TestFunctionalConnectivity(unittest.TestCase):
  def test_cf1(self):
    bp.math.random.seed()
    act = bp.math.random.random((10000, 3))
    r1 = bp.measure.functional_connectivity(act)

    jit_f = jit(partial(bp.measure.functional_connectivity, numpy=False))
    r2 = jit_f(act)

    self.assertTrue(bm.allclose(r1, r2))


class TestMatrixCorrelation(unittest.TestCase):
  def test_mc(self):
    bp.math.random.seed()
    A = bp.math.random.random((100, 100))
    B = bp.math.random.random((100, 100))
    r1 = (bp.measure.matrix_correlation(A, B))

    jit_f = jit(partial(bp.measure.matrix_correlation, numpy=False))
    r2 = jit_f(A, B)

    self.assertTrue(bm.allclose(r1, r2))


