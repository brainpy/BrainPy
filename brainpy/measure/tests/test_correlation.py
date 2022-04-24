# -*- coding: utf-8 -*-


import unittest
import brainpy as bp


class TestCrossCorrelation(unittest.TestCase):
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
    bp.math.random.seed()
    voltages = bp.math.random.normal(0, 10, size=(1000, 100))
    print(bp.measure.voltage_fluctuation(voltages))

    voltages = bp.math.ones((1000, 100))
    print(bp.measure.voltage_fluctuation(voltages))


class TestFunctionalConnectivity(unittest.TestCase):
  def test_cf1(self):
    bp.math.random.seed()
    act = bp.math.random.random((10000, 3))
    print(bp.measure.functional_connectivity(act))


class TestMatrixCorrelation(unittest.TestCase):
  def test_mc(self):
    bp.math.random.seed()
    A = bp.math.random.random((100, 100))
    B = bp.math.random.random((100, 100))
    print(bp.measure.matrix_correlation(A, B))

