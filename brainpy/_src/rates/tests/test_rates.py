# -*- coding: utf-8 -*-


import brainpy as bp
from absl.testing import parameterized
from brainpy._src.rates import populations
from unittest import TestCase


class TestRate(TestCase):
  def test_fhn(self):
    fhn = bp.rates.FHN(10)
    self.assertTrue(fhn.tau is not None)

  def test_ffhn(self):
    ffhn = bp.rates.FeedbackFHN(size=1)
    self.assertTrue(ffhn.tau is not None)

  def test_qif(self):
    qif = bp.rates.QIF(size=1)
    self.assertTrue(qif.tau is not None)

  def test_slo(self):
    slo = bp.rates.StuartLandauOscillator(size=1)
    self.assertTrue(slo.x_ou_tau is not None)

  def test_wcm(self):
    wcm = bp.rates.WilsonCowanModel(size=1)
    self.assertTrue(wcm.x_ou_tau is not None)

  def test_tlm(self):
    tlm = bp.rates.ThresholdLinearModel(size=1)
    self.assertTrue(tlm.tau_e is not None)


class TestPopulation(parameterized.TestCase):
  @parameterized.named_parameters(
    {'testcase_name': f'noise_of_{name}', 'neuron': name}
    for name in populations.__all__
  )
  def test_runner(self, neuron):
    model = getattr(populations, neuron)(size=10)
    runner = bp.DSRunner(model, progress_bar=False)
    runner.run(10.)

class TestShape(parameterized.TestCase):
  def test_FHN_shape(self):
    model = getattr(populations, 'FHN')(size=10)
    runner = bp.DSRunner(model,
                         monitors=['x'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon.x.shape, (100, 10))

  def test_FFHN_shape(self):
    model = getattr(populations, 'FeedbackFHN')(size=10)
    runner = bp.DSRunner(model,
                         monitors=['x'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon.x.shape, (100, 10))

  def test_QIF_shape(self):
    model = getattr(populations, 'QIF')(size=10)
    runner = bp.DSRunner(model,
                         monitors=['x'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon.x.shape, (100, 10))

  def test_SLO_shape(self):
    model = getattr(populations, 'StuartLandauOscillator')(size=10)
    runner = bp.DSRunner(model,
                         monitors=['x'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon.x.shape, (100, 10))

  def test_TLM_shape(self):
    model = getattr(populations, 'ThresholdLinearModel')(size=10)
    runner = bp.DSRunner(model,
                         monitors=['e'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon.e.shape, (100, 10))