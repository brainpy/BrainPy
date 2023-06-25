# -*- coding: utf-8 -*-


import brainpy as bp
from absl.testing import parameterized
from brainpy._src.neurons import fractional_models


class Test_Fractional(parameterized.TestCase):
  def test_FractionalFHR(self):
    model = fractional_models.FractionalFHR(size=1, alpha=0.5)
    runner = bp.DSRunner(model,
                         monitors=['V', 'w', 'y', 'spike'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['w'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['y'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

  def test_FractionalIzhikevich(self):
    model = fractional_models.FractionalIzhikevich(size=1, alpha=0.5, num_memory=1000)
    runner = bp.DSRunner(model,
                         monitors=['V', 'u', 'spike'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['u'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))