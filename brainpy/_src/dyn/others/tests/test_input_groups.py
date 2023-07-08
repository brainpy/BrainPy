# -*- coding: utf-8 -*-


import brainpy as bp
from absl.testing import parameterized
from brainpy._src.neurons import input_groups


class Test_input_Group(parameterized.TestCase):
  def test_SpikeTimeGroup(self):
    bp.math.random.seed()
    model = input_groups.SpikeTimeGroup(size=2, times=[10, 20, 20, 30], indices=[0, 0, 1, 1])
    runner = bp.DSRunner(model,
                         monitors=['spike'],
                         progress_bar=False)
    runner.run(30.)
    self.assertTupleEqual(runner.mon['spike'].shape, (300, 2))
    bp.math.clear_buffer_memory()

  def test_PoissonGroup(self):
    bp.math.random.seed()
    model = input_groups.PoissonGroup(size=2, freqs=1000, seed=0)
    runner = bp.DSRunner(model,
                         monitors=['spike'],
                         progress_bar=False)
    runner.run(30.)
    self.assertTupleEqual(runner.mon['spike'].shape, (300, 2))
    bp.math.clear_buffer_memory()
