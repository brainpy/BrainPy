# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
from absl.testing import parameterized
from brainpy._src.dyn.neurons import lif

class Test_lif(parameterized.TestCase):
  @parameterized.named_parameters(
    {'testcase_name': f'{name}', 'neuron': name}
    for name in lif.__all__
  )
  def test_run_shape(self, neuron):
    model = getattr(lif, neuron)(size=1)
    if neuron in ['IF', 'IFLTC']:
      runner = bp.DSRunner(model,
                           monitors=['V'],
                           progress_bar=False)
      runner.run(10.)
      self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    else:
      runner = bp.DSRunner(model,
                           monitors=['V', 'spike'],
                           progress_bar=False)
      runner.run(10.)
      self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
      self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))


  @parameterized.named_parameters(
    {'testcase_name': f'{name}', 'neuron': name}
    for name in lif.__all__
  )
  def test_training_shape(self, neuron):
    model = getattr(lif, neuron)(size=10, mode=bm.training_mode)
    runner = bp.DSRunner(model,
                         monitors=['V'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
