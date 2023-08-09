# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
from absl.testing import parameterized
from brainpy._src.dynold.neurons import reduced_models


class Test_Reduced(parameterized.TestCase):
  @parameterized.named_parameters(
    {'testcase_name': f'noise_of_{name}', 'neuron': name}
    for name in reduced_models.__all__
  )
  def test_run_shape(self, neuron):
    bm.random.seed()
    model = getattr(reduced_models, neuron)(size=1)
    if neuron == 'LeakyIntegrator':
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
    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    {'testcase_name': f'noise_of_{name}', 'neuron': name}
    for name in reduced_models.__all__
  )
  def test_noise_shape(self, neuron):
    bm.random.seed()
    model = getattr(reduced_models, neuron)(size=1, noise=0.1)
    if neuron == 'LeakyIntegrator':
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
    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    {'testcase_name': f'noise_of_{name}', 'neuron': name}
    for name in reduced_models.__all__
  )
  def test_training_shape(self, neuron):
    bm.random.seed()
    if neuron == 'FHN':
      model = getattr(reduced_models, neuron)(size=10)
      runner = bp.DSRunner(model,
                           monitors=['V'],
                           progress_bar=False)
      runner.run(10.)
      self.assertTupleEqual(runner.mon['V'].shape, (100, 10))
    else:
      model = getattr(reduced_models, neuron)(size=10, mode=bm.training_mode)
      runner = bp.DSRunner(model,
                           monitors=['V'],
                           progress_bar=False)
      runner.run(10.)
      self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
    bm.clear_buffer_memory()
