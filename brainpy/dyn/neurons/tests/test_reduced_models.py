# -*- coding: utf-8 -*-


import brainpy as bp
from absl.testing import parameterized
from brainpy.dyn.neurons import reduced_models


class TestNoise(parameterized.TestCase):
  @parameterized.named_parameters(
    {'testcase_name': f'noise_of_{name}', 'neuron': name}
    for name in reduced_models.__all__
  )
  def test_noise(self, neuron):
    model = getattr(reduced_models, neuron)(size=1, noise=0.1)
    runner = bp.dyn.DSRunner(model, progress_bar=False)
    runner.run(10.)
