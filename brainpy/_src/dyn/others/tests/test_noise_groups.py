# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
from absl.testing import parameterized


class Test_Noise_Group(parameterized.TestCase):
  def test_OU(self):
    bm.random.seed(1234)
    model = bp.dyn.OUProcess(size=1, mean=0., sigma=0.1)
    runner = bp.DSRunner(model,
                         monitors=['x'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['x'].shape, (100, 1))
    x = runner.mon['x']
    self.assertLessEqual(abs(x.mean()), 0.1)
    self.assertLessEqual(abs(x.std() - 0.1), 0.1)
    bm.clear_buffer_memory()
