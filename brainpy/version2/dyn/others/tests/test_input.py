# -*- coding: utf-8 -*-


from absl.testing import parameterized

import brainpy.version2 as bp
from brainpy.version2.dyn.others import input


class Test_input(parameterized.TestCase):
    def test_SpikeTimeGroup(self):
        model = input.SpikeTimeGroup(size=2, times=[10, 20, 20, 30], indices=[0, 0, 1, 1])
        runner = bp.DSRunner(model,
                             monitors=['spike'],
                             progress_bar=False)
        runner.run(30.)
        self.assertTupleEqual(runner.mon['spike'].shape, (300, 2))

    def test_PoissonGroup(self):
        model = input.PoissonGroup(size=2, freqs=1000, seed=0)
        runner = bp.DSRunner(model,
                             monitors=['spike'],
                             progress_bar=False)
        runner.run(30.)
        self.assertTupleEqual(runner.mon['spike'].shape, (300, 2))
