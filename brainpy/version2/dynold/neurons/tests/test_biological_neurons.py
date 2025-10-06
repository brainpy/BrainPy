# -*- coding: utf-8 -*-


from absl.testing import parameterized

import brainpy.version2 as bp
import brainpy.version2.math as bm
from brainpy.version2.dynold.neurons import biological_models


class Test_Biological(parameterized.TestCase):
    def test_HH(self):
        bm.random.seed()
        model = biological_models.HH(size=1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'm', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['m'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['n'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['h'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_HH_with_noise(self):
        bm.random.seed()
        model = biological_models.HH(size=1, noise=0.1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'm', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['m'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['n'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['h'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_HH_batching_mode(self):
        bm.random.seed()
        model = biological_models.HH(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['V', 'm', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['m'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['n'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['h'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

    def test_MorrisLecar(self):
        bm.random.seed()
        model = biological_models.MorrisLecar(size=1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'W', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['W'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_MorrisLecar_with_noise(self):
        bm.random.seed()
        model = biological_models.MorrisLecar(size=1, noise=0.1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'W', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['W'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_MorrisLecar_batching_mode(self):
        bm.random.seed()
        model = biological_models.MorrisLecar(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['V', 'W', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['W'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

    def test_PinskyRinzelModel(self):
        bm.random.seed()
        model = biological_models.PinskyRinzelModel(size=1)
        runner = bp.DSRunner(model,
                             monitors=['Vs', 'Vd'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['Vs'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['Vd'].shape, (100, 1))

    def test_PinskyRinzelModel_with_noise(self):
        bm.random.seed()
        model = biological_models.PinskyRinzelModel(size=1, noise=0.1)
        runner = bp.DSRunner(model,
                             monitors=['Vs', 'Vd'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['Vs'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['Vd'].shape, (100, 1))

    def test_PinskyRinzelModel_batching_mode(self):
        bm.random.seed()
        model = biological_models.PinskyRinzelModel(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['Vs', 'Vd'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['Vs'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['Vd'].shape, (1, 100, 10))

    def test_WangBuzsakiModel(self):
        bm.random.seed()
        model = biological_models.WangBuzsakiModel(size=1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['n'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['h'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_WangBuzsakiModel_with_noise(self):
        bm.random.seed()
        model = biological_models.WangBuzsakiModel(size=1, noise=0.1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['n'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['h'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_WangBuzsakiModel_batching_mode(self):
        bm.random.seed()
        model = biological_models.WangBuzsakiModel(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['V', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['n'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['h'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))
