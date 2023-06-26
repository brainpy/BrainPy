# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
from absl.testing import parameterized
from brainpy._src.dyn.neurons import hh

class Test_HH(parameterized.TestCase):
  def test_HH(self):
    model = hh.HH(size=1)
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
    model = hh.HH(size=10, mode=bm.batching_mode)
    runner = bp.DSRunner(model,
                         monitors=['V', 'm', 'n', 'h', 'spike'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
    self.assertTupleEqual(runner.mon['m'].shape, (1, 100, 10))
    self.assertTupleEqual(runner.mon['n'].shape, (1, 100, 10))
    self.assertTupleEqual(runner.mon['h'].shape, (1, 100, 10))
    self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

  def test_HHLTC(self):
    model = hh.HHLTC(size=1)
    runner = bp.DSRunner(model,
                         monitors=['V', 'm', 'n', 'h', 'spike'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['m'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['n'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['h'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

  def test_HHLTC_batching_mode(self):
    model = hh.HHLTC(size=10, mode=bm.batching_mode)
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
    model = hh.MorrisLecar(size=1)
    runner = bp.DSRunner(model,
                          monitors=['V', 'W', 'spike'],
                          progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['W'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

  def test_MorrisLecar_batching_mode(self):
    model = hh.MorrisLecar(size=10, mode=bm.batching_mode)
    runner = bp.DSRunner(model,
                          monitors=['V', 'W', 'spike'],
                          progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
    self.assertTupleEqual(runner.mon['W'].shape, (1, 100, 10))
    self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

  def test_MorrisLecarLTC(self):
    model = hh.MorrisLecarLTC(size=1)
    runner = bp.DSRunner(model,
                          monitors=['V', 'W', 'spike'],
                          progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['W'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

  def test_MorrisLecarLTC_batching_mode(self):
    model = hh.MorrisLecarLTC(size=10, mode=bm.batching_mode)
    runner = bp.DSRunner(model,
                          monitors=['V', 'W', 'spike'],
                          progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
    self.assertTupleEqual(runner.mon['W'].shape, (1, 100, 10))
    self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

  def test_WangBuzsakiModel(self):
    model = hh.WangBuzsakiModel(size=1)
    runner = bp.DSRunner(model,
                          monitors=['V', 'n', 'h', 'spike'],
                          progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['n'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['h'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

  def test_WangBuzsakiModel_batching_mode(self):
    model = hh.WangBuzsakiModel(size=10, mode=bm.batching_mode)
    runner = bp.DSRunner(model,
                          monitors=['V', 'n', 'h', 'spike'],
                          progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
    self.assertTupleEqual(runner.mon['n'].shape, (1, 100, 10))
    self.assertTupleEqual(runner.mon['h'].shape, (1, 100, 10))
    self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

  def test_WangBuzsakiModelLTC(self):
    model = hh.WangBuzsakiModelLTC(size=1)
    runner = bp.DSRunner(model,
                          monitors=['V', 'n', 'h', 'spike'],
                          progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['n'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['h'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

  def test_WangBuzsakiModelLTC_batching_mode(self):
    model = hh.WangBuzsakiModelLTC(size=10, mode=bm.batching_mode)
    runner = bp.DSRunner(model,
                          monitors=['V', 'n', 'h', 'spike'],
                          progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
    self.assertTupleEqual(runner.mon['n'].shape, (1, 100, 10))
    self.assertTupleEqual(runner.mon['h'].shape, (1, 100, 10))
    self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))