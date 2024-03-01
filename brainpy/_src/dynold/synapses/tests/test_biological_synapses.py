# -*- coding: utf-8 -*-

import pytest
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm

from brainpy._src.dependency_check import import_taichi

if import_taichi(error_if_not_found=False) is None:
  pytest.skip('no taichi', allow_module_level=True)

biological_models = [
  bp.synapses.AMPA,
  bp.synapses.GABAa,
  bp.synapses.BioNMDA,
]


class Test_Biological_Synapse(parameterized.TestCase):
  @parameterized.product(
    synapse=biological_models,
    delay_step=[None, 5, 1],
    mode=[bm.NonBatchingMode(), bm.BatchingMode(5)],
    stp=[None, bp.synplast.STP(), bp.synplast.STD()]
  )
  def test_all2all_synapse(self, synapse, delay_step, mode, stp):
    bm.random.seed()
    with bm.environment(mode=mode):
      pre_neu = bp.neurons.LIF(5)
      post_neu = bp.neurons.LIF(5)
      syn = synapse(pre_neu, post_neu, conn=bp.conn.All2All(), delay_step=delay_step, stp=stp)
      net = bp.Network(pre=pre_neu, syn=syn, post=post_neu)

    # 运行模拟
    runner = bp.DSRunner(net,
                         monitors=['pre.V', 'syn.g', 'post.V'],
                         inputs=('pre.input', 35.))
    runner(10.)

    expected_shape = (100, 5)
    if isinstance(mode, bm.BatchingMode):
      expected_shape = (mode.batch_size,) + expected_shape

    self.assertTupleEqual(runner.mon['pre.V'].shape, expected_shape)
    self.assertTupleEqual(runner.mon['syn.g'].shape, expected_shape)
    self.assertTupleEqual(runner.mon['post.V'].shape, expected_shape)
    bm.clear_buffer_memory()

  @parameterized.product(
    synapse=biological_models,
    delay_step=[None, 10, 1],
    mode=[bm.NonBatchingMode(), bm.BatchingMode(5), ],
    stp=[None, bp.synplast.STP(), bp.synplast.STD()]
  )
  def test_one2one_synapse(self, synapse, delay_step, mode, stp):
    bm.random.seed()
    with bm.environment(mode=mode):
      pre_neu = bp.neurons.LIF(5)
      post_neu = bp.neurons.LIF(5)
      syn = synapse(pre_neu, post_neu, conn=bp.conn.One2One(), delay_step=delay_step, stp=stp)
      net = bp.Network(pre=pre_neu, syn=syn, post=post_neu)

    # 运行模拟
    runner = bp.DSRunner(net,
                         monitors=['pre.V', 'syn.g', 'post.V'],
                         inputs=('pre.input', 35.))
    runner(10.)

    expected_shape = (100, 5)
    if isinstance(mode, bm.BatchingMode):
      expected_shape = (mode.batch_size,) + expected_shape
    self.assertTupleEqual(runner.mon['pre.V'].shape, expected_shape)
    self.assertTupleEqual(runner.mon['syn.g'].shape, expected_shape)
    self.assertTupleEqual(runner.mon['post.V'].shape, expected_shape)
    bm.clear_buffer_memory()

  @parameterized.product(
    synapse=biological_models,
    comp_method=['sparse', 'dense'],
    delay_step=[None, 10, 1],
    mode=[bm.NonBatchingMode(), bm.BatchingMode(5)],
    stp=[None, bp.synplast.STP(), bp.synplast.STD()]
  )
  def test_sparse_synapse(self, synapse, comp_method, delay_step, mode, stp):
    bm.random.seed()
    with bm.environment(mode=mode):
      pre_neu = bp.neurons.LIF(10)
      post_neu = bp.neurons.LIF(10)
      syn = synapse(pre_neu, post_neu, conn=bp.conn.FixedProb(0.5),
                    comp_method=comp_method, delay_step=delay_step,
                    stp=stp)
      net = bp.Network(pre=pre_neu, syn=syn, post=post_neu)

    # 运行模拟
    runner = bp.DSRunner(net,
                         monitors=['pre.V', 'syn.g', 'post.V'],
                         inputs=('pre.input', 35.))
    runner(10.)

    expected_shape = (100, 10)
    if isinstance(mode, bm.BatchingMode):
      expected_shape = (mode.batch_size,) + expected_shape
    self.assertTupleEqual(runner.mon['pre.V'].shape, expected_shape)
    self.assertTupleEqual(runner.mon['syn.g'].shape, expected_shape)
    self.assertTupleEqual(runner.mon['post.V'].shape, expected_shape)
    bm.clear_buffer_memory()
