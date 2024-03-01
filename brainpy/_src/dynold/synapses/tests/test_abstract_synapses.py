# -*- coding: utf-8 -*-


from absl.testing import parameterized

import pytest
import brainpy as bp
import brainpy.math as bm
from brainpy._src.dynold.synapses import abstract_models
from brainpy._src.dependency_check import import_taichi

if import_taichi(error_if_not_found=False) is None:
  pytest.skip('no taichi', allow_module_level=True)


class Test_Abstract_Synapse(parameterized.TestCase):
  @parameterized.product(
    name=['Exponential', 'DualExponential', 'Alpha', 'NMDA'],
    stp=[None, bp.synplast.STD(), bp.synplast.STP()],
    mode=[bm.nonbatching_mode, bm.BatchingMode(5), bm.TrainingMode(5)]
  )
  def test_all2all_synapse(self, name, stp, mode):
    bm.random.seed()
    with bm.environment(mode=mode):
      pre_neu = bp.neurons.LIF(5)
      post_neu = bp.neurons.LIF(5)
      syn_model = getattr(bp.synapses, name)
      syn = syn_model(pre_neu, post_neu, conn=bp.conn.All2All(), stp=stp)
      net = bp.Network(pre=pre_neu, syn=syn, post=post_neu)

    # 运行模拟
    runner = bp.DSRunner(net, monitors=['pre.V', 'syn.g', 'post.V'], inputs=('pre.input', 35.))
    runner(10.)

    expected_shape = (100, 5)
    if isinstance(mode, bm.BatchingMode):
      expected_shape = (mode.batch_size, ) + expected_shape
    self.assertTupleEqual(runner.mon['pre.V'].shape, expected_shape)
    self.assertTupleEqual(runner.mon['syn.g'].shape, expected_shape)
    self.assertTupleEqual(runner.mon['post.V'].shape, expected_shape)
    bm.clear_buffer_memory()

  @parameterized.product(
    name=['Exponential', 'DualExponential', 'Alpha', 'NMDA'],
    stp=[None, bp.synplast.STD(), bp.synplast.STP()],
    mode=[bm.nonbatching_mode, bm.BatchingMode(5), bm.TrainingMode(5)]
  )
  def test_one2one_synapse(self, name, stp, mode):
    bm.random.seed()
    with bm.environment(mode=mode):
      pre_neu = bp.neurons.LIF(5)
      post_neu = bp.neurons.LIF(5)
      syn_model = getattr(abstract_models, name)
      syn = syn_model(pre_neu, post_neu, conn=bp.conn.One2One(), stp=stp)
      net = bp.Network(pre=pre_neu, syn=syn, post=post_neu)

    # 运行模拟
    runner = bp.DSRunner(net,
                         monitors=['pre.V', 'syn.g', 'post.V'],
                         inputs=('pre.input', 35.))
    runner(10.)

    expected_shape = (100, 5)
    if isinstance(mode, bm.BatchingMode):
      expected_shape = (mode.batch_size, ) + expected_shape
    self.assertTupleEqual(runner.mon['pre.V'].shape, expected_shape)
    self.assertTupleEqual(runner.mon['syn.g'].shape, expected_shape)
    self.assertTupleEqual(runner.mon['post.V'].shape, expected_shape)
    bm.clear_buffer_memory()

  @parameterized.product(
    comp_type=['sparse', 'dense'],
    name=['Exponential', 'DualExponential', 'Alpha', 'NMDA'],
    stp=[None, bp.synplast.STD(), bp.synplast.STP()],
    mode=[bm.nonbatching_mode, bm.BatchingMode(5), bm.TrainingMode(5)]
  )
  def test_sparse_synapse(self, comp_type, name, stp, mode):
    bm.random.seed()
    with bm.environment(mode=mode):
      pre_neu = bp.neurons.LIF(5)
      post_neu = bp.neurons.LIF(5)
      syn_model = getattr(abstract_models, name)
      syn = syn_model(pre_neu, post_neu, conn=bp.conn.FixedProb(0.1), comp_method=comp_type, stp=stp)
      net = bp.Network(pre=pre_neu, syn=syn, post=post_neu)

    # 运行模拟
    runner = bp.DSRunner(net,
                         monitors=['pre.V', 'syn.g', 'post.V'],
                         inputs=('pre.input', 35.))
    runner(10.)

    expected_shape = (100, 5)
    if isinstance(mode, bm.BatchingMode):
      expected_shape = (mode.batch_size, ) + expected_shape
    self.assertTupleEqual(runner.mon['pre.V'].shape, expected_shape)
    self.assertTupleEqual(runner.mon['syn.g'].shape, expected_shape)
    self.assertTupleEqual(runner.mon['post.V'].shape, expected_shape)
    bm.clear_buffer_memory()

  @parameterized.product(
    post_ref_key=[None, 'refractory'],
    stp=[None, bp.synplast.STD(), bp.synplast.STP()],
    mode=[bm.nonbatching_mode, bm.BatchingMode(5), bm.TrainingMode(5)]
  )
  def test_delta_synapse(self, post_ref_key, stp, mode):
    bm.random.seed()
    with bm.environment(mode=mode):
      pre_neu = bp.neurons.LIF(5, ref_var=True)
      post_neu = bp.neurons.LIF(3, ref_var=True)
      syn = bp.synapses.Delta(pre_neu, post_neu,
                              conn=bp.conn.All2All(),
                              post_ref_key=post_ref_key,
                              stp=stp, )
      net = bp.Network(pre=pre_neu, syn=syn, post=post_neu)

    # 运行模拟
    runner = bp.DSRunner(net,
                         monitors=['pre.V', 'post.V'],
                         inputs=('pre.input', 35.))
    runner(10.)

    pre_expected_shape = (100, 5)
    post_expected_shape = (100, 3)
    if isinstance(mode, bm.BatchingMode):
      pre_expected_shape = (mode.batch_size,) + pre_expected_shape
      post_expected_shape = (mode.batch_size,) + post_expected_shape
    self.assertTupleEqual(runner.mon['pre.V'].shape, pre_expected_shape)
    self.assertTupleEqual(runner.mon['post.V'].shape, post_expected_shape)
    bm.clear_buffer_memory()