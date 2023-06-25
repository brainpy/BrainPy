# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
from absl.testing import parameterized
from brainpy._src.synapses_v2 import abstract_synapses


class Test_Abstract_Synapse(parameterized.TestCase):
  @parameterized.named_parameters(
    {'testcase_name': f'noise_of_{name}', 'synapse': name}
    for name in ['Exponential', 'DualExponential', 'Alpha']
  )
  def test_all2all_synapse(self, synapse):
    pre_neu = bp.neurons.LIF(5)
    post_neu = bp.neurons.LIF(5)
    syn_model = getattr(abstract_synapses, synapse)
    syn = syn_model(conn=bp.conn.All2All(), out=bp.experimental.COBA(E=0.))

    net = bp.Network(pre=pre_neu, syn=syn, post=post_neu)

    # 运行模拟
    runner = bp.DSRunner(net,
                         monitors=['pre.V', 'syn.g', 'post.V'],
                         inputs=('pre.input', 35.))
    runner(10.)
    self.assertTupleEqual(runner.mon['pre.V'].shape, (100, 5))
    self.assertTupleEqual(runner.mon['syn.g'].shape, (100, 5))
    self.assertTupleEqual(runner.mon['post.V'].shape, (100, 5))

  # @parameterized.named_parameters(
  #   {'testcase_name': f'noise_of_{name}', 'synapse': name}
  #   for name in ['Exponential', 'DualExponential', 'Alpha', 'NMDA']
  # )
  # def test_one2one_synapse(self, synapse):
  #   pre_neu = bp.neurons.LIF(5)
  #   post_neu = bp.neurons.LIF(5)
  #   syn_model = getattr(abstract_synapses, synapse)
  #   syn = syn_model(pre_neu, post_neu, conn=bp.conn.One2One())
  #   net = bp.Network(pre=pre_neu, syn=syn, post=post_neu)
  #
  #   # 运行模拟
  #   runner = bp.DSRunner(net,
  #                        monitors=['pre.V', 'syn.g', 'post.V'],
  #                        inputs=('pre.input', 35.))
  #   runner(10.)
  #   self.assertTupleEqual(runner.mon['pre.V'].shape, (100, 5))
  #   self.assertTupleEqual(runner.mon['syn.g'].shape, (100, 5))
  #   self.assertTupleEqual(runner.mon['post.V'].shape, (100, 5))
  #
  # @parameterized.named_parameters(
  #   {'testcase_name': f'noise_of_{name}', 'synapse': name}
  #   for name in ['Exponential', 'DualExponential', 'Alpha', 'NMDA']
  # )
  # def test_sparse_synapse(self, synapse):
  #   pre_neu = bp.neurons.LIF(5)
  #   post_neu = bp.neurons.LIF(5)
  #   syn_model = getattr(abstract_synapses, synapse)
  #   syn = syn_model(pre_neu, post_neu, conn=bp.conn.FixedProb(0.1), comp_method='sparse')
  #   net = bp.Network(pre=pre_neu, syn=syn, post=post_neu)
  #
  #   # 运行模拟
  #   runner = bp.DSRunner(net,
  #                        monitors=['pre.V', 'syn.g', 'post.V'],
  #                        inputs=('pre.input', 35.))
  #   runner(10.)
  #   self.assertTupleEqual(runner.mon['pre.V'].shape, (100, 5))
  #   self.assertTupleEqual(runner.mon['syn.g'].shape, (100, 5))
  #   self.assertTupleEqual(runner.mon['post.V'].shape, (100, 5))
