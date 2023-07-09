# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
from absl.testing import parameterized


class Test_learning_rule(parameterized.TestCase):
  @parameterized.product(
    delay_step=[None, 5, 1],
    mode=[bm.NonBatchingMode(), bm.BatchingMode(5), bm.TrainingMode(5)]
  )
  def test_learning_rule(self, delay_step, mode):
    bm.random.seed()
    with bm.environment(mode=mode):
      neu1 = bp.neurons.LIF(5)
      neu2 = bp.neurons.LIF(5)
      syn1 = bp.synapses.STP(neu1, neu2, bp.connect.All2All(), U=0.1, tau_d=10, tau_f=100.,
                             delay_step=delay_step)
      net = bp.Network(pre=neu1, syn=syn1, post=neu2)

    runner = bp.DSRunner(net, inputs=[('pre.input', 28.)], monitors=['syn.I', 'syn.u', 'syn.x'])
    runner.run(10.)

    expected_shape = (100, 5)
    if isinstance(mode, bm.BatchingMode):
      expected_shape = (mode.batch_size,) + expected_shape
    self.assertTupleEqual(runner.mon['syn.I'].shape, expected_shape)
    self.assertTupleEqual(runner.mon['syn.u'].shape, expected_shape)
    self.assertTupleEqual(runner.mon['syn.x'].shape, expected_shape)
    bm.clear_buffer_memory()

