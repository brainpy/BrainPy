# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
from absl.testing import parameterized
from brainpy._src.synapses import learning_rules

class Test_learning_rule(parameterized.TestCase):
  def test_learning_rule(self):
    neu1 = bp.neurons.LIF(5)
    neu2 = bp.neurons.LIF(5)
    syn1 = learning_rules.STP(neu1, neu2, bp.connect.All2All(), U=0.1, tau_d=10, tau_f=100.)
    net = bp.Network(pre=neu1, syn=syn1, post=neu2)

    runner = bp.DSRunner(net, inputs=[('pre.input', 28.)], monitors=['syn.I', 'syn.u', 'syn.x'])
    runner.run(10.)
    self.assertTupleEqual(runner.mon['syn.I'].shape, (100, 25))
    self.assertTupleEqual(runner.mon['syn.u'].shape, (100, 25))
    self.assertTupleEqual(runner.mon['syn.x'].shape, (100, 25))