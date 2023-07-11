# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
from absl.testing import parameterized


class Test_Leaky(parameterized.TestCase):
  bm.random.seed(1234)

  def test_leaky(self):
    class Neuron(bp.dyn.CondNeuGroup):
      def __init__(self, size):
        super(Neuron, self).__init__(size, V_initializer=bp.init.Uniform(-70, -50.))
        self.leaky1 = bp.dyn.IL(size)
        self.leaky2 = bp.dyn.IKL(size)

    model = Neuron(1)
    runner = bp.DSRunner(model,
                         monitors=['V'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
