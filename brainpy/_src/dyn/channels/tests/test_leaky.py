# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
from absl.testing import parameterized
from brainpy._src.dyn.channels import leaky

class Test_Leaky(parameterized.TestCase):
  bm.random.seed(1234)
  def test_leaky(self):
    class Neuron(bp.CondNeuGroup):
      def __init__(self, size):
        super(Neuron, self).__init__(size, V_initializer=bp.init.Uniform(-70, -50.))
        self.leaky1 = leaky.IL(size)
        self.leaky2 = leaky.IKL(size)

    model = Neuron(1)
    runner = bp.DSRunner(model,
                         monitors=['V'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))