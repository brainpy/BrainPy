# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
from absl.testing import parameterized
from brainpy._src.dyn.channels import IH, Ca


class Test_IH(parameterized.TestCase):
  bm.random.seed(1234)
  def test_IH(self):
    class Neuron(bp.CondNeuGroup):
      def __init__(self, size):
        super(Neuron, self).__init__(size)
        self.IH = IH.Ih_HM1992(size)
        self.Ca = Ca.CalciumDetailed(size, IH=IH.Ih_De1996(size))

    model = Neuron(1)
    runner = bp.DSRunner(model,
                         monitors=['V', 'IH.p', 'Ca.IH.O'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['IH.p'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['Ca.IH.O'].shape, (100, 1))