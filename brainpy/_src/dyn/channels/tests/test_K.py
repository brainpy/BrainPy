# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
from absl.testing import parameterized
from brainpy._src.dyn.channels import K

class Test_K(parameterized.TestCase):
  bm.random.seed(1234)
  def test_K(self):
    class Neuron(bp.CondNeuGroup):
      def __init__(self, size):
        super(Neuron, self).__init__(size, V_initializer=bp.init.Uniform(-70, -50.))
        self.IK_1 = K.IKDR_Ba2002(size)
        self.IK_2 = K.IK_TM1991(size)
        self.IK_3 = K.IK_HH1952(size)
        self.IK_4 = K.IKA1_HM1992(size)
        self.IK_5 = K.IKA2_HM1992(size)
        self.IK_6 = K.IKK2A_HM1992(size)
        self.IK_7 = K.IKK2B_HM1992(size)
        self.IK_8 = K.IKNI_Ya1989(size)

    model = Neuron(1)
    runner = bp.DSRunner(model,
                         monitors=['V', 'IK_1.p', 'IK_2.p', 'IK_3.p', 'IK_4.p', 'IK_5.p', 'IK_6.p', 'IK_7.p', 'IK_8.p'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['IK_1.p'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['IK_2.p'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['IK_3.p'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['IK_4.p'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['IK_5.p'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['IK_6.p'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['IK_7.p'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['IK_8.p'].shape, (100, 1))