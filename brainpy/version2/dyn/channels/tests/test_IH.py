# -*- coding: utf-8 -*-


from absl.testing import parameterized

import brainpy.version2 as bp
import brainpy.version2.math as bm


class Test_IH(parameterized.TestCase):
    bm.random.seed(1234)

    def test_IH(self):
        class Neuron(bp.dyn.CondNeuGroup):
            def __init__(self, size):
                super(Neuron, self).__init__(size)
                self.IH = bp.dyn.Ih_HM1992(size)
                self.Ca = bp.dyn.CalciumDetailed(size, IH=bp.dyn.Ih_De1996(size))

        model = Neuron(1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'IH.p', 'Ca.IH.O'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['IH.p'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['Ca.IH.O'].shape, (100, 1))
