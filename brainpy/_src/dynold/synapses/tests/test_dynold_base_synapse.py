
import unittest
import brainpy as bp


class Test_TwoEndConnAlignPre(unittest.TestCase):
  def test1(self):
    E = bp.neurons.HH(size=4)
    syn = bp.synapses.AMPA(E, E, bp.conn.All2All(include_self=False))
    self.assertTrue(syn.conn.include_self == syn.comm.include_self)


