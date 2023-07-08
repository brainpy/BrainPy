import brainpy as bp

import unittest


class TestParamDesc(unittest.TestCase):
  def test1(self):
    a = bp.dyn.Expon(1)
    self.assertTrue(not isinstance(a, bp.mixin.ParamDesc[bp.dyn.Expon]))
    self.assertTrue(not isinstance(a, bp.mixin.ParamDesc[bp.DynamicalSystem]))

  def test2(self):
    a = bp.dyn.Expon.desc(1)
    self.assertTrue(isinstance(a, bp.mixin.ParamDesc[bp.dyn.Expon]))
    self.assertTrue(isinstance(a, bp.mixin.ParamDesc[bp.DynamicalSystem]))


class TestJointType(unittest.TestCase):
  def test1(self):
    T = bp.mixin.JointType[bp.DynamicalSystem]
    self.assertTrue(isinstance(bp.dnn.Layer(), T))

    T = bp.mixin.JointType[bp.DynamicalSystem, bp.mixin.ParamDesc]
    self.assertTrue(isinstance(bp.dyn.Expon(1), T))

  def test2(self):
    T = bp.mixin.JointType[bp.DynamicalSystem, bp.mixin.ParamDesc]
    self.assertTrue(not isinstance(bp.dyn.Expon(1), bp.mixin.ParamDesc[T]))
    self.assertTrue(isinstance(bp.dyn.Expon.desc(1), bp.mixin.ParamDesc[T]))

