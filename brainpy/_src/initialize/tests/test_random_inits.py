# -*- coding: utf-8 -*-

import unittest

import brainpy as bp


class TestNormalInit(unittest.TestCase):
  def test_normal_init1(self):
    init = bp.init.Normal()
    for size in [(100,), (10, 20), (10, 20, 30)]:
      weights = init(size)
      assert weights.shape == size

  def test_normal_init2(self):
    init = bp.init.Normal(scale=0.5)
    for size in [(100,), (10, 20)]:
      weights = init(size)
      assert weights.shape == size

  def test_normal_init3(self):
    init1 = bp.init.Normal(scale=0.5, seed=10)
    init2 = bp.init.Normal(scale=0.5, seed=10)
    size = (10,)
    weights1 = init1(size)
    weights2 = init2(size)
    assert weights1.shape == size
    assert (weights1 == weights2).all()


class TestUniformInit(unittest.TestCase):
  def test_uniform_init1(self):
    init = bp.init.Normal()
    for size in [(100,), (10, 20), (10, 20, 30)]:
      weights = init(size)
      assert weights.shape == size

  def test_uniform_init2(self):
    init = bp.init.Uniform(min_val=10, max_val=20)
    for size in [(100,), (10, 20)]:
      weights = init(size)
      assert weights.shape == size


class TestVarianceScaling(unittest.TestCase):
  def test_var_scaling1(self):
    init = bp.init.VarianceScaling(scale=1., mode='fan_in', distribution='truncated_normal')
    for size in [(10, 20), (10, 20, 30)]:
      weights = init(size)
      assert weights.shape == size

  def test_var_scaling2(self):
    init = bp.init.VarianceScaling(scale=2, mode='fan_out', distribution='normal')
    for size in [(10, 20), (10, 20, 30)]:
      weights = init(size)
      assert weights.shape == size

  def test_var_scaling3(self):
    init = bp.init.VarianceScaling(scale=2 / 4, mode='fan_avg', in_axis=0, out_axis=1,
                                   distribution='uniform')
    for size in [(10, 20), (10, 20, 30)]:
      weights = init(size)
      assert weights.shape == size


class TestKaimingUniformUnit(unittest.TestCase):
  def test_kaiming_uniform_init(self):
    init = bp.init.KaimingUniform()
    for size in [(10, 20), (10, 20, 30)]:
      weights = init(size)
      assert weights.shape == size


class TestKaimingNormalUnit(unittest.TestCase):
  def test_kaiming_normal_init(self):
    init = bp.init.KaimingNormal()
    for size in [(10, 20), (10, 20, 30)]:
      weights = init(size)
      assert weights.shape == size


class TestXavierUniformUnit(unittest.TestCase):
  def test_xavier_uniform_init(self):
    init = bp.init.XavierUniform()
    for size in [(10, 20), (10, 20, 30)]:
      weights = init(size)
      assert weights.shape == size


class TestXavierNormalUnit(unittest.TestCase):
  def test_xavier_normal_init(self):
    init = bp.init.XavierNormal()
    for size in [(10, 20), (10, 20, 30)]:
      weights = init(size)
      assert weights.shape == size


class TestLecunUniformUnit(unittest.TestCase):
  def test_lecun_uniform_init(self):
    init = bp.init.LecunUniform()
    for size in [(10, 20), (10, 20, 30)]:
      weights = init(size)
      assert weights.shape == size


class TestLecunNormalUnit(unittest.TestCase):
  def test_lecun_normal_init(self):
    init = bp.init.LecunNormal()
    for size in [(10, 20), (10, 20, 30)]:
      weights = init(size)
      assert weights.shape == size


class TestOrthogonalUnit(unittest.TestCase):
  def test_orthogonal_init1(self):
    init = bp.init.Orthogonal()
    for size in [(20, 20), (10, 20, 30)]:
      weights = init(size)
      assert weights.shape == size

  def test_orthogonal_init2(self):
    init = bp.init.Orthogonal(scale=2., axis=0)
    for size in [(10, 20), (10, 20, 30)]:
      weights = init(size)
      assert weights.shape == size


class TestDeltaOrthogonalUnit(unittest.TestCase):
  def test_delta_orthogonal_init1(self):
    init = bp.init.DeltaOrthogonal()
    for size in [(20, 20, 20), (10, 20, 30, 40), (50, 40, 30, 20, 20)]:
      weights = init(size)
      assert weights.shape == size
