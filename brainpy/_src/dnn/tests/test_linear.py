import pytest
from absl.testing import absltest
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm

from brainpy._src.dependency_check import import_taichi

if import_taichi(error_if_not_found=False) is None:
  pytest.skip('no taichi', allow_module_level=True)


class TestLinear(parameterized.TestCase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    bm.random.seed()

  @parameterized.product(
    size=[(10,),
          (20, 10),
          (5, 8, 10)],
    num_out=[20,]
  )
  def test_Dense1(self, size, num_out):
    bm.random.seed()
    f = bp.dnn.Linear(10, num_out)
    x = bm.random.random(size)
    y = f(x)
    self.assertTrue(y.shape == size[:-1] + (num_out,))
    bm.clear_buffer_memory()

  @parameterized.product(
    size=[(10,),
          (20, 10),
          (5, 8, 10)],
  )
  def test_Identity(self, size):
    bm.random.seed()
    f = bp.dnn.Identity()
    x = bm.random.random(size)
    y = f(x)
    self.assertTrue(y.shape == size)
    bm.clear_buffer_memory()

  def test_AllToAll1(self):
    bm.random.seed()
    with bm.environment(mode=bm.BatchingMode()):
      f = bp.dnn.AllToAll(10, 20, weight=.1, include_self=True)
      x = bm.random.random((8, 10))
      y = f(x)
      expected = bm.sum(x, axis=1, keepdims=True) * 0.1
      self.assertTrue(bm.allclose(y, expected))

    with bm.environment(mode=bm.NonBatchingMode()):
      f = bp.dnn.AllToAll(10, 20, weight=.1, include_self=True)
      x = bm.random.random((10,))
      y = f(x)
      expected = bm.sum(x, keepdims=True) * 0.1
      self.assertTrue(bm.allclose(y, expected))
    bm.clear_buffer_memory()

  def test_OneToOne(self):
    bm.random.seed()
    with bm.environment(mode=bm.BatchingMode()):
      f = bp.dnn.OneToOne(10, weight=.1)
      x = bm.random.random((8, 10))
      y = f(x)
      expected = x * 0.1
      self.assertTrue(bm.allclose(y, expected))

    with bm.environment(mode=bm.NonBatchingMode()):
      f = bp.dnn.OneToOne(10, weight=.1)
      x = bm.random.random((10,))
      y = f(x)
      expected = x * 0.1
      self.assertTrue(bm.allclose(y, expected))
    bm.clear_buffer_memory()

  @parameterized.product(
    conn=[
      # bp.conn.FixedProb(0.1, pre=100, post=100),
      bp.conn.GridFour(pre=100, post=100),
      bp.conn.GaussianProb(0.1, pre=100, post=100),
    ]
  )
  def test_MaskedLinear(self, conn):
    bm.random.seed()
    bm.random.DEFAULT.seed(123)
    f = bp.dnn.MaskedLinear(conn, weight=bp.init.XavierNormal(seed=123))
    x = bm.random.random((16, 100))
    y = f(x)
    self.assertTrue(y.shape == (16, 100))
    bm.clear_buffer_memory()

  @parameterized.product(
    conn=[
      bp.conn.FixedProb(0.1, pre=100, post=100),
      bp.conn.GridFour(pre=100, post=100),
      bp.conn.GaussianProb(0.1, pre=100, post=100),
    ]
  )
  def test_CSRLinear(self, conn):
    bm.random.seed()
    f = bp.dnn.CSRLinear(conn, weight=bp.init.Normal())
    x = bm.random.random((16, 100))
    y = f(x)
    self.assertTrue(y.shape == (16, 100))

    x = bm.random.random((100,))
    y = f(x)
    self.assertTrue(y.shape == (100,))
    bm.clear_buffer_memory()

  @parameterized.product(
    conn=[
      bp.conn.FixedProb(0.1, pre=100, post=100),
      bp.conn.GridFour(pre=100, post=100),
      bp.conn.GaussianProb(0.1, pre=100, post=100),
    ]
  )
  def test_EventCSRLinear(self, conn):
    bm.random.seed()
    f = bp.layers.EventCSRLinear(conn, weight=bp.init.Normal())
    x = bm.random.random((16, 100))
    y = f(x)
    self.assertTrue(y.shape == (16, 100))
    x = bm.random.random((100,))
    y = f(x)
    self.assertTrue(y.shape == (100,))
    bm.clear_buffer_memory()

  @parameterized.product(
    prob=[0.1],
    weight=[0.01],
    shape=[(), (10,), (10, 20), (10, 20, 25)]
  )
  def test_JitFPHomoLinear(self, prob, weight, shape):
    bm.random.seed()
    f = bp.dnn.JitFPHomoLinear(100, 200, prob, weight, seed=123)
    x = bm.random.random(shape + (100,))
    y = f(x)
    self.assertTrue(y.shape == shape + (200,))
    bm.clear_buffer_memory()

  @parameterized.product(
    prob=[0.1],
    w_low=[-0.01, ],
    w_high=[0.01, ],
    shape=[(), (10,), (10, 20), (10, 20, 25)]
  )
  def test_JitFPUniformLinear(self, prob, w_low, w_high, shape):
    bm.random.seed()
    f = bp.dnn.JitFPUniformLinear(100, 200, prob, w_low, w_high, seed=123)
    x = bm.random.random(shape + (100,))
    y = f(x)
    self.assertTrue(y.shape == shape + (200,))
    bm.clear_buffer_memory()

  @parameterized.product(
    prob=[0.1],
    w_mu=[-0.01],
    w_sigma=[0.01],
    shape=[(), (10,), (10, 20), (10, 20, 25)]
  )
  def test_JitFPNormalLinear(self, prob, w_mu, w_sigma, shape):
    bm.random.seed()
    f = bp.dnn.JitFPNormalLinear(100, 200, prob, w_mu, w_sigma, seed=123)
    x = bm.random.random(shape + (100,))
    y = f(x)
    self.assertTrue(y.shape == shape + (200,))
    bm.clear_buffer_memory()

  @parameterized.product(
    prob=[0.1],
    weight=[0.01,],
    shape=[(), (10,), (10, 20), (10, 20, 25)]
  )
  def test_EventJitFPHomoLinear(self, prob, weight, shape):
    bm.random.seed()
    f = bp.dnn.EventJitFPHomoLinear(100, 200, prob, weight, seed=123)
    y = f(bm.random.random(shape + (100,)) < 0.1)
    self.assertTrue(y.shape == shape + (200,))

    y2 = f(bm.as_jax(bm.random.random(shape + (100,)) < 0.1, dtype=float))
    self.assertTrue(y2.shape == shape + (200,))
    bm.clear_buffer_memory()

  @parameterized.product(
    prob=[0.1],
    w_low=[-0.01],
    w_high=[0.01],
    shape=[(), (10,), (10, 20), (10, 20, 25)]
  )
  def test_EventJitFPUniformLinear(self, prob, w_low, w_high, shape):
    bm.random.seed()
    f = bp.dnn.EventJitFPUniformLinear(100, 200, prob, w_low, w_high, seed=123)
    y = f(bm.random.random(shape + (100,)) < 0.1)
    self.assertTrue(y.shape == shape + (200,))

    y2 = f(bm.as_jax(bm.random.random(shape + (100,)) < 0.1, dtype=float))
    self.assertTrue(y2.shape == shape + (200,))
    bm.clear_buffer_memory()

  @parameterized.product(
    prob=[0.1],
    w_mu=[-0.01],
    w_sigma=[0.01],
    shape=[(), (10,), (10, 20), (10, 20, 25)]
  )
  def test_EventJitFPNormalLinear(self, prob, w_mu, w_sigma, shape):
    bm.random.seed()
    f = bp.dnn.EventJitFPNormalLinear(100, 200, prob, w_mu, w_sigma, seed=123)
    y = f(bm.random.random(shape + (100,)) < 0.1)
    self.assertTrue(y.shape == shape + (200,))

    y2 = f(bm.as_jax(bm.random.random(shape + (100,)) < 0.1, dtype=float))
    self.assertTrue(y2.shape == shape + (200,))
    bm.clear_buffer_memory()


if __name__ == '__main__':
  absltest.main()
