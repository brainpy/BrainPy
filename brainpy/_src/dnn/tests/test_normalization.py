from absl.testing import parameterized
from absl.testing import absltest

import brainpy as bp
import brainpy.math as bm


class Test_Normalization(parameterized.TestCase):
  @parameterized.product(
    fit=[True, False],
  )
  def test_BatchNorm1d(self, fit):
    bm.random.seed()
    net = bp.dnn.BatchNorm1d(num_features=10, mode=bm.training_mode)
    bp.share.save(fit=fit)
    input = bm.random.randn(1, 3, 10)
    output = net(input)
    bm.clear_buffer_memory()

  @parameterized.product(
    fit=[True, False]
  )
  def test_BatchNorm2d(self, fit):
    bm.random.seed()
    net = bp.dnn.BatchNorm2d(10, mode=bm.training_mode)
    bp.share.save(fit=fit)
    input = bm.random.randn(1, 3, 4, 10)
    output = net(input)
    bm.clear_buffer_memory()

  @parameterized.product(
    fit=[True, False]
  )
  def test_BatchNorm3d(self, fit):
    bm.random.seed()
    net = bp.dnn.BatchNorm3d(10, mode=bm.training_mode)
    bp.share.save(fit=fit)
    input = bm.random.randn(1, 3, 4, 5, 10)
    output = net(input)
    bm.clear_buffer_memory()

  @parameterized.product(
    normalized_shape=(10, [5, 10])
  )
  def test_LayerNorm(self, normalized_shape):
    bm.random.seed()
    net = bp.dnn.LayerNorm(normalized_shape, mode=bm.training_mode)
    input = bm.random.randn(20, 5, 10)
    output = net(input)
    bm.clear_buffer_memory()

  @parameterized.product(
    num_groups=[1, 2, 3, 6]
  )
  def test_GroupNorm(self, num_groups):
    bm.random.seed()
    input = bm.random.randn(20, 10, 10, 6)
    net = bp.dnn.GroupNorm(num_groups=num_groups, num_channels=6, mode=bm.training_mode)
    output = net(input)
    bm.clear_buffer_memory()

  def test_InstanceNorm(self):
    bm.random.seed()
    input = bm.random.randn(20, 10, 10, 6)
    net = bp.dnn.InstanceNorm(num_channels=6, mode=bm.training_mode)
    output = net(input)
    bm.clear_buffer_memory()


if __name__ == '__main__':
  absltest.main()
