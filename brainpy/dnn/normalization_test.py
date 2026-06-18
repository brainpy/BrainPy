# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from absl.testing import absltest
from absl.testing import parameterized

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

    @parameterized.product(
        fit=[True, False]
    )
    def test_BatchNorm2d(self, fit):
        bm.random.seed()
        net = bp.dnn.BatchNorm2d(10, mode=bm.training_mode)
        bp.share.save(fit=fit)
        input = bm.random.randn(1, 3, 4, 10)
        output = net(input)

    @parameterized.product(
        fit=[True, False]
    )
    def test_BatchNorm3d(self, fit):
        bm.random.seed()
        net = bp.dnn.BatchNorm3d(10, mode=bm.training_mode)
        bp.share.save(fit=fit)
        input = bm.random.randn(1, 3, 4, 5, 10)
        output = net(input)

    @parameterized.product(
        normalized_shape=(10, [5, 10])
    )
    def test_LayerNorm(self, normalized_shape):
        bm.random.seed()
        net = bp.dnn.LayerNorm(normalized_shape, mode=bm.training_mode)
        input = bm.random.randn(20, 5, 10)
        output = net(input)

    @parameterized.product(
        num_groups=[1, 2, 3, 6]
    )
    def test_GroupNorm(self, num_groups):
        bm.random.seed()
        input = bm.random.randn(20, 10, 10, 6)
        net = bp.dnn.GroupNorm(num_groups=num_groups, num_channels=6, mode=bm.training_mode)
        output = net(input)

    def test_InstanceNorm(self):
        bm.random.seed()
        input = bm.random.randn(20, 10, 10, 6)
        net = bp.dnn.InstanceNorm(num_channels=6, mode=bm.training_mode)
        output = net(input)

    def test_LayerNorm_shape_mismatch_raises_valueerror(self):
        # Regression for P12-M1: the wrong-shape diagnostic used ``", ".join(<ints>)``
        # which raised ``TypeError`` and masked the intended ``ValueError``.
        net = bp.dnn.LayerNorm(10, mode=bm.training_mode)
        bad_input = bm.random.randn(2, 5, 8)  # last dim 8 != 10
        with self.assertRaises(ValueError):
            net(bad_input)

    def test_BatchNorm_running_var_is_unbiased(self):
        # Regression for P12-M3: the running variance buffer must use the unbiased
        # (Bessel-corrected, divisor N-1) batch variance, matching PyTorch, instead
        # of the biased (divisor N) variance used to normalize the current batch.
        import jax.numpy as jnp
        import numpy as np
        bm.random.seed(123)
        net = bp.dnn.BatchNorm1d(num_features=3, affine=False, mode=bm.training_mode)
        bp.share.save(fit=True)
        x = bm.random.randn(4, 5, 3) * 3.0 + 7.0  # N = 4*5 = 20 reduced elements
        net(x)

        xj = bm.as_jax(x)
        n = xj.shape[0] * xj.shape[1]
        biased = jnp.var(xj, axis=(0, 1))
        unbiased = biased * n / (n - 1)
        # After one update: running_var = 0.99 * 1.0 + 0.01 * <var>.
        expected_unbiased = 0.99 * 1.0 + 0.01 * unbiased
        expected_biased = 0.99 * 1.0 + 0.01 * biased
        rv = bm.as_jax(net.running_var.value)
        self.assertTrue(bool(jnp.allclose(rv, expected_unbiased, atol=1e-5)))
        # And it must NOT match the biased estimate (the previous behaviour).
        self.assertFalse(bool(jnp.allclose(rv, expected_biased, atol=1e-5)))

    def test_BatchNorm_batch_is_biased_normalized(self):
        # The normalization of the current batch itself must remain unit-variance
        # (biased), unaffected by the running-buffer correction.
        import jax.numpy as jnp
        bm.random.seed(7)
        net = bp.dnn.BatchNorm1d(num_features=3, affine=False, mode=bm.training_mode)
        bp.share.save(fit=True)
        x = bm.random.randn(8, 6, 3) * 2.0 - 1.0
        out = bm.as_jax(net(x))
        self.assertTrue(bool(jnp.allclose(out.mean(axis=(0, 1)), 0.0, atol=1e-5)))
        self.assertTrue(bool(jnp.allclose(out.var(axis=(0, 1)), 1.0, atol=1e-4)))


if __name__ == '__main__':
    absltest.main()
