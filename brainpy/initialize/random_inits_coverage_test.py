# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy/initialize/random_inits.py``.

Target: the random weight initializers and their helper functions that the
existing ``random_inits_test.py`` left uncovered:

* ``calculate_gain`` -- every nonlinearity branch + the two error branches.
* ``_format_shape`` -- int / empty / nested / plain / multi-dim branches.
* ``Gamma`` / ``Exponential`` / ``TruncatedNormal`` -- never instantiated before.
* ``VarianceScaling`` -- the ``assert`` validation branches + ``__repr__``.
* ``Orthogonal`` -- the non-square ``n_rows < n_cols`` reshape branch + repr.
* ``DeltaOrthogonal`` -- the two ``ValueError`` branches (bad ndim, fan ordering).

Tiny shapes; verifies output shape/dtype and structural properties.
"""

import math
import unittest

import jax.numpy as jnp
import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.initialize.random_inits import (
    calculate_gain, _format_shape, _compute_fans,
    Normal, TruncatedNormal, Uniform, Gamma, Exponential,
    VarianceScaling, KaimingUniform, KaimingNormal,
    XavierUniform, XavierNormal, LecunUniform, LecunNormal,
    Orthogonal, DeltaOrthogonal,
)


class TestCalculateGain(unittest.TestCase):
    def test_linear_family(self):
        for nl in ['linear', 'conv1d', 'conv2d', 'conv3d',
                   'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d',
                   'sigmoid']:
            self.assertEqual(calculate_gain(nl), 1)

    def test_tanh(self):
        self.assertAlmostEqual(calculate_gain('tanh'), 5.0 / 3)

    def test_relu(self):
        self.assertAlmostEqual(calculate_gain('relu'), math.sqrt(2.0))

    def test_leaky_relu_default(self):
        self.assertAlmostEqual(calculate_gain('leaky_relu'),
                               math.sqrt(2.0 / (1 + 0.01 ** 2)))

    def test_leaky_relu_with_param(self):
        self.assertAlmostEqual(calculate_gain('leaky_relu', 0.2),
                               math.sqrt(2.0 / (1 + 0.2 ** 2)))

    def test_leaky_relu_int_param(self):
        # ints (other than bool) are valid negative slopes.
        self.assertAlmostEqual(calculate_gain('leaky_relu', 0),
                               math.sqrt(2.0))

    def test_leaky_relu_bad_param_raises(self):
        with self.assertRaises(ValueError):
            calculate_gain('leaky_relu', 'bad')

    def test_leaky_relu_bool_param_raises(self):
        # bool is explicitly excluded -> falls to the error branch.
        with self.assertRaises(ValueError):
            calculate_gain('leaky_relu', True)

    def test_selu(self):
        self.assertAlmostEqual(calculate_gain('selu'), 3.0 / 4)

    def test_unsupported_raises(self):
        with self.assertRaises(ValueError):
            calculate_gain('does-not-exist')


class TestFormatShape(unittest.TestCase):
    def test_int(self):
        self.assertEqual(_format_shape(5), (5,))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            _format_shape(())

    def test_nested_single(self):
        self.assertEqual(_format_shape([(2, 3)]), (2, 3))
        self.assertEqual(_format_shape([[2, 3]]), [2, 3])

    def test_plain_single(self):
        self.assertEqual(_format_shape([5]), [5])

    def test_multi(self):
        self.assertEqual(_format_shape((2, 3)), (2, 3))

    def test_compute_fans(self):
        fan_in, fan_out = _compute_fans((4, 8))
        self.assertEqual(fan_in, 4)
        self.assertEqual(fan_out, 8)


class TestGamma(unittest.TestCase):
    def setUp(self):
        bm.random.seed(0)

    def test_shape(self):
        g = Gamma(shape=2.0, scale=1.0, seed=1)
        out = g((3, 4))
        self.assertEqual(tuple(out.shape), (3, 4))
        # Gamma samples are strictly positive.
        self.assertTrue(bool(jnp.all(bm.as_jax(out) > 0)))

    def test_repr(self):
        self.assertIn('Gamma', repr(Gamma(shape=2.0)))


class TestExponential(unittest.TestCase):
    def setUp(self):
        bm.random.seed(0)

    def test_shape(self):
        e = Exponential(scale=2.0, seed=1)
        out = e((3, 4))
        self.assertEqual(tuple(out.shape), (3, 4))
        self.assertTrue(bool(jnp.all(bm.as_jax(out) >= 0)))

    def test_repr(self):
        self.assertIn('Exponential', repr(Exponential()))


class TestTruncatedNormal(unittest.TestCase):
    def setUp(self):
        bm.random.seed(0)

    def test_shape_and_bounds(self):
        tn = TruncatedNormal(loc=0., scale=1., lower=-2., upper=2., seed=1)
        out = bm.as_jax(tn((200,)))
        self.assertEqual(out.shape, (200,))
        self.assertTrue(bool(jnp.all(out >= -2.)))
        self.assertTrue(bool(jnp.all(out <= 2.)))

    def test_scale_must_be_positive(self):
        with self.assertRaises(AssertionError):
            TruncatedNormal(scale=-1.)

    def test_repr(self):
        self.assertIn('TruncatedNormal', repr(TruncatedNormal()))


class TestVarianceScalingValidation(unittest.TestCase):
    def test_bad_mode_assert(self):
        with self.assertRaises(AssertionError):
            VarianceScaling(1.0, 'bad-mode', 'normal')

    def test_bad_distribution_assert(self):
        with self.assertRaises(AssertionError):
            VarianceScaling(1.0, 'fan_in', 'bad-dist')

    def test_repr(self):
        vs = VarianceScaling(1.0, 'fan_in', 'normal')
        self.assertIn('VarianceScaling', repr(vs))

    def test_all_modes_run(self):
        bm.random.seed(0)
        for mode in ['fan_in', 'fan_out', 'fan_avg']:
            vs = VarianceScaling(2.0, mode, 'normal')
            self.assertEqual(tuple(vs((6, 8)).shape), (6, 8))

    def test_all_distributions_run(self):
        bm.random.seed(0)
        for dist in ['truncated_normal', 'normal', 'uniform']:
            vs = VarianceScaling(2.0, 'fan_in', dist)
            self.assertEqual(tuple(vs((6, 8)).shape), (6, 8))


class TestNormalUniformRepr(unittest.TestCase):
    def test_normal_repr(self):
        self.assertIn('Normal', repr(Normal()))

    def test_uniform_repr(self):
        self.assertIn('Uniform', repr(Uniform()))

    def test_uniform_bounds(self):
        bm.random.seed(0)
        u = Uniform(min_val=10., max_val=20.)
        out = bm.as_jax(u((500,)))
        self.assertTrue(bool(jnp.all(out >= 10.)))
        self.assertTrue(bool(jnp.all(out <= 20.)))


class TestKaimingXavierLecunDefaults(unittest.TestCase):
    """Construct the convenience subclasses with their default args."""

    def setUp(self):
        bm.random.seed(0)

    def test_subclasses_run(self):
        for cls in [KaimingUniform, KaimingNormal, XavierUniform,
                    XavierNormal, LecunUniform, LecunNormal]:
            init = cls()
            self.assertEqual(tuple(init((6, 8)).shape), (6, 8))


class TestOrthogonal(unittest.TestCase):
    def setUp(self):
        bm.random.seed(0)

    def test_square(self):
        o = Orthogonal()
        out = bm.as_jax(o((8, 8)))
        self.assertEqual(out.shape, (8, 8))
        # Columns should be (approximately) orthonormal.
        gram = out.T @ out
        self.assertTrue(bool(jnp.allclose(gram, jnp.eye(8), atol=1e-4)))

    def test_tall_matrix_nrows_gt_ncols(self):
        # axis=-1 -> n_rows = shape[-1] = 5, n_cols = 10 -> n_rows < n_cols path.
        o = Orthogonal(axis=-1)
        out = o((5, 10))
        self.assertEqual(tuple(out.shape), (5, 10))

    def test_wide_matrix_nrows_lt_ncols(self):
        o = Orthogonal(axis=-1)
        out = o((10, 5))
        self.assertEqual(tuple(out.shape), (10, 5))

    def test_scaled(self):
        o = Orthogonal(scale=2.0, axis=0)
        out = o((6, 6))
        self.assertEqual(tuple(out.shape), (6, 6))

    def test_repr(self):
        self.assertIn('Orthogonal', repr(Orthogonal()))


class TestDeltaOrthogonal(unittest.TestCase):
    def setUp(self):
        bm.random.seed(0)

    def test_3d_4d_5d(self):
        for shape in [(3, 4, 8), (3, 3, 4, 8), (3, 3, 3, 4, 8)]:
            out = DeltaOrthogonal()(shape)
            self.assertEqual(tuple(out.shape), shape)

    def test_bad_ndim_raises(self):
        with self.assertRaises(ValueError):
            DeltaOrthogonal()((5, 5))

    def test_fan_in_gt_fan_out_raises(self):
        # shape[-1] (4) < shape[-2] (5) -> fan_in > fan_out error branch.
        with self.assertRaises(ValueError):
            DeltaOrthogonal()((3, 5, 4))

    def test_repr(self):
        self.assertIn('DeltaOrthogonal', repr(DeltaOrthogonal()))


if __name__ == '__main__':
    unittest.main()
