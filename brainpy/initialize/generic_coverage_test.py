# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy/initialize/generic.py``.

Target: the parameter/variable dispatch helpers
(``parameter``, ``variable``, ``variable_``, ``noise``, ``delay``) and the
small predicate helpers (``_is_scalar``, ``_check_var``).  These convert
callables / arrays / scalars into parameters or :class:`~brainpy.math.Variable`
instances and own a number of error branches that the existing suite never
touched (the module sat at 15% line coverage).

Exercises tiny shapes, every batch/mode branch of ``variable`` and the
``ValueError`` paths of ``parameter`` and ``delay``.
"""

import unittest

import jax.numpy as jnp
import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.initialize.generic import (
    parameter, variable, variable_, noise, delay,
    _is_scalar, _check_var, _check_none,
)


class TestHelpers(unittest.TestCase):
    def test_is_scalar(self):
        for v in (1, 1.0, True, 1 + 2j):
            self.assertTrue(_is_scalar(v))
        for v in ([1], (1,), np.ones(2), bm.ones(2)):
            self.assertFalse(_is_scalar(v))

    def test_check_var_sets_ready_to_trace(self):
        v = bm.Variable(bm.zeros(3))
        out = _check_var(v)
        self.assertIs(out, v)
        self.assertTrue(v.ready_to_trace)

    def test_check_var_passthrough_non_variable(self):
        arr = bm.ones(3)
        self.assertIs(_check_var(arr), arr)

    def test_check_none_is_noop(self):
        # _check_none is currently a stub (pass); call it for coverage.
        self.assertIsNone(_check_none(None))
        self.assertIsNone(_check_none(5, allow_none=True))


class TestParameter(unittest.TestCase):
    def setUp(self):
        bm.random.seed(42)

    def test_none_allowed(self):
        self.assertIsNone(parameter(None, (3,), allow_none=True))

    def test_none_not_allowed_raises(self):
        with self.assertRaises(ValueError):
            parameter(None, (3,), allow_none=False)

    def test_scalar_returned_directly(self):
        self.assertEqual(parameter(2.5, (3,)), 2.5)
        self.assertEqual(parameter(7, (3,)), 7)
        self.assertEqual(parameter(True, (3,)), True)

    def test_scalar_disallowed_falls_through_to_error(self):
        # allow_scalar=False with a raw python scalar -> not callable, not array
        with self.assertRaises(ValueError):
            parameter(2.5, (3,), allow_scalar=False)

    def test_callable_initializer(self):
        out = parameter(bp.init.Normal(), (4,))
        self.assertEqual(out.shape, (4,))

    def test_numpy_array_matching_shape(self):
        out = parameter(np.ones((2, 3)), (2, 3))
        self.assertEqual(tuple(out.shape), (2, 3))

    def test_jnp_array_matching_shape(self):
        out = parameter(jnp.ones((5,)), (5,))
        self.assertEqual(tuple(out.shape), (5,))

    def test_scalar_shaped_array_returned(self):
        # shape () passes the allow_scalar early-return.
        out = parameter(np.array(3.0), (4,))
        self.assertEqual(tuple(out.shape), ())
        # shape (1,) also short-circuits.
        out1 = parameter(np.array([3.0]), (4,))
        self.assertEqual(tuple(out1.shape), (1,))

    def test_bm_array_passthrough(self):
        out = parameter(bm.ones((2, 3)), (2, 3))
        self.assertEqual(tuple(out.shape), (2, 3))

    def test_variable_passthrough(self):
        v = bm.Variable(bm.ones((2, 3)))
        out = parameter(v, (2, 3))
        self.assertIsInstance(out, bm.Variable)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            parameter('not-a-param', (3,), allow_scalar=False)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            parameter(np.ones((4,)), (3,), allow_scalar=False)


class TestVariableCallable(unittest.TestCase):
    def setUp(self):
        bm.random.seed(0)

    def test_callable_none_batch(self):
        v = variable(bm.zeros, None, sizes=(4,))
        self.assertEqual(v.shape, (4,))
        self.assertIsInstance(v, bm.Variable)

    def test_callable_false_batch(self):
        v = variable(bm.zeros, False, sizes=(4,))
        self.assertEqual(v.shape, (4,))

    def test_callable_int_batch(self):
        v = variable(bm.zeros, 3, sizes=(4,))
        self.assertEqual(v.shape, (3, 4))

    def test_callable_nonbatching_mode(self):
        v = variable(bm.zeros, bm.NonBatchingMode(), sizes=(4,))
        self.assertEqual(v.shape, (4,))

    def test_callable_batching_mode(self):
        v = variable(bm.zeros, bm.BatchingMode(5), sizes=(4,))
        self.assertEqual(v.shape, (5, 4))

    def test_callable_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            variable(bm.zeros, object(), sizes=(4,))

    def test_callable_none_sizes_raises(self):
        with self.assertRaises(ValueError):
            variable(bm.zeros, None, sizes=None)

    def test_axis_names_no_batch(self):
        v = variable(bm.zeros, None, sizes=(4,), axis_names=['x'])
        self.assertEqual(v.shape, (4,))

    def test_axis_names_with_batch_inserts_name(self):
        v = variable(bm.zeros, 2, sizes=(4,), axis_names=['x'], batch_axis_name='b')
        self.assertEqual(v.shape, (2, 4))


class TestVariableData(unittest.TestCase):
    def setUp(self):
        bm.random.seed(0)

    def test_data_none_batch(self):
        v = variable(bm.ones((4,)), None)
        self.assertEqual(v.shape, (4,))

    def test_data_false_batch(self):
        v = variable(bm.ones((4,)), False)
        self.assertEqual(v.shape, (4,))

    def test_data_int_batch_repeats(self):
        v = variable(bm.ones((4,)), 3, batch_axis=0)
        self.assertEqual(v.shape, (3, 4))

    def test_data_nonbatching_mode(self):
        v = variable(bm.ones((4,)), bm.NonBatchingMode())
        self.assertEqual(v.shape, (4,))

    def test_data_batching_mode_repeats(self):
        v = variable(bm.ones((4,)), bm.BatchingMode(3), batch_axis=0)
        self.assertEqual(v.shape, (3, 4))

    def test_data_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            variable(bm.ones((4,)), object())

    def test_data_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            variable(bm.ones((4,)), None, sizes=(5,))

    def test_data_shape_match_ok(self):
        v = variable(bm.ones((4,)), None, sizes=(4,))
        self.assertEqual(v.shape, (4,))


class TestVariableUnderscore(unittest.TestCase):
    def test_variable_underscore_delegates(self):
        v = variable_(bm.zeros, sizes=(4,))
        self.assertEqual(v.shape, (4,))
        self.assertIsInstance(v, bm.Variable)


class TestNoise(unittest.TestCase):
    def setUp(self):
        bm.random.seed(0)

    def test_none_noise(self):
        self.assertIsNone(noise(None, (3,)))

    def test_callable_returned_directly(self):
        f = lambda *a, **k: 1.0
        self.assertIs(noise(f, (3,)), f)

    def test_scalar_noise_returns_callable(self):
        nf = noise(0.5, (3,))
        self.assertTrue(callable(nf))
        self.assertEqual(nf(), 0.5)

    def test_array_noise_multi_vars(self):
        nf = noise(bm.ones((3,)), (3,), num_vars=2, noise_idx=1)
        res = nf()
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsNone(res[0])
        self.assertEqual(res[1].shape, (3,))


class TestDelay(unittest.TestCase):
    def setUp(self):
        bm.random.seed(0)

    def test_delay_none(self):
        dtype, step, delays = delay(None, bm.ones((5,)))
        self.assertEqual(dtype, 'none')
        self.assertIsNone(delays)

    def test_delay_homo_int(self):
        dtype, step, delays = delay(3, bm.ones((5,)))
        self.assertEqual(dtype, 'homo')
        self.assertIsInstance(delays, bm.LengthDelay)

    def test_delay_heter_array(self):
        steps = bm.ones(5, dtype=bm.int32)
        dtype, step, delays = delay(steps, bm.ones((5,)))
        self.assertEqual(dtype, 'heter')
        self.assertIsInstance(delays, bm.LengthDelay)

    def test_delay_callable(self):
        def cb(shape, dtype=None):
            return bm.ones(shape, dtype=bm.int32)
        dtype, step, delays = delay(cb, bm.ones((5,)))
        self.assertEqual(dtype, 'heter')

    def test_delay_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            delay('bad', bm.ones((5,)))

    def test_delay_heter_bad_dtype_raises(self):
        steps = bm.ones(5, dtype=bm.float32)
        with self.assertRaises(ValueError):
            delay(steps, bm.ones((5,)))

    def test_delay_heter_axis0_mismatch_raises(self):
        steps = bm.ones(4, dtype=bm.int32)
        with self.assertRaises(ValueError):
            delay(steps, bm.ones((5,)))

    def test_delay_heter_size_mismatch_raises(self):
        # axis-0 matches but the total size does not (2D target).
        steps = bm.ones(5, dtype=bm.int32)
        with self.assertRaises(ValueError):
            delay(steps, bm.ones((5, 3)))


if __name__ == '__main__':
    unittest.main()
