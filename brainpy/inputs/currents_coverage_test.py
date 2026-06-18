# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy/inputs/currents.py``.

Target: the input-current generators not exercised by ``currents_test.py``:

* the deprecated thin wrappers ``constant_current`` / ``spike_current`` /
  ``ramp_current`` (each emits a ``DeprecationWarning`` and delegates), and
* ``sinusoidal_input`` / ``square_input`` (the existing test commented these
  out because they require unit-aware ``frequency``/``duration`` arguments).

Tiny durations; verifies output shape and warning emission.

NOTE: ``brainpy.inputs.sinusoidal_input`` / ``square_input`` forward to
``braintools.input`` which *requires* ``frequency`` to be a ``brainunit`` Hz
quantity and ``duration``/``dt``/``t_start``/``t_end`` to be time quantities;
passing the documented bare floats (as the brainpy docstrings show, e.g.
``frequency=2.0``) raises ``AssertionError``/``UnitMismatchError``.  The
brainpy wrapper signature still advertises plain floats, so the docstring
examples are stale.  Tests therefore pass ``brainunit`` quantities.
"""

import unittest
import warnings

import numpy as np

import brainpy as bp
import brainpy.math as bm

try:
    import brainunit as u
    HAS_BRAINUNIT = True
except Exception:  # pragma: no cover - brainunit is a hard dep in practice
    HAS_BRAINUNIT = False


class TestDeprecatedWrappers(unittest.TestCase):
    def test_constant_current_warns_and_delegates(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            current, duration = bp.inputs.constant_current([(0, 100), (1, 100)])
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
        self.assertEqual(duration, 200)

    def test_spike_current_warns(self):
        # NOTE: ``spike_current`` is documented as a spike-input shim but its
        # body actually delegates to ``constant_input`` (copy/paste defect in
        # the deprecation wrapper), so it must be fed constant_input-style
        # ``[(value, duration), ...]`` pairs rather than spike arguments.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            out = bp.inputs.spike_current([(0, 50), (1, 50)])
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
        self.assertIsNotNone(out)

    def test_ramp_current_warns(self):
        # NOTE: like ``spike_current``, ``ramp_current`` also delegates to
        # ``constant_input`` rather than ``ramp_input`` (same wrapper defect),
        # so it expects ``[(value, duration), ...]`` pairs.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            out = bp.inputs.ramp_current([(0, 50), (1, 50)])
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
        self.assertIsNotNone(out)


@unittest.skipUnless(HAS_BRAINUNIT, 'brainunit required for unit-aware inputs')
class TestOscillatoryInputs(unittest.TestCase):
    def test_sinusoidal_input(self):
        out = bp.inputs.sinusoidal_input(
            amplitude=1., frequency=10. * u.Hz, duration=100. * u.ms,
            dt=0.1 * u.ms, t_start=0. * u.ms)
        self.assertEqual(out.shape[0], 1000)

    def test_sinusoidal_input_bias_and_window(self):
        out = bp.inputs.sinusoidal_input(
            amplitude=1., frequency=10. * u.Hz, duration=100. * u.ms,
            dt=0.1 * u.ms, bias=True, t_start=10. * u.ms, t_end=90. * u.ms)
        self.assertEqual(out.shape[0], 1000)

    def test_square_input(self):
        out = bp.inputs.square_input(
            amplitude=1., frequency=10. * u.Hz, duration=100. * u.ms,
            dt=0.1 * u.ms, t_start=0. * u.ms)
        self.assertEqual(out.shape[0], 1000)

    def test_square_input_bias_and_window(self):
        out = bp.inputs.square_input(
            amplitude=1., frequency=10. * u.Hz, duration=100. * u.ms,
            dt=0.1 * u.ms, bias=True, t_start=10. * u.ms, t_end=90. * u.ms)
        self.assertEqual(out.shape[0], 1000)


class TestCoreGeneratorsExtra(unittest.TestCase):
    """A couple of extra structural checks on the non-deprecated generators
    (these are smoke-covered elsewhere, but we assert numeric properties)."""

    def setUp(self):
        bm.random.seed(0)

    def test_wiener_process_shape(self):
        out = bp.inputs.wiener_process(50., n=3, t_start=10., t_end=40.)
        n_step = int(50. / bm.get_dt())
        self.assertEqual(out.shape, (n_step, 3))

    def test_ou_process_shape(self):
        out = bp.inputs.ou_process(mean=1., sigma=0.1, tau=10., duration=50.,
                                   n=2, t_start=5., t_end=45.)
        n_step = int(50. / bm.get_dt())
        self.assertEqual(out.shape, (n_step, 2))


if __name__ == '__main__':
    unittest.main()
