# -*- coding: utf-8 -*-
"""Coverage tests for :mod:`brainpy.transform`.

Exercises :class:`brainpy.LoopOverTime`:

- construction validation (bad ``target`` type, bad ``out_vars``,
  ``data_first_axis`` assertion, deprecated ``remat`` warning,
  ``shared_arg`` handling, ``t0``/``i0`` toggles, default ``dt``).
- the float-duration forward path (with and without ``out_vars``).
- the PyTree-input path for non-batching mode.
- the batching-mode path with ``data_first_axis`` ``'T'`` and ``'B'``.
- the ``no_state=True`` path (stateless ANN layer) and its
  duration-not-allowed error.
- input-shape error branches (mismatched time length / batch size).
- ``reset_state`` restoring ``t0`` / ``i0``.
"""
import jax.numpy as jnp
import pytest

import brainpy as bp
import brainpy.math as bm


# --------------------------------------------------------------------------- #
# helper systems
# --------------------------------------------------------------------------- #
class _Tiny(bp.DynamicalSystem):
    """A minimal non-batching dynamical system with a single state."""

    def __init__(self, size=1, mode=None):
        super().__init__(mode=mode)
        self.size = size
        self.v = bm.Variable(bm.zeros(size))

    def update(self, x=None):
        x = 0. if x is None else x
        self.v.value = self.v.value + x
        return self.v.value

    def reset_state(self, batch_size=None):
        if batch_size is None:
            self.v.value = bm.zeros(self.size)
        else:
            self.v.value = bm.zeros((batch_size, self.size))


class _TinyBatch(bp.DynamicalSystem):
    """A minimal batching dynamical system."""

    def __init__(self, size=2, mode=None):
        super().__init__(mode=mode)
        self.size = size
        self.v = bm.Variable(bm.zeros((1, size)), batch_axis=0)

    def update(self, x=None):
        x = 0. if x is None else x
        self.v.value = self.v.value + x
        return self.v.value

    def reset_state(self, batch_size=1):
        self.v.value = bm.zeros((batch_size, self.size))


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_bad_target_type(self):
        with pytest.raises(TypeError):
            bp.LoopOverTime(target=object())

    def test_bad_data_first_axis(self):
        with pytest.raises(AssertionError):
            bp.LoopOverTime(_Tiny(1), data_first_axis='X')

    def test_remat_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match='remat'):
            bp.LoopOverTime(_Tiny(1), remat=True)

    def test_out_vars_bad(self):
        with pytest.raises(TypeError):
            bp.LoopOverTime(_Tiny(1), out_vars=[123])

    def test_shared_arg_dict(self):
        looper = bp.LoopOverTime(_Tiny(1), shared_arg={'fit': False}, dt=0.1)
        assert looper.shared_arg['fit'] is False
        assert looper.shared_arg['dt'] == 0.1

    def test_shared_arg_not_dict(self):
        with pytest.raises(AssertionError):
            bp.LoopOverTime(_Tiny(1), shared_arg=123, dt=0.1)

    def test_t0_i0_none(self):
        looper = bp.LoopOverTime(_Tiny(1), t0=None, i0=None, dt=0.1)
        assert looper.t0 is None
        assert looper.i0 is None

    def test_default_dt(self):
        # dt=None -> uses share.dt
        looper = bp.LoopOverTime(_Tiny(1), dt=None)
        assert looper.dt == bm.dt


# --------------------------------------------------------------------------- #
# float-duration forward
# --------------------------------------------------------------------------- #
class TestDurationForward:
    def test_duration_with_out_vars(self):
        with bp.math.environment(dt=0.1):
            t = _Tiny(1)
            looper = bp.LoopOverTime(t, out_vars=t.v)
            out, mon = looper(0.5)
            assert jnp.shape(out)[0] == 5
            assert jnp.shape(mon)[0] == 5

    def test_duration_no_out_vars(self):
        with bp.math.environment(dt=0.1):
            t = _Tiny(1)
            looper = bp.LoopOverTime(t)
            out = looper(0.5)
            assert jnp.shape(out)[0] == 5

    def test_duration_advances_t0_i0(self):
        with bp.math.environment(dt=0.1):
            t = _Tiny(1)
            looper = bp.LoopOverTime(t)
            looper(0.5)
            # i0 advanced by number of steps; t0 by steps*dt
            assert int(looper.i0.value) == 5
            assert float(looper.t0.value) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# pytree-input forward (non-batching)
# --------------------------------------------------------------------------- #
class TestInputForward:
    def test_input_with_out_vars(self):
        with bp.math.environment(dt=0.1):
            t = _Tiny(1)
            looper = bp.LoopOverTime(t, out_vars=t.v)
            out, mon = looper(bm.ones((5, 1)))
            assert jnp.shape(out) == (5, 1)
            assert jnp.shape(mon) == (5, 1)

    def test_input_no_out_vars(self):
        with bp.math.environment(dt=0.1):
            t = _Tiny(1)
            looper = bp.LoopOverTime(t)
            out = looper(bm.ones((5, 1)))
            assert jnp.shape(out) == (5, 1)

    def test_input_mismatched_time_length(self):
        with bp.math.environment(dt=0.1):
            t = _Tiny(1)
            looper = bp.LoopOverTime(t)
            with pytest.raises(ValueError):
                # two leaves with different time length
                looper([bm.ones((5, 1)), bm.ones((6, 1))])

    def test_input_advances_counters(self):
        with bp.math.environment(dt=0.1):
            t = _Tiny(1)
            looper = bp.LoopOverTime(t)
            looper(bm.ones((4, 1)))
            assert int(looper.i0.value) == 4
            assert float(looper.t0.value) == pytest.approx(0.4)

    def test_input_leaf_without_shape(self):
        # a leaf lacking ``.shape`` triggers the AttributeError -> ValueError
        # branch (non-batching path).
        with bp.math.environment(dt=0.1):
            t = _Tiny(1)
            looper = bp.LoopOverTime(t)
            with pytest.raises(ValueError):
                looper([object()])


# --------------------------------------------------------------------------- #
# batching mode
# --------------------------------------------------------------------------- #
class TestBatchingMode:
    def test_data_first_axis_T(self):
        with bp.math.environment(dt=0.1):
            mode = bm.BatchingMode(4)
            t = _TinyBatch(2, mode=mode)
            t.reset_state(4)
            looper = bp.LoopOverTime(t, data_first_axis='T')
            out = looper(bm.ones((5, 4, 2)))
            assert jnp.shape(out) == (5, 4, 2)

    def test_data_first_axis_B(self):
        with bp.math.environment(dt=0.1):
            mode = bm.BatchingMode(4)
            t = _TinyBatch(2, mode=mode)
            t.reset_state(4)
            looper = bp.LoopOverTime(t, data_first_axis='B')
            out = looper(bm.ones((4, 5, 2)))
            assert jnp.shape(out) == (5, 4, 2)

    def test_mismatched_batch(self):
        with bp.math.environment(dt=0.1):
            mode = bm.BatchingMode(4)
            t = _TinyBatch(2, mode=mode)
            t.reset_state(4)
            looper = bp.LoopOverTime(t, data_first_axis='T')
            with pytest.raises(ValueError):
                looper([bm.ones((5, 4, 2)), bm.ones((5, 3, 2))])

    def test_mismatched_time(self):
        # NOTE (defect, transform.py:241-244): after computing ``length`` the
        # batching branch re-checks ``if len(batch) != 1`` instead of
        # ``len(length) != 1``. The duplicate batch-check at line 241 is dead
        # (batch was already validated at line 233), so the "same batch size"
        # message at 242-244 is unreachable. A genuine time-length mismatch is
        # still caught by the subsequent ``if len(length) != 1`` (line 245),
        # which is what raises here.
        with bp.math.environment(dt=0.1):
            mode = bm.BatchingMode(4)
            t = _TinyBatch(2, mode=mode)
            t.reset_state(4)
            looper = bp.LoopOverTime(t, data_first_axis='T')
            with pytest.raises(ValueError):
                looper([bm.ones((5, 4, 2)), bm.ones((6, 4, 2))])

    def test_batch_leaf_without_shape(self):
        # leaf lacking ``.shape`` -> AttributeError -> ValueError (batching path,
        # x.shape[b_idx] access at the first try-block).
        with bp.math.environment(dt=0.1):
            mode = bm.BatchingMode(4)
            t = _TinyBatch(2, mode=mode)
            t.reset_state(4)
            looper = bp.LoopOverTime(t, data_first_axis='T')
            with pytest.raises(ValueError):
                looper([object()])


# --------------------------------------------------------------------------- #
# no_state (stateless ANN)
# --------------------------------------------------------------------------- #
class TestNoState:
    def test_no_state_forward(self):
        with bp.math.environment(dt=0.1):
            mode = bm.BatchingMode(4)
            dense = bp.layers.Dense(3, 2, mode=mode)
            looper = bp.LoopOverTime(dense, no_state=True, data_first_axis='T')
            out = looper(bm.random.rand(5, 4, 3))
            assert jnp.shape(out) == (5, 4, 2)
            # counters advanced
            assert int(looper.i0.value) == 5

    def test_no_state_duration_raises(self):
        with bp.math.environment(dt=0.1):
            t = _Tiny(1)
            looper = bp.LoopOverTime(t, no_state=True)
            with pytest.raises(ValueError):
                looper(1.0)


# --------------------------------------------------------------------------- #
# reset_state
# --------------------------------------------------------------------------- #
class TestResetState:
    def test_reset_restores_counters(self):
        with bp.math.environment(dt=0.1):
            t = _Tiny(1)
            looper = bp.LoopOverTime(t, t0=0., i0=0)
            looper(bm.ones((5, 1)))
            assert int(looper.i0.value) == 5
            looper.reset_state()
            assert int(looper.i0.value) == 0
            assert float(looper.t0.value) == pytest.approx(0.0)

    def test_reset_with_none_counters(self):
        with bp.math.environment(dt=0.1):
            t = _Tiny(1)
            looper = bp.LoopOverTime(t, t0=None, i0=None)
            # should be a no-op without error
            looper.reset_state()
