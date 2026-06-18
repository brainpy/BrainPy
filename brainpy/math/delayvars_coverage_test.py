# -*- coding: utf-8 -*-
"""Supplementary coverage tests for ``brainpy/math/delayvars.py``.

``delay_vars_test.py`` and ``math_sparse_surrogate_fixes_test.py`` already cover
the common ``TimeDelay`` / ``LengthDelay`` paths.  This module targets the
remaining uncovered branches:

* the module helpers ``_as_jax_array`` and ``_accepts_dtype_kwarg`` (signature
  introspectable / not introspectable / explicit ``dtype`` / ``**kwargs``).
* ``batch_axis`` inference from a batched ``Variable`` delay target (TimeDelay
  line 188 and LengthDelay line 437).
* the ``_check_time1`` / ``_check_time2`` / ``_check_delay`` raising callbacks
  (these are the bodies the ``jit_error`` checks point at).
* ``_after_t0`` with the ``round`` interp method and the unsupported-method
  ``UnsupportedError`` branch.
* ``LengthDelay.reset`` with ``delay_len=None`` (reuse of ``num_delay_step``) and
  the ``dtype``-accepting callable ``initial_delay_data`` branch.
* ``retrieve`` / ``update`` unknown-method ValueErrors and the non-integer
  ``delay_len`` ValueError.
* ``update(value=None)`` with no stored target -> ValueError.
* ``TimeDelay.reset`` with an unsupported ``before_t0`` type -> ValueError.

NOTE: two source lines remain uncovered:
* ``_after_t0`` line 288 (``diff = diff.value``) only runs when the time-diff is
  a brainpy ``ndarray``; through the public API the subtraction always yields a
  plain jax array, so the branch is not reachable here.
* ``reset`` line 429 (``"delay_len" cannot be None``) requires
  ``num_delay_step is None`` while ``delay_len is None``, but the constructor
  always initialises ``num_delay_step`` to ``0`` first, so it is unreachable via
  the public constructor/reset path.
"""

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

import brainpy.math as bm
from brainpy._errors import UnsupportedError
from brainpy.math.ndarray import Array
from brainpy.math.delayvars import (
    _as_jax_array, _accepts_dtype_kwarg,
    TimeDelay, LengthDelay, ROTATE_UPDATE, CONCAT_UPDATE,
)


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------

def test_as_jax_array_unwraps_only_arrays():
    import jax
    out = _as_jax_array(bm.asarray([1., 2.]))
    assert isinstance(out, jax.Array)
    # non-Array passes straight through
    assert _as_jax_array(5) == 5


def test_accepts_dtype_kwarg_variants():
    # explicit dtype parameter
    assert _accepts_dtype_kwarg(lambda shape, dtype=None: shape) is True
    # **kwargs catch-all
    assert _accepts_dtype_kwarg(lambda shape, **kw: shape) is True
    # plain callable without dtype
    assert _accepts_dtype_kwarg(lambda shape: shape) is False
    # un-introspectable signature (built-in ``range``) -> conservative True
    assert _accepts_dtype_kwarg(range) is True


# ---------------------------------------------------------------------------
# batch_axis inference
# ---------------------------------------------------------------------------

def test_time_delay_batch_axis_from_variable():
    v = bm.Variable(bm.zeros((4, 3)), batch_axis=0)
    d = TimeDelay(v, delay_len=1.0, dt=0.1)
    # data prepends the delay-step axis, so batch_axis shifts by +1
    assert d.data.batch_axis == 1


def test_length_delay_batch_axis_from_variable():
    v = bm.Variable(bm.zeros((4, 3)), batch_axis=0)
    ld = LengthDelay(v, delay_len=3)
    assert ld.data.batch_axis == 1


# ---------------------------------------------------------------------------
# the raising check callbacks (bodies the jit_error checks point at)
# ---------------------------------------------------------------------------

def test_time_delay_check_callbacks_raise():
    d = TimeDelay(bm.zeros(1), delay_len=1.0, dt=0.1)
    with pytest.raises(ValueError):
        d._check_time1((1.0, 0.0))
    with pytest.raises(ValueError):
        d._check_time2((-5.0, 0.0))


def test_length_delay_check_callback_raises():
    ld = LengthDelay(bm.zeros(2), delay_len=3)
    with pytest.raises(ValueError):
        ld._check_delay(100)


# ---------------------------------------------------------------------------
# _after_t0 interp-method branches
# ---------------------------------------------------------------------------

def test_after_t0_round_method():
    d = TimeDelay(bm.zeros(1), delay_len=1.0, dt=0.1, interp_method='round')
    for k in range(1, 12):
        d.update(bm.ones(1) * (k * 0.1))
    out = np.asarray(d._after_t0(d.current_time[0]))
    assert out.shape == (1,)


def test_after_t0_unsupported_method_raises():
    d = TimeDelay(bm.zeros(1), delay_len=1.0, dt=0.1)
    d.interp_method = 'bogus'   # bypass the constructor guard
    with pytest.raises(UnsupportedError):
        d._after_t0(d.current_time[0])


# ---------------------------------------------------------------------------
# LengthDelay.reset / initial_delay_data callable-with-dtype branch
# ---------------------------------------------------------------------------

def test_length_delay_reset_delay_len_none_reuses_steps():
    ld = LengthDelay(bm.zeros(2), delay_len=3)   # num_delay_step == 4
    ld.reset(bm.ones(2), delay_len=None)         # reuse -> still 4
    assert ld.num_delay_step == 4


def test_length_delay_initial_delay_data_callable_with_dtype():
    # callable whose signature DOES accept dtype -> called as fn(shape, dtype=...)
    def init(shape, dtype=None):
        return jnp.ones(shape, dtype=dtype) * 7.0

    ld = LengthDelay(bm.zeros(2), delay_len=3, initial_delay_data=init)
    np.testing.assert_allclose(np.asarray(ld.retrieve(2)), [7., 7.], atol=1e-6)


# ---------------------------------------------------------------------------
# unknown-method / non-integer / update-None error branches
# ---------------------------------------------------------------------------

def test_retrieve_unknown_update_method_raises():
    ld = LengthDelay(bm.zeros(2), delay_len=3, update_method=ROTATE_UPDATE)
    ld.update_method = 'bogus'
    with pytest.raises(ValueError):
        ld.retrieve(1)


def test_update_unknown_update_method_raises():
    ld = LengthDelay(bm.zeros(2), delay_len=3, update_method=ROTATE_UPDATE)
    ld.update_method = 'bogus'
    with pytest.raises(ValueError):
        ld.update(bm.ones(2))


def test_retrieve_non_integer_delay_len_raises():
    ld = LengthDelay(bm.zeros(2), delay_len=3, update_method=CONCAT_UPDATE)
    # CONCAT_UPDATE uses delay_len directly as the index; a float dtype trips
    # the integer guard.
    with pytest.raises(ValueError):
        ld.retrieve(jnp.asarray(1.5))


def test_time_delay_reset_bad_before_t0_type_raises():
    d = TimeDelay(bm.zeros(1), delay_len=1.0, dt=0.1)
    with pytest.raises(ValueError):
        d.reset(bm.zeros(1), delay_len=1.0, before_t0='bad')   # unsupported type


def test_update_value_none_without_target_raises():
    ld = LengthDelay(bm.zeros(2), delay_len=3)   # plain array target -> no stored Variable
    assert ld.delay_target is None
    with pytest.raises(ValueError):
        ld.update(None)


# ---------------------------------------------------------------------------
# ring-buffer correctness regressions (guards the ``% num_delay_step`` modulo
# in ``TimeDelay._true_fn`` and the rotate index in ``LengthDelay``)
# ---------------------------------------------------------------------------

def test_time_delay_ring_buffer_wraps_modulo():
    """``_true_fn`` must read ``data[(idx + step) % num_delay_step]``.

    Without the modulo, when the read index wraps past the end of the buffer JAX
    clamps the out-of-bounds index to the last slot and returns a stale value.
    We feed a long ramp (many wraps) and check the exact-step (no-interp) reads.
    """
    dt = 0.1
    delay_len = 1.0  # exact multiple of dt -> exact-step (``_true_fn``) branch
    d = TimeDelay(bm.zeros(1), delay_len=delay_len, dt=dt, before_t0=lambda t: t)
    # ``num_delay_step == 11``; iterate well past one full wrap of the buffer.
    n = 37
    for i in range(n):
        d.update(bm.asarray([float(i)]))
    ct = float(d.current_time[0])
    last = n - 1  # the most recently stored ramp value
    # delay d_ms -> value stored ``round(d_ms/dt)`` steps before ``last``.
    for d_ms in [0.0, 0.1, 0.3, 0.5, 1.0]:
        got = float(d(ct - d_ms)[0])
        expected = last - round(d_ms / dt)
        assert abs(got - expected) < 1e-4, (d_ms, got, expected)


def test_length_delay_ramp_matches_reference():
    for method in (ROTATE_UPDATE, CONCAT_UPDATE):
        d = LengthDelay(bm.zeros(1), delay_len=5, update_method=method)
        for i in range(23):  # many wraps for the rotate buffer (len 6)
            d.update(bm.asarray([float(i)]))
        got = [float(d(k)[0]) for k in range(6)]
        assert got == [22 - k for k in range(6)], (method, got)
