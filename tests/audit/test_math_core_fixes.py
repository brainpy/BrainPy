# -*- coding: utf-8 -*-
"""Regression + coverage tests for the BrainPy v2.7.8 math-core audit
(see ``docs/issues-found-20260618.md``).

This module exercises the fixes recorded in the audit for:

* ``brainpy/math/ndarray.py``     — H-11 (``Array.device`` is a property),
  H-12 (``Array(scalar)``), M-09 (``ShardedArray.value`` read), Array.tree_*
  pytree round-trip under abstract eval, and L-03 (base vs sharded value
  setter policy).
* ``brainpy/math/environment.py`` — C-10 (``disable_x64`` re-syncs brainstate
  precision) and M-07 (``set()`` validate-before-mutate).
* ``brainpy/math/modes.py``       — H-10 (``Mode`` is hashable again).
* ``brainpy/math/scales.py``      — L-02 (``IdScaling`` rejects non-default
  bias/scale instead of silently ignoring them).
* ``brainpy/math/sharding.py``    — M-10 (``get_sharding`` warns on a full
  axis-name mismatch).
* ``brainpy/math/remove_vmap.py`` — M-08 (documented global-reduction batching
  behaviour).

Every test that toggles x64 / global precision / global environment restores
the original state in a ``finally`` block (or via the ``restore_environment``
fixture) so it cannot corrupt other tests in the suite.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainstate
from jax import config

import brainpy
import brainpy.math as bm
from brainpy._errors import MathError
from brainpy.math import modes, scales, sharding
from brainpy.math.ndarray import Array, ShardedArray, JaxArray, ndarray
from brainpy.math.remove_vmap import remove_vmap
from brainpy.math.defaults import defaults as _defaults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def restore_environment():
    """Snapshot global precision / x64 / float state and restore it afterwards.

    Any test that flips ``enable_x64``/``disable_x64`` or mutates global dtype
    settings MUST run inside this fixture so the rest of the suite keeps the
    default float32 environment.
    """
    orig_precision = brainstate.environ.get_precision()
    orig_x64 = config.read('jax_enable_x64')
    orig_float = bm.get_float()
    orig_int = bm.get_int()
    orig_complex = bm.get_complex()
    try:
        yield
    finally:
        # Restore JAX + brainstate precision symmetrically.
        if orig_x64:
            bm.enable_x64()
        else:
            bm.disable_x64()
        brainstate.environ.set(precision=orig_precision)
        bm.set_float(orig_float)
        bm.set_int(orig_int)
        bm.set_complex(orig_complex)


# ===========================================================================
# Regression tests
# ===========================================================================

# --- environment.py : C-10 -------------------------------------------------

def test_disable_x64_resyncs_brainstate_precision(restore_environment):
    """C-10: after ``enable_x64(); disable_x64()`` the brainstate precision is
    back to 32 (it used to be left at 64 while JAX was at float32)."""
    bm.enable_x64()
    assert brainstate.environ.get_precision() == 64
    assert config.read('jax_enable_x64') is True

    bm.disable_x64()
    assert brainstate.environ.get_precision() == 32
    assert config.read('jax_enable_x64') is False


def test_enable_then_disable_x64_leaves_precision_32(restore_environment):
    """The brainstate precision and JAX x64 flag stay in lock-step."""
    bm.enable_x64()
    bm.disable_x64()
    assert brainstate.environ.get_precision() == 32
    # float default tracks the disabled state.
    assert bm.get_float() == jnp.float32


# --- environment.py : M-07 -------------------------------------------------

def test_set_validate_before_mutate_invalid_numpy_func_return():
    """M-07: an invalid ``numpy_func_return`` raises and does NOT mutate
    ``get_float()`` (validation happens before any global write)."""
    before_float = bm.get_float()
    before_return = _defaults.numpy_func_return
    with pytest.raises(AssertionError):
        bm.set(float_=jnp.float64, numpy_func_return='not-a-valid-option')
    # No partial-config leak: float type unchanged.
    assert bm.get_float() == before_float
    assert _defaults.numpy_func_return == before_return


def test_set_validate_before_mutate_invalid_dt():
    """A non-float ``dt`` is rejected before any other arg is applied."""
    before_mode = bm.get_mode()
    with pytest.raises(AssertionError):
        bm.set(mode=modes.batching_mode, dt='not-a-float')
    assert bm.get_mode() is before_mode


# --- modes.py : H-10 -------------------------------------------------------

def test_modes_are_hashable():
    """H-10: defining ``__eq__`` had nuked ``__hash__``; restore hashability."""
    assert hash(modes.nonbatching_mode) is not None
    assert hash(brainpy.math.modes.nonbatching_mode) is not None


def test_modes_usable_in_a_set():
    """All three default modes can live together in a set."""
    s = {modes.nonbatching_mode, modes.batching_mode, modes.training_mode}
    assert len(s) == 3
    # Two non-batching instances compare/hash equal (by class).
    s2 = {modes.NonBatchingMode(), modes.NonBatchingMode()}
    assert len(s2) == 1


# --- ndarray.py : H-11 -----------------------------------------------------

def test_array_device_is_property_returning_jax_device():
    """H-11: ``Array.device`` is now a property (calling the old method form
    raised ``TypeError``). It returns a ``jax.Device``."""
    a = bm.asarray([1., 2.])
    dev = a.device          # property access, not a call
    assert isinstance(dev, jax.Device)


# --- ndarray.py : H-12 -----------------------------------------------------

def test_array_scalar_conversion_shape():
    """H-12: ``Array(scalar)`` stores a real jax array, so ``.shape == ()``."""
    assert Array(5).shape == ()
    assert Array(5).ndim == 0
    assert Array(5.0).shape == ()
    # value is a jax array, not a bare python scalar.
    assert isinstance(Array(5).value, jax.Array)


# --- ndarray.py : pytree round-trip ----------------------------------------

def test_array_pytree_round_trip_under_abstract_eval():
    """``Array.tree_unflatten`` must accept abstract leaves so transforms like
    ``jax.eval_shape`` work without re-running ``jnp.asarray``."""
    out = jax.eval_shape(lambda x: x * 2, bm.asarray([1., 2., 3.]))
    assert out.shape == (3,)


def test_array_tree_flatten_unflatten_concrete():
    a = Array([1., 2., 3.])
    leaves, aux = a.tree_flatten()
    assert aux is None
    rebuilt = Array.tree_unflatten(aux, leaves)
    np.testing.assert_array_equal(np.asarray(rebuilt.value), np.asarray(a.value))


def test_array_jit_round_trip():
    """A full jit round-trip preserves the Array pytree contract."""
    a = bm.asarray([1., 2., 3.])
    out = jax.jit(lambda x: x + 1.)(a)
    np.testing.assert_allclose(np.asarray(out), np.array([2., 3., 4.]))


# --- scales.py : L-02 ------------------------------------------------------

def test_idscaling_rejects_non_default_bias():
    ids = scales.IdScaling()
    with pytest.raises(ValueError):
        ids.offset_scaling(1.0, bias=5.0)


def test_idscaling_rejects_non_default_scale():
    ids = scales.IdScaling()
    with pytest.raises(ValueError):
        ids.std_scaling(1.0, scale=2.0)
    with pytest.raises(ValueError):
        ids.inv_scaling(1.0, scale=2.0)
    with pytest.raises(ValueError):
        ids.clone(scale=2.0)
    with pytest.raises(ValueError):
        ids.clone(bias=3.0)


def test_idscaling_identity_for_default_calls():
    ids = scales.IdScaling()
    assert ids.offset_scaling(7.0) == 7.0
    assert ids.std_scaling(7.0) == 7.0
    assert ids.inv_scaling(7.0) == 7.0
    # Explicit default values (0./1.) are accepted (no-op overrides).
    assert ids.offset_scaling(7.0, bias=0., scale=1.) == 7.0
    assert isinstance(ids.clone(), scales.IdScaling)
    assert ids.scale == 1.0 and ids.bias == 0.0


# --- sharding.py : M-10 ----------------------------------------------------

def _single_axis_mesh(name='x'):
    return jax.sharding.Mesh(np.asarray(jax.devices()), axis_names=(name,))


def test_get_sharding_warns_on_full_axis_mismatch():
    """M-10: when *every* requested axis name is absent, a fully-replicated
    PartitionSpec is produced; warn instead of silently dropping intent."""
    mesh = _single_axis_mesh('x')
    with pytest.warns(UserWarning):
        sh = sharding.get_sharding(['this_axis_does_not_exist'], mesh)
    assert sh is not None  # still returns a (replicated) NamedSharding


def test_get_sharding_no_warning_on_partial_match():
    """A partial match is tolerated on purpose: no warning."""
    mesh = _single_axis_mesh('x')
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any warning -> test failure
        sh = sharding.get_sharding(['x', 'missing'], mesh)
    assert sh is not None


def test_get_sharding_none_axis_names_returns_none():
    assert sharding.get_sharding(None) is None


# --- remove_vmap.py : M-08 -------------------------------------------------

def test_remove_vmap_global_reduction_any():
    """M-08 documents that the reduction is global. Outside vmap it is a real
    scalar."""
    out = remove_vmap(jnp.array([False, True]))
    assert out.shape == ()
    assert bool(out) is True


def test_remove_vmap_global_reduction_all():
    assert bool(remove_vmap(jnp.array([True, True]), 'all')) is True
    assert bool(remove_vmap(jnp.array([True, False]), 'all')) is False


def test_remove_vmap_under_vmap_is_global():
    """Under ``vmap`` the result is a *global* reduction across the batch (the
    documented, intentional behaviour): every batch slot sees the same value."""
    data = jnp.array([[0.1], [0.9]])
    out = jax.vmap(lambda x: remove_vmap(x > 0.5, 'any'))(data)
    # Global any over the whole batch is True; broadcast across the batch.
    assert bool(out[0]) is True and bool(out[1]) is True
    out_all = jax.vmap(lambda x: remove_vmap(x > 0.5, 'all'))(data)
    # Global all over the whole batch is False.
    assert bool(out_all[0]) is False and bool(out_all[1]) is False


def test_remove_vmap_invalid_op_raises():
    with pytest.raises(ValueError):
        remove_vmap(jnp.array([1]), 'unsupported')


def test_remove_vmap_unwraps_bp_array():
    """A ``brainpy.math.Array`` input is unwrapped (line ``x = x.value``)."""
    assert bool(remove_vmap(Array([False, True]))) is True
    assert bool(remove_vmap(Array([True, True]), 'all')) is True


def test_remove_vmap_under_jit_triggers_abstract_eval():
    """Running under ``jit`` exercises the abstract-eval rules of both
    primitives."""
    f_any = jax.jit(lambda x: remove_vmap(x > 0., 'any'))
    f_all = jax.jit(lambda x: remove_vmap(x > 0., 'all'))
    assert bool(f_any(jnp.array([-1., 1.]))) is True
    assert bool(f_all(jnp.array([1., 1.]))) is True
    assert bool(f_all(jnp.array([-1., 1.]))) is False


# ===========================================================================
# Coverage tests
# ===========================================================================

# --- ndarray.py : Array methods --------------------------------------------

def test_array_arithmetic():
    a = Array([1., 2., 3.])
    np.testing.assert_allclose(np.asarray(a + 1), [2., 3., 4.])
    np.testing.assert_allclose(np.asarray(a - 1), [0., 1., 2.])
    np.testing.assert_allclose(np.asarray(a * 2), [2., 4., 6.])
    np.testing.assert_allclose(np.asarray(-a), [-1., -2., -3.])
    np.testing.assert_allclose(np.asarray(a / 2), [0.5, 1., 1.5])


def test_array_indexing_and_iteration():
    a = Array([10., 20., 30.])
    assert float(a[0]) == 10.0
    assert len(a) == 3
    assert [float(x) for x in a] == [10.0, 20.0, 30.0]
    # slice
    np.testing.assert_allclose(np.asarray(a[1:]), [20., 30.])


def test_array_inplace_set():
    a = Array([1., 2., 3.])
    a[0] = 99.
    np.testing.assert_allclose(np.asarray(a.value), [99., 2., 3.])
    # .at accessor delegates to the underlying jax array.
    updated = a.at[1].set(0.)
    np.testing.assert_allclose(np.asarray(updated), [99., 0., 3.])


def test_array_repr_scalar_and_vector():
    r = repr(Array([1., 2., 3.]))
    assert r.startswith('Array(value=')
    # multi-line repr branch
    big = Array(np.arange(40.).reshape(8, 5))
    r2 = repr(big)
    assert 'Array(value=' in r2 and '\n' in r2 and 'dtype=' in r2


def test_array_dtype_shape_ndim():
    a = Array([[1., 2.], [3., 4.]])
    assert a.shape == (2, 2)
    assert a.ndim == 2
    assert a.dtype == jnp.float32


def test_array_value_setter_with_array_np_and_state():
    a = Array([0., 0.])
    # set from numpy
    a.value = np.array([4., 5.])
    np.testing.assert_allclose(np.asarray(a.value), [4., 5.])
    # set from another Array
    a.value = Array([7., 8.])
    np.testing.assert_allclose(np.asarray(a.value), [7., 8.])
    # set from a jax array
    a.value = jnp.array([1., 2.])
    np.testing.assert_allclose(np.asarray(a.value), [1., 2.])
    # set from a State / Variable
    a.value = bm.Variable(jnp.array([9., 10.]))
    np.testing.assert_allclose(np.asarray(a.value), [9., 10.])
    # set from a python list (the "else -> jnp.asarray" branch)
    a.value = [11., 12.]
    np.testing.assert_allclose(np.asarray(a.value), [11., 12.])


def test_array_data_property_and_update():
    a = Array([1., 2.])
    np.testing.assert_allclose(np.asarray(a.data), [1., 2.])
    a.data = jnp.array([3., 4.])
    np.testing.assert_allclose(np.asarray(a.data), [3., 4.])
    a.update(jnp.array([5., 6.]))
    np.testing.assert_allclose(np.asarray(a.value), [5., 6.])


def test_array_fill():
    a = Array([1., 2., 3.])
    a.fill_(7.)
    np.testing.assert_allclose(np.asarray(a.value), [7., 7., 7.])
    # Array scalar as fill value
    a.fill_(Array(2.))
    np.testing.assert_allclose(np.asarray(a.value), [2., 2., 2.])
    # numpy scalar
    a.fill_(np.float32(3.))
    np.testing.assert_allclose(np.asarray(a.value), [3., 3., 3.])
    # non-scalar fill value is rejected
    with pytest.raises(MathError):
        a.fill_(np.array([1., 2.]))


def test_array_numpy_and_jax_protocols():
    a = Array([1., 2., 3.])
    np.testing.assert_allclose(np.asarray(a), [1., 2., 3.])
    assert isinstance(a.__jax_array__(), jax.Array)
    # __array__ with dtype
    arr = np.asarray(a, dtype=np.int32)
    assert arr.dtype == np.int32


def test_array_as_variable():
    a = Array([1., 2.])
    v = a.as_variable()
    assert isinstance(v, bm.Variable)


def test_array_block_until_ready_and_device_buffer():
    a = Array([1., 2., 3.])
    assert isinstance(a.block_until_ready(), jax.Array)
    assert isinstance(a.block_host_until_ready(), jax.Array)
    np.testing.assert_allclose(np.asarray(a.device_buffer), [1., 2., 3.])


def test_array_aliases():
    assert JaxArray is Array
    assert ndarray is Array


def test_array_constructed_from_array_and_dtype():
    base = Array([1., 2., 3.])
    a = Array(base, dtype=jnp.int32)
    assert a.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(a.value), [1, 2, 3])
    # tuple input branch
    b = Array((1., 2.))
    np.testing.assert_allclose(np.asarray(b.value), [1., 2.])


# --- ndarray.py : ShardedArray ---------------------------------------------

def test_sharded_array_value_read_write():
    sa = ShardedArray(jnp.array([1., 2., 3.]))
    np.testing.assert_allclose(np.asarray(sa.value), [1., 2., 3.])
    sa.value = jnp.array([4., 5., 6.])
    np.testing.assert_allclose(np.asarray(sa.value), [4., 5., 6.])


def test_sharded_array_enforces_shape_and_dtype():
    """L-03 / M-09: the ShardedArray setter enforces shape & dtype."""
    sa = ShardedArray(jnp.array([1., 2., 3.]))
    with pytest.raises(MathError):
        sa.value = jnp.array([1., 2.])          # wrong shape
    with pytest.raises(MathError):
        sa.value = jnp.array([1, 2, 3])         # wrong dtype


def test_sharded_array_keep_sharding_false():
    sa = ShardedArray(jnp.array([1., 2.]), keep_sharding=False)
    np.testing.assert_allclose(np.asarray(sa.value), [1., 2.])


def test_sharded_array_setter_from_array_and_np():
    sa = ShardedArray(jnp.array([1., 2.]))
    sa.value = Array([3., 4.])
    np.testing.assert_allclose(np.asarray(sa.value), [3., 4.])
    sa.value = np.array([5., 6.], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(sa.value), [5., 6.])


# --- environment.py : getters / setters ------------------------------------

def test_environment_dtype_getters():
    assert bm.get_float() in (jnp.float32, jnp.float64)
    assert bm.get_int() in (jnp.int32, jnp.int64)
    assert bm.get_complex() in (jnp.complex64, jnp.complex128)
    assert bm.get_bool() == jnp.bool_
    # deprecated aliases still resolve.
    assert bm.dftype() == bm.get_float()
    assert bm.ditype() == bm.get_int()


def test_environment_dtype_setters_round_trip():
    orig_float = bm.get_float()
    orig_int = bm.get_int()
    orig_complex = bm.get_complex()
    orig_bool = bm.get_bool()
    try:
        bm.set_float(jnp.float16)
        assert bm.get_float() == jnp.float16
        bm.set_int(jnp.int16)
        assert bm.get_int() == jnp.int16
        bm.set_complex(jnp.complex64)
        assert bm.get_complex() == jnp.complex64
        bm.set_bool(jnp.bool_)
        assert bm.get_bool() == jnp.bool_
    finally:
        bm.set_float(orig_float)
        bm.set_int(orig_int)
        bm.set_complex(orig_complex)
        bm.set_bool(orig_bool)


def test_environment_dt_setter_round_trip():
    orig = bm.get_dt()
    try:
        bm.set_dt(0.05)
        assert bm.get_dt() == 0.05
    finally:
        bm.set_dt(orig)
    assert bm.get_dt() == orig


def test_environment_mode_setter_round_trip():
    orig = bm.get_mode()
    try:
        bm.set_mode(modes.batching_mode)
        assert bm.get_mode() is modes.batching_mode
    finally:
        bm.set_mode(orig)
    with pytest.raises(TypeError):
        bm.set_mode('not-a-mode')


def test_environment_membrane_scaling_setter_round_trip():
    orig = bm.get_membrane_scaling()
    try:
        s = scales.Scaling(scale=2., bias=1.)
        bm.set_membrane_scaling(s)
        assert bm.get_membrane_scaling() is s
    finally:
        bm.set_membrane_scaling(orig)
    with pytest.raises(TypeError):
        bm.set_membrane_scaling('not-a-scaling')


def test_environment_get_platform():
    assert bm.get_platform() in ('cpu', 'gpu', 'tpu')


def test_set_applies_all_valid_args_round_trip(restore_environment):
    """Exercise the apply branches of ``set()`` with every argument, then
    restore each global it touched."""
    orig_dt = bm.get_dt()
    orig_mode = bm.get_mode()
    orig_ms = bm.get_membrane_scaling()
    orig_float = bm.get_float()
    orig_int = bm.get_int()
    orig_bool = bm.get_bool()
    orig_complex = bm.get_complex()
    orig_pytree = _defaults.bp_object_as_pytree
    orig_return = _defaults.numpy_func_return
    try:
        bm.set(
            mode=modes.batching_mode,
            membrane_scaling=scales.Scaling(scale=2., bias=0.),
            dt=0.25,
            x64=False,
            complex_=jnp.complex64,
            float_=jnp.float32,
            int_=jnp.int32,
            bool_=jnp.bool_,
            bp_object_as_pytree=True,
            numpy_func_return='jax_array',
        )
        assert bm.get_dt() == 0.25
        assert isinstance(bm.get_mode(), modes.BatchingMode)
        assert bm.get_membrane_scaling().scale == 2.
        assert bm.get_float() == jnp.float32
        assert bm.get_int() == jnp.int32
        assert bm.get_complex() == jnp.complex64
        assert _defaults.bp_object_as_pytree is True
        assert _defaults.numpy_func_return == 'jax_array'
    finally:
        bm.set_dt(orig_dt)
        bm.set_mode(orig_mode)
        bm.set_membrane_scaling(orig_ms)
        bm.set_float(orig_float)
        bm.set_int(orig_int)
        bm.set_bool(orig_bool)
        bm.set_complex(orig_complex)
        _defaults.bp_object_as_pytree = orig_pytree
        _defaults.numpy_func_return = orig_return


def test_set_environment_is_set_alias():
    from brainpy.math.environment import set_environment, set as _set
    assert set_environment is _set


def test_environment_context_all_dtype_kwargs(restore_environment):
    """Exercise the ``__init__``/``__enter__``/``__exit__`` branches for the
    dtype + scaling + pytree + numpy_func_return arguments."""
    orig_float = bm.get_float()
    with bm.environment(
        membrane_scaling=scales.Scaling(scale=3., bias=0.),
        float_=jnp.float32,
        int_=jnp.int32,
        bool_=jnp.bool_,
        complex_=jnp.complex64,
        bp_object_as_pytree=True,
        numpy_func_return='jax_array',
    ):
        assert bm.get_membrane_scaling().scale == 3.
        assert _defaults.bp_object_as_pytree is True
        assert _defaults.numpy_func_return == 'jax_array'
    assert bm.get_float() == orig_float
    assert _defaults.numpy_func_return != 'jax_array' or _defaults.numpy_func_return == 'bp_array'


def test_environment_as_decorator(restore_environment):
    """``environment`` doubles as a decorator (``_DecoratorContextManager``)."""

    @bm.environment(dt=0.07)
    def get_dt_inside():
        return bm.get_dt()

    orig = bm.get_dt()
    assert get_dt_inside() == 0.07
    assert bm.get_dt() == orig


def test_environment_init_rejects_bad_types():
    with pytest.raises(AssertionError):
        bm.environment(dt='not-a-float')
    with pytest.raises(AssertionError):
        bm.environment(mode='not-a-mode')
    with pytest.raises(AssertionError):
        bm.environment(x64='not-a-bool')
    with pytest.raises(AssertionError):
        bm.environment(numpy_func_return='bad-option')


def test_environment_context_manager_restores(restore_environment):
    orig_mode = bm.get_mode()
    orig_dt = bm.get_dt()
    with bm.environment(mode=modes.batching_mode, dt=0.2):
        assert bm.get_mode() is modes.batching_mode
        assert bm.get_dt() == 0.2
    assert bm.get_mode() is orig_mode
    assert bm.get_dt() == orig_dt


def test_batching_and_training_environment(restore_environment):
    with bm.batching_environment(dt=0.3):
        assert isinstance(bm.get_mode(), modes.BatchingMode)
    with bm.training_environment(batch_size=4):
        m = bm.get_mode()
        assert isinstance(m, modes.TrainingMode)
        assert m.batch_size == 4


def test_environment_x64_context_manager(restore_environment):
    """``environment(x64=...)`` flips and restores the precision symmetrically."""
    start = config.read('jax_enable_x64')
    with bm.environment(x64=not start):
        assert config.read('jax_enable_x64') == (not start)
    assert config.read('jax_enable_x64') == start


def test_enable_x64_with_bool_argument_warns(restore_environment):
    """The legacy ``enable_x64(True)`` path emits a DeprecationWarning."""
    with pytest.warns(DeprecationWarning):
        bm.enable_x64(True)
    assert config.read('jax_enable_x64') is True


def test_set_x64_helper(restore_environment):
    from brainpy.math.environment import set_x64
    set_x64(True)
    assert config.read('jax_enable_x64') is True
    set_x64(False)
    assert config.read('jax_enable_x64') is False


def test_enable_x64_false_branch_warns(restore_environment):
    """The deprecated ``enable_x64(False)`` path routes through the disable
    branch and warns."""
    bm.enable_x64()  # go to 64 first
    with pytest.warns(DeprecationWarning):
        bm.enable_x64(False)
    assert config.read('jax_enable_x64') is False
    assert brainstate.environ.get_precision() == 32


def test_environment_decorator_on_generator(restore_environment):
    """Cover ``_DecoratorContextManager._wrap_generator`` by decorating a
    generator function with ``environment``."""

    @bm.environment(dt=0.09)
    def gen():
        yield bm.get_dt()
        yield bm.get_dt()

    orig = bm.get_dt()
    values = list(gen())
    assert values == [0.09, 0.09]
    assert bm.get_dt() == orig


def test_set_host_device_count_sets_env_var(monkeypatch):
    """``set_host_device_count`` writes the XLA flag (no global precision
    change)."""
    monkeypatch.setenv('XLA_FLAGS', '')
    bm.set_host_device_count(3)
    import os
    assert '--xla_force_host_platform_device_count=3' in os.environ['XLA_FLAGS']


def test_gpu_memory_preallocation_toggles(monkeypatch):
    import os
    from brainpy.math.environment import gpu_memory_preallocation
    monkeypatch.delenv('XLA_PYTHON_CLIENT_PREALLOCATE', raising=False)
    monkeypatch.delenv('XLA_PYTHON_CLIENT_ALLOCATOR', raising=False)
    bm.disable_gpu_memory_preallocation()
    assert os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] == 'false'
    assert os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] == 'platform'
    bm.enable_gpu_memory_preallocation()
    assert os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] == 'true'
    assert 'XLA_PYTHON_CLIENT_ALLOCATOR' not in os.environ
    gpu_memory_preallocation(0.5)
    assert os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] == '0.5'
    with pytest.raises(AssertionError):
        gpu_memory_preallocation(1.5)


def test_set_platform_rejects_unknown():
    with pytest.raises(AssertionError):
        bm.set_platform('quantum')


def test_environment_identity_equality():
    """``environment.__eq__`` is identity-based."""
    env1 = bm.environment(dt=0.1)
    env2 = bm.environment(dt=0.1)
    assert env1 == env1
    assert env1 != env2


def test_environment_clone_preserves_settings():
    env = bm.environment(dt=0.1, mode=modes.batching_mode)
    clone = env.clone()
    assert clone.dt == 0.1
    assert clone.mode is modes.batching_mode


# --- modes.py --------------------------------------------------------------

def test_mode_repr():
    assert repr(modes.nonbatching_mode) == 'NonBatchingMode'
    assert repr(modes.batching_mode) == 'BatchingMode(batch_size=1)'
    assert repr(modes.training_mode) == 'TrainingMode(batch_size=1)'


def test_mode_equality():
    assert modes.NonBatchingMode() == modes.NonBatchingMode()
    assert (modes.NonBatchingMode() == modes.BatchingMode()) is False
    # comparison against a non-Mode returns False (not an error)
    assert (modes.nonbatching_mode == 5) is False


def test_mode_predicates():
    assert modes.batching_mode.is_batch_mode() is True
    assert modes.training_mode.is_train_mode() is True
    assert modes.nonbatching_mode.is_nonbatch_mode() is True
    assert modes.nonbatching_mode.batch_size == tuple()
    assert modes.batching_mode.batch_size == 1


def test_mode_type_queries():
    assert modes.batching_mode.is_one_of(modes.BatchingMode, modes.TrainingMode)
    assert modes.batching_mode.is_a(modes.BatchingMode)
    assert modes.batching_mode.is_parent_of(modes.TrainingMode)
    assert modes.training_mode.is_child_of(modes.BatchingMode)
    # invalid (non-type) arguments raise TypeError
    with pytest.raises(TypeError):
        modes.batching_mode.is_one_of(modes.batching_mode)
    with pytest.raises(TypeError):
        modes.batching_mode.is_parent_of(modes.batching_mode)
    with pytest.raises(TypeError):
        modes.batching_mode.is_child_of(modes.batching_mode)


def test_training_mode_to_batch_mode():
    tm = modes.TrainingMode(3)
    bmode = tm.to_batch_mode()
    assert isinstance(bmode, modes.BatchingMode)
    assert not isinstance(bmode, modes.TrainingMode)
    assert bmode.batch_size == 3


# --- scales.py -------------------------------------------------------------

def test_scaling_offset_std_inv():
    s = scales.Scaling(scale=2., bias=1.)
    assert s.offset_scaling(3.0) == (3.0 + 1.0) / 2.0
    assert s.std_scaling(4.0) == 4.0 / 2.0
    assert s.inv_scaling(4.0) == 4.0 * 2.0
    # explicit overrides honored on a plain Scaling
    assert s.offset_scaling(3.0, bias=0., scale=1.) == 3.0
    assert s.std_scaling(4.0, scale=4.0) == 1.0
    assert s.inv_scaling(4.0, scale=0.5) == 2.0


def test_scaling_clone():
    s = scales.Scaling(scale=2., bias=1.)
    c = s.clone()
    assert c.scale == 2. and c.bias == 1.
    c2 = s.clone(bias=5., scale=3.)
    assert c2.scale == 3. and c2.bias == 5.


def test_scaling_transform():
    s = scales.Scaling.transform([0., 10.], [0., 1.])
    assert s.scale == 10.0
    assert s.bias == 0.0
    # round-trip: offset then inv recovers the offset-corrected value
    assert s.offset_scaling(10.0) == 1.0


# --- sharding.py -----------------------------------------------------------

def test_device_mesh_context_sets_and_restores():
    devs = np.asarray(jax.devices())
    with sharding.device_mesh(devs, ('x',)) as mesh:
        assert mesh.axis_names == ('x',)
        sh = sharding.get_sharding(['x'])
        assert sh is not None
    # default mesh restored to None afterwards
    assert sharding.get_sharding(['x']) is None


def test_partition_none_passthrough():
    x = jnp.array([1., 2.])
    assert sharding.partition(x, None) is x


def test_partition_by_axname_no_mesh_returns_input():
    x = jnp.array([1., 2.])
    # No default mesh -> input returned unchanged.
    out = sharding.partition_by_axname(x, ['x'])
    np.testing.assert_allclose(np.asarray(out), [1., 2.])
    # axis_names None -> input returned.
    assert sharding.partition_by_axname(x, None) is x


def test_partition_by_axname_shape_mismatch_raises():
    devs = np.asarray(jax.devices())
    with sharding.device_mesh(devs, ('x',)):
        with pytest.raises(ValueError):
            # 1-D array but two requested axis names -> dim mismatch
            sharding.partition_by_axname(jnp.array([1., 2.]), ['x', 'y'])


def test_partition_by_sharding_none_and_typecheck():
    x = jnp.array([1., 2.])
    assert sharding.partition_by_sharding(x, None) is x
    with pytest.raises(TypeError):
        sharding.partition_by_sharding(x, 'not-a-sharding')


def test_partition_invalid_type_raises():
    with pytest.raises(TypeError):
        sharding.partition(jnp.array([1.]), 12345)


def test_keep_constraint_passthrough():
    out = sharding.keep_constraint(jnp.array([1., 2.]))
    np.testing.assert_allclose(np.asarray(out), [1., 2.])
    # non-array leaves pass through untouched
    assert sharding.keep_constraint(7) == 7


def test_is_bp_array_helper():
    assert sharding.is_bp_array(Array([1.])) is True
    assert sharding.is_bp_array(jnp.array([1.])) is False


def test_partition_by_axname_with_mesh_devices():
    """Exercise the real device-put path of ``partition_by_axname`` /
    ``_device_put`` with an actual single-device mesh."""
    devs = np.asarray(jax.devices())
    with sharding.device_mesh(devs, ('x',)):
        # 1-D array, one axis name -> dims match -> resharded via _device_put.
        out = sharding.partition_by_axname(jnp.array([1., 2.]), ['x'])
        np.testing.assert_allclose(np.asarray(out), [1., 2.])
        # Array leaf goes through the ``isinstance(x, Array)`` branch.
        out2 = sharding.partition_by_axname(Array([3., 4.]), ['x'])
        leaf = jax.tree_util.tree_leaves(out2, is_leaf=sharding.is_bp_array)[0]
        np.testing.assert_allclose(np.asarray(leaf), [3., 4.])


def test_partition_with_sharding_object():
    """``partition`` with a concrete ``Sharding`` instance routes through
    ``partition_by_sharding``/``_device_put``."""
    devs = np.asarray(jax.devices())
    mesh = jax.sharding.Mesh(devs, axis_names=('x',))
    sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))
    out = sharding.partition(jnp.array([1., 2.]), sh)
    np.testing.assert_allclose(np.asarray(out.value if isinstance(out, Array) else out),
                               [1., 2.])


def test_partition_with_axis_name_sequence():
    devs = np.asarray(jax.devices())
    with sharding.device_mesh(devs, ('x',)):
        out = sharding.partition(jnp.array([1., 2.]), ['x'])
        leaf = jax.tree_util.tree_leaves(out, is_leaf=sharding.is_bp_array)[0]
        np.testing.assert_allclose(np.asarray(leaf), [1., 2.])


def test_keep_constraint_on_bp_array():
    out = sharding.keep_constraint(Array([1., 2., 3.]))
    np.testing.assert_allclose(np.asarray(out), [1., 2., 3.])
