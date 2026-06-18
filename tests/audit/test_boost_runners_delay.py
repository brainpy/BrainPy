# -*- coding: utf-8 -*-
"""Audit coverage-boost tests for ``brainpy/runners.py`` and ``brainpy/delay.py``.

These tests exercise the *uncovered* option matrix of ``DSRunner`` (jit on/off,
progress bar, memory-efficient mode, list/dict/callable monitors, deprecated
``fun_monitors``, ``numpy_mon_after_run``, ``data_first_axis`` = 'T'/'B', nonzero
``t0``, the several input formats, ``shared_args``/dyn args, repeated ``.run()``,
``.predict()`` and ``.reset_state()``) plus all the ``__init__``/``predict``
validation error paths.

For ``delay.py`` it drives ``Delay``/``VarDelay``/``DataDelay``/``DelayAccess``,
both ``ROTATE_UPDATE`` and ``CONCAT_UPDATE`` methods, ``register_entry`` by
``time=`` and by ``step=``, ``.at()``/``.retrieve()``/``.update()``,
``init_delay_by_return`` (Variable + ReturnInfo), ``before_t0``-style ``init`` as
array and as callable, and the validation/error branches.

Sibling audit files already cover basic ``DSRunner`` and ``VarDelay(time=T)``;
here we focus on the previously-uncovered options and code paths so that line
coverage of both modules rises toward >=90%.
"""

import warnings

import numpy as np
import pytest

import brainpy as bp
import brainpy.math as bm
import jax.numpy as jnp

from brainpy import check
from brainpy.delay import (
    Delay,
    VarDelay,
    DataDelay,
    DelayAccess,
    init_delay_by_return,
    register_delay_by_return,
)
from brainpy.math.delayvars import ROTATE_UPDATE, CONCAT_UPDATE
from brainpy.mixin import ReturnInfo


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _net(n=4, **kw):
    """A tiny spiking network used across the runner tests."""
    bm.random.seed(123)
    return bp.dyn.LifRef(n, **kw)


# ===========================================================================
# DSRunner -- monitors
# ===========================================================================

def test_runner_list_monitors_jit_memory_efficient():
    n = _net()
    r = bp.DSRunner(n, monitors=['V', 'spike'], jit=True,
                    progress_bar=False, memory_efficient=True)
    out = r.run(2.0)  # 20 steps @ dt=0.1
    assert r.mon['V'].shape == (20, 4)
    assert r.mon['spike'].shape == (20, 4)
    assert r.mon['ts'].shape == (20,)
    # memory-efficient mode forces numpy monitors
    assert isinstance(r.mon['V'], np.ndarray)


def test_runner_list_monitors_with_index():
    n = _net()
    r = bp.DSRunner(n, monitors=[('V', 0), ('spike', [1, 2])],
                    progress_bar=False)
    r.run(1.0)
    assert np.asarray(r.mon['V']).shape == (10, 1)
    assert np.asarray(r.mon['spike']).shape == (10, 2)


def test_runner_dict_monitors_variants():
    """dict monitors: explicit var, (var, idx) tuple, and a callable."""
    n = _net()
    r = bp.DSRunner(
        n,
        monitors={'v0': (n.V, 0), 'vall': n.V, 'fcb': lambda: n.V[:2]},
        jit=False,
        progress_bar=False,
        numpy_mon_after_run=False,  # keep jax arrays
        data_first_axis='T',
    )
    r.run(1.0)
    assert np.asarray(r.mon['v0']).shape == (10, 1)
    assert np.asarray(r.mon['vall']).shape == (10, 4)
    assert np.asarray(r.mon['fcb']).shape == (10, 2)
    # numpy_mon_after_run=False -> ts is a jax array
    assert isinstance(bm.as_jax(r.mon['ts']), jnp.ndarray)


def test_runner_fun_monitors_deprecated_path():
    n = _net()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = bp.DSRunner(n, fun_monitors={'sp': lambda: n.spike[:2]},
                        progress_bar=False)
        r.run(1.0)
    assert np.asarray(r.mon['sp']).shape == (10, 2)


def test_runner_progress_bar_true():
    """progress_bar=True exercises the brainstate pbar branch."""
    n = _net()
    r = bp.DSRunner(n, monitors=['V'], progress_bar=True, jit=True)
    r.run(1.0)
    assert r.mon['V'].shape == (10, 4)


def test_runner_t0_nonzero_shifts_time_axis():
    n = _net()
    r = bp.DSRunner(n, monitors=['V'], progress_bar=False, t0=5.0)
    r.run(1.0)
    assert float(r.mon['ts'][0]) == pytest.approx(5.0)
    assert float(r.mon['ts'][1]) == pytest.approx(5.1)


# ===========================================================================
# DSRunner -- inputs
# ===========================================================================

def test_runner_input_fix_tuple():
    n = _net()
    r = bp.DSRunner(n, monitors=['V'], inputs=(n.V, 1.0), progress_bar=False)
    r.run(1.0)
    assert r.mon['V'].shape == (10, 4)


def test_runner_input_iter_array():
    n = _net()
    arr = np.ones((10, 4)) * 0.5
    r = bp.DSRunner(n, monitors=['V'], inputs=(n.V, arr, 'iter'),
                    progress_bar=False)
    r.run(1.0)
    assert r.mon['V'].shape == (10, 4)


def test_runner_input_iter_generator():
    """A non-array iterable goes through the ``next(...)`` path."""
    n = _net()
    r = bp.DSRunner(n, monitors=['V'], inputs=(n.V, [0.1] * 20, 'iter'),
                    progress_bar=False)
    r.run(1.0)
    assert r.mon['V'].shape == (10, 4)


def test_runner_input_func():
    n = _net()
    r = bp.DSRunner(n, monitors=['V'], inputs=(n.V, lambda: 0.3, 'func', '+'),
                    progress_bar=False)
    r.run(1.0)
    assert r.mon['V'].shape == (10, 4)


def test_runner_input_string_target_assign_op():
    """String target ('V') with the '=' operation (relative/absolute access)."""
    n = _net()
    r = bp.DSRunner(n, monitors=['V'], inputs=('V', 0.1, 'fix', '='),
                    progress_bar=False)
    r.run(1.0)
    assert r.mon['V'].shape == (10, 4)


def test_runner_callable_inputs():
    n = _net()

    def fin():
        n.V += 0.1

    r = bp.DSRunner(n, monitors=['V'], inputs=fin, progress_bar=False)
    r.run(1.0)
    assert r.mon['V'].shape == (10, 4)


def test_runner_multiple_inputs_and_ops():
    """Several inputs with different operations in one runner."""
    n = _net()
    r = bp.DSRunner(
        n,
        monitors=['V'],
        inputs=[(n.V, 0.2, 'fix', '+'),
                ('V', 0.01, 'fix', '*')],
        progress_bar=False,
    )
    r.run(1.0)
    assert r.mon['V'].shape == (10, 4)


# ===========================================================================
# DSRunner -- predict / run / reset_state
# ===========================================================================

def test_predict_with_xs_array_nonbatching():
    n = _net()
    r = bp.DSRunner(n, monitors=['V'], progress_bar=False)
    xs = np.ones((15, 4)) * 0.4
    out = r.predict(inputs=xs)
    assert np.asarray(out).shape == (15, 4)
    assert r.mon['ts'].shape == (15,)


def test_predict_eval_time_and_reset_state_arg():
    n = _net()
    r = bp.DSRunner(n, monitors=['V'], progress_bar=False)
    running_time, out = r.predict(1.0, reset_state=True, eval_time=True)
    assert isinstance(running_time, float)
    assert np.asarray(out).shape == (10, 4)


def test_runner_reset_state_method():
    n = _net()
    r = bp.DSRunner(n, monitors=['V'], progress_bar=False)
    r.run(1.0)
    assert r.i0 == 10
    r.reset_state()
    assert r.i0 == 0


def test_runner_repeated_run_accumulates_i0():
    n = _net()
    r = bp.DSRunner(n, monitors=['V'], progress_bar=False)
    r.run(1.0)
    assert r.i0 == 10
    r.run(1.0)
    assert r.i0 == 20


def test_runner_shared_args_dyn_args():
    n = _net()
    r = bp.DSRunner(n, monitors=['V'], progress_bar=False)
    r.predict(1.0, shared_args={'fit': False})
    assert r.mon['V'].shape == (10, 4)


def test_runner_call_dunder():
    n = _net()
    r = bp.DSRunner(n, monitors=['V'], progress_bar=False)
    out = r(1.0)  # __call__
    assert np.asarray(out).shape == (10, 4)


def test_runner_repr():
    n = _net()
    r = bp.DSRunner(n, monitors=['V'], progress_bar=False)
    s = repr(r)
    assert 'DSRunner' in s and 'data_first_axis' in s


def test_predict_duration_and_inputs_warns():
    """Providing both duration and inputs warns and uses inputs' time axis."""
    n = _net()
    r = bp.DSRunner(n, monitors=['V'], progress_bar=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r.predict(1.0, inputs=np.ones((10, 4)) * 0.3)
    assert r.mon['ts'].shape == (10,)


# ===========================================================================
# DSRunner -- batching mode + data_first_axis='B'
# ===========================================================================

def test_runner_batching_mode_first_axis_B():
    bm.random.seed(7)
    n = bp.dyn.LifRef(4, mode=bm.batching_mode)
    n.reset(batch_size=3)
    r = bp.DSRunner(n, monitors=['V'], progress_bar=False, data_first_axis='B')
    xs = np.ones((3, 12, 4)) * 0.4  # (batch, time, feat)
    out = r.predict(inputs=xs, reset_state=True)
    assert np.asarray(out).shape == (3, 12, 4)
    assert r.mon['V'].shape == (3, 12, 4)


def test_runner_batching_default_data_first_axis():
    """A BatchingMode target defaults data_first_axis to 'B'."""
    n = bp.dyn.LifRef(4, mode=bm.batching_mode)
    r = bp.DSRunner(n, monitors=['V'], progress_bar=False)
    assert r.data_first_axis == 'B'


# ===========================================================================
# DSRunner -- validation / error branches
# ===========================================================================

def test_runner_bad_target_type():
    from brainpy._errors import RunningError
    with pytest.raises(RunningError):
        bp.DSRunner(object())


def test_runner_bad_monitors_type():
    with pytest.raises(Exception):  # MonitorError
        bp.DSRunner(_net(), monitors=123)


def test_runner_bad_inputs_type():
    from brainpy._errors import RunningError
    with pytest.raises(RunningError):
        bp.DSRunner(_net(), inputs=123)


def test_runner_bad_input_structure():
    from brainpy._errors import RunningError
    # first element neither str nor Variable and not a list/tuple
    with pytest.raises(RunningError):
        bp.DSRunner(_net(), inputs=[1.0, 2.0])


def test_runner_bad_input_length():
    from brainpy._errors import RunningError
    with pytest.raises(RunningError):
        bp.DSRunner(_net(), inputs=(_net().V,))  # length 1


def test_runner_bad_input_op():
    from brainpy._errors import RunningError
    with pytest.raises(RunningError):
        bp.DSRunner(_net(), inputs=('V', 1.0, 'fix', '^'))


def test_runner_bad_input_type_str():
    from brainpy._errors import RunningError
    with pytest.raises(RunningError):
        bp.DSRunner(_net(), inputs=('V', 1.0, 'bogus'))


def test_runner_iter_value_not_iterable():
    with pytest.raises(ValueError):
        bp.DSRunner(_net(), inputs=('V', 5, 'iter'))


def test_runner_func_value_not_callable():
    with pytest.raises(ValueError):
        bp.DSRunner(_net(), inputs=('V', 5, 'func'))


def test_runner_bad_input_target_attr():
    with pytest.raises(AttributeError):
        bp.DSRunner(_net(), inputs=('nonexist.attr', 1.0))


def test_runner_nonvar_nonstr_target():
    from brainpy._errors import RunningError
    with pytest.raises(RunningError):
        bp.DSRunner(_net(), inputs=(123, 1.0))


def test_runner_memory_efficient_requires_numpy_mon():
    with pytest.raises(ValueError):
        bp.DSRunner(_net(), memory_efficient=True, numpy_mon_after_run=False)


def test_predict_without_duration_or_inputs():
    n = _net()
    r = bp.DSRunner(n, progress_bar=False)
    with pytest.raises(ValueError):
        r.predict()


def test_runner_bad_dt_type():
    from brainpy._errors import RunningError
    with pytest.raises(RunningError):
        bp.DSRunner(_net(), dt=1)  # int, not float


# ===========================================================================
# delay.py -- VarDelay ROTATE_UPDATE
# ===========================================================================

def _drive_delay(delay, target_var, n_steps, value_fn=None):
    """Run a delay update loop with the proper shared-arg context."""
    dt = bm.get_dt()
    for i in range(n_steps):
        bp.share.save(i=i, t=i * dt, dt=dt)
        v = (value_fn(i) if value_fn is not None
             else bm.ones_like(target_var.value) * i)
        target_var.value = v
        delay.update()


def test_vardelay_rotate_register_time_and_step():
    bm.random.seed(0)
    v = bm.Variable(bm.zeros(3))
    d = VarDelay(v, time=2.0)  # 20 steps capacity
    d.register_entry('by_time', delay_time=1.0)   # -> 10 steps
    d.register_entry('by_step', delay_step=5)
    d.register_entry('zero', delay_time=None)     # zero-delay -> target value
    assert d._registered_entries['by_time'] == 10
    assert d._registered_entries['by_step'] == 5
    assert d._registered_entries['zero'] is None

    _drive_delay(d, v, 25)
    # at the zero entry returns the current target value
    np.testing.assert_allclose(np.asarray(d.at('zero')), 24.0)
    # delayed entries return earlier values
    np.testing.assert_allclose(np.asarray(d.at('by_time')), 15.0)
    np.testing.assert_allclose(np.asarray(d.at('by_step', 0)), 20.0)
    # direct retrieve
    np.testing.assert_allclose(np.asarray(d.retrieve(3)), 22.0)


def test_vardelay_at_with_indices():
    bm.random.seed(0)
    v = bm.Variable(bm.zeros(4))
    d = VarDelay(v, time=1.0)
    d.register_entry('e', delay_step=5)
    _drive_delay(d, v, 12)
    out = d.at('e', 1)
    assert np.asarray(out).shape == ()


def test_vardelay_init_array_and_callable():
    """``init`` provided as array and as callable (before_t0-style data)."""
    v = bm.Variable(bm.zeros(3))
    d_arr = VarDelay(v, time=0.5, init=jnp.ones((5, 3)) * 2.0)
    assert np.asarray(d_arr.data.value).sum() == pytest.approx(2.0 * 15)

    v2 = bm.Variable(bm.zeros(3))
    d_call = VarDelay(v2, time=0.5,
                      init=lambda shape, dtype: jnp.ones(shape, dtype) * 3.0)
    assert np.asarray(d_call.data.value).sum() == pytest.approx(3.0 * 15)


def test_vardelay_reset_state_and_repr():
    v = bm.Variable(bm.zeros(3))
    d = VarDelay(v, time=1.0, init=5.0)
    d.register_entry('e', delay_step=3)
    _drive_delay(d, v, 5)
    d.reset_state()
    # after reset the buffer is re-initialised to init=5.0
    np.testing.assert_allclose(np.asarray(d.data.value), 5.0)
    assert 'VarDelay' in repr(d)
    assert d.delay_target_shape == (3,)


def test_vardelay_register_entry_array_delay_time():
    v = bm.Variable(bm.zeros(3))
    d = VarDelay(v, time=1.0)
    d.register_entry('arr', delay_time=jnp.asarray(0.5))
    assert d._registered_entries['arr'] == 5


# ===========================================================================
# delay.py -- VarDelay CONCAT_UPDATE
# ===========================================================================

def test_vardelay_concat_update_multi_step():
    bm.random.seed(0)
    v = bm.Variable(bm.zeros(3))
    d = VarDelay(v, time=1.0, method=CONCAT_UPDATE)
    assert d.method == CONCAT_UPDATE
    d.register_entry('a', delay_step=5)
    _drive_delay(d, v, 12)
    np.testing.assert_allclose(np.asarray(d.at('a')), 7.0)
    np.testing.assert_allclose(np.asarray(d.retrieve(3)), 9.0)


def test_vardelay_concat_update_single_step():
    """max_length==1 hits the special-case concat branch."""
    bm.random.seed(0)
    v = bm.Variable(bm.zeros(2))
    d = VarDelay(v, time=0.1, method=CONCAT_UPDATE)  # length 1
    d.register_entry('b', delay_step=1)
    _drive_delay(d, v, 3)
    assert np.asarray(d.at('b')).shape == (2,)


# ===========================================================================
# delay.py -- DataDelay
# ===========================================================================

def test_datadelay_update_retrieve_reset():
    bm.random.seed(0)
    data = bm.Variable(bm.zeros(3))
    dd = DataDelay(data, data_init=bm.zeros, time=0.5)
    dd.register_entry('c', delay_step=3)
    dt = bm.get_dt()
    for i in range(8):
        bp.share.save(i=i, t=i * dt, dt=dt)
        dd.update(bm.ones(3) * i)
    np.testing.assert_allclose(np.asarray(dd.at('c')), 5.0)
    dd.reset_state()
    assert np.asarray(dd.data.value).sum() == pytest.approx(0.0)


def test_datadelay_reset_state_with_batch():
    bm.random.seed(0)
    data = bm.Variable(bm.zeros((1, 3)), batch_axis=0)
    dd = DataDelay(data, data_init=bm.zeros, time=0.5)
    dd.register_entry('c', delay_step=2)
    dd.reset_state(batch_size=1)
    assert dd.data is not None


# ===========================================================================
# delay.py -- DelayAccess
# ===========================================================================

def test_delay_access_update_and_reset():
    bm.random.seed(0)
    v = bm.Variable(bm.zeros(3))
    d = VarDelay(v, time=1.0)
    acc = DelayAccess(d, time=0.5, delay_entry='myacc')
    _drive_delay(d, v, 12)
    np.testing.assert_allclose(np.asarray(acc.update()), 7.0)
    acc.reset_state()  # no-op, just exercise the branch


def test_delay_access_with_indices():
    bm.random.seed(0)
    v = bm.Variable(bm.zeros(4))
    d = VarDelay(v, time=1.0)
    acc = DelayAccess(d, 0.3, 0, delay_entry='idxacc')
    _drive_delay(d, v, 10)
    assert np.asarray(acc.update()).shape == ()


# ===========================================================================
# delay.py -- init_delay_by_return
# ===========================================================================

def test_init_delay_by_return_variable():
    v = bm.Variable(bm.zeros(4))
    d = init_delay_by_return(v)
    assert isinstance(d, VarDelay)


def test_init_delay_by_return_returninfo_nonbatching():
    ri = ReturnInfo(size=(4,), batch_or_mode=bm.NonBatchingMode(), data=bm.zeros)
    d = init_delay_by_return(ri)
    assert isinstance(d, DataDelay)
    assert d.target.shape == (4,)


def test_init_delay_by_return_returninfo_int_batch():
    ri = ReturnInfo(size=(4,), batch_or_mode=2, data=bm.zeros)
    d = init_delay_by_return(ri)
    assert isinstance(d, DataDelay)
    assert d.target.shape == (2, 4)


def test_init_delay_by_return_returninfo_batchingmode():
    ri = ReturnInfo(size=(4,), batch_or_mode=bm.BatchingMode(3), data=bm.zeros)
    d = init_delay_by_return(ri)
    assert isinstance(d, DataDelay)
    assert d.target.shape == (3, 4)


def test_init_delay_by_return_returninfo_array_data():
    ri = ReturnInfo(size=(4,), batch_or_mode=bm.NonBatchingMode(),
                    data=jnp.ones((4,)))
    d = init_delay_by_return(ri)
    assert isinstance(d, DataDelay)


# ===========================================================================
# delay.py -- validation / error branches
# ===========================================================================

def test_vardelay_bad_target():
    with pytest.raises(ValueError):
        VarDelay(bm.zeros(3), time=1.0)  # plain array, not a Variable


def test_delay_bad_time_type():
    with pytest.raises(TypeError):
        VarDelay(bm.Variable(bm.zeros(3)), time='oops')


def test_vardelay_duplicate_entry():
    v = bm.Variable(bm.zeros(3))
    d = VarDelay(v, time=1.0)
    d.register_entry('e')
    with pytest.raises(KeyError):
        d.register_entry('e', delay_step=2)


def test_vardelay_at_missing_entry():
    v = bm.Variable(bm.zeros(3))
    d = VarDelay(v, time=1.0)
    with pytest.raises(KeyError):
        d.at('does-not-exist')


def test_get_delay_both_time_and_step():
    v = bm.Variable(bm.zeros(3))
    d = VarDelay(v, time=1.0)
    with pytest.raises(AssertionError):
        d.register_entry('z', delay_time=0.5, delay_step=3)


def test_init_delay_by_return_bad_type():
    with pytest.raises(TypeError):
        init_delay_by_return(123)


def test_init_delay_by_return_returninfo_bad_data():
    ri = ReturnInfo(size=(4,), batch_or_mode=bm.NonBatchingMode(), data=123)
    with pytest.raises(TypeError):
        init_delay_by_return(ri)


def test_base_delay_register_entry_not_implemented():
    d = Delay(time=1.0)
    with pytest.raises(NotImplementedError):
        d.register_entry('x', delay_step=1)


def test_base_delay_at_not_implemented():
    d = Delay(time=1.0)
    with pytest.raises(NotImplementedError):
        d.at('x')


def test_base_delay_retrieve_not_implemented():
    d = Delay(time=1.0)
    with pytest.raises(NotImplementedError):
        d.retrieve(1)


# ===========================================================================
# Extra coverage: input ops, multi-leaf inputs, memory-efficient w/o jit
# ===========================================================================

def test_runner_input_ops_minus_mul_div():
    """Exercise the '-', '*', '/' branches of ``_f_ops``."""
    n = _net(3)
    r = bp.DSRunner(
        n,
        monitors=['V'],
        inputs=[('V', 0.1, 'fix', '-'),
                ('V', 1.0, 'fix', '*'),
                ('V', 2.0, 'fix', '/')],
        progress_bar=False,
    )
    r.run(0.5)
    assert r.mon['V'].shape == (5, 3)


def test_runner_input_func_with_shared_arg():
    """Input callable that *accepts* a shared argument (deprecated bind path)."""
    n = _net(3)

    def fin(shared):
        n.V += 0.1

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = bp.DSRunner(n, monitors=['V'], inputs=fin, progress_bar=False)
        r.run(0.5)
    assert r.mon['V'].shape == (5, 3)


def test_runner_memory_efficient_without_jit():
    n = _net(3)
    r = bp.DSRunner(n, monitors=['V'], jit=False, progress_bar=False,
                    memory_efficient=True)
    r.run(0.5)
    assert r.mon['V'].shape == (5, 3)
    assert isinstance(r.mon['V'], np.ndarray)


def _TwoPop():
    class Two(bp.DynamicalSystem):
        def __init__(self):
            super().__init__()
            self.a = bp.dyn.LifRef(3)
            self.b = bp.dyn.LifRef(3)

        def update(self, x, y):
            s1 = self.a(x)
            self.b(y)
            return s1

    return Two()


def test_runner_multi_leaf_inputs_time_step():
    """A 2-leaf input tuple exercises the multi-leaf time-step branch."""
    net = _TwoPop()
    r = bp.DSRunner(net, monitors={'va': net.a.V}, progress_bar=False)
    xs = (np.ones((12, 3)) * 0.4, np.ones((12, 3)) * 0.2)
    out = r.predict(inputs=xs)
    assert r.mon['ts'].shape == (12,)
    assert np.asarray(out).shape == (12, 3)


# ===========================================================================
# Extra coverage: delay constructor entries, growth, batching, modes
# ===========================================================================

def test_vardelay_entries_in_constructor():
    """The ``entries=`` constructor argument registers entries (264-265)."""
    v = bm.Variable(bm.zeros(3))
    d = VarDelay(v, time=2.0, entries={'e1': 1.0, 'e2': 0.5})
    assert d._registered_entries['e1'] == 10
    assert d._registered_entries['e2'] == 5


def test_vardelay_register_entry_grows_max_length():
    """Registering an entry larger than current capacity grows the buffer."""
    v = bm.Variable(bm.zeros(3))
    d = VarDelay(v, time=0.5)  # 5 steps
    assert d.max_length == 5
    d.register_entry('big', delay_step=10)
    assert d.max_length == 10
    # data buffer was reallocated to the new length
    assert d.data.value.shape[0] == 10


def test_vardelay_training_mode_uses_concat():
    """A TrainingMode delay defaults to CONCAT_UPDATE (lines 95-96)."""
    v = bm.Variable(bm.zeros((2, 3)), batch_axis=0)
    d = VarDelay(v, time=0.5, mode=bm.training_mode)
    assert d.method == CONCAT_UPDATE


def test_vardelay_batching_mode_at_with_index():
    """BatchingMode delay: ``.at`` inserts a slice at the batch axis (319-321)."""
    bm.random.seed(0)
    v = bm.Variable(bm.zeros((2, 3)), batch_axis=0)
    d = VarDelay(v, time=0.5, mode=bm.batching_mode)
    d.register_entry('e', delay_step=3)
    dt = bm.get_dt()
    for i in range(6):
        bp.share.save(i=i, t=i * dt, dt=dt)
        v.value = bm.ones((2, 3)) * i
        d.update()
    assert np.asarray(d.at('e')).shape == (2, 3)
    # indexing into the feature axis keeps the batch dim
    assert np.asarray(d.at('e', 0)).shape == (2,)


def test_vardelay_retrieve_under_checking():
    """With checking enabled, ``retrieve`` runs the ``jit_error`` guard path."""
    was_checking = check.is_checking()
    check.turn_on()
    try:
        bm.random.seed(0)
        v = bm.Variable(bm.zeros(3))
        d = VarDelay(v, time=0.5)
        d.register_entry('e', delay_step=3)
        dt = bm.get_dt()
        for i in range(6):
            bp.share.save(i=i, t=i * dt, dt=dt)
            v.value = bm.ones(3) * i
            d.update()
        # the checked retrieve path returns a concrete delayed value
        np.testing.assert_allclose(np.asarray(d.at('e')), 3.0)
    finally:
        if not was_checking:
            check.turn_off()


def test_vardelay_unknown_update_method_raises():
    v = bm.Variable(bm.zeros(3))
    d = VarDelay(v, time=0.5)
    d.register_entry('e', delay_step=2)
    d.method = 'bogus'
    bp.share.save(i=0, t=0.0, dt=bm.get_dt())
    with pytest.raises(ValueError):
        d.update()


def test_vardelay_unknown_method_retrieve_raises():
    v = bm.Variable(bm.zeros(3))
    d = VarDelay(v, time=0.5)
    d.register_entry('e', delay_step=2)
    # populate first with a valid method
    dt = bm.get_dt()
    for i in range(4):
        bp.share.save(i=i, t=i * dt, dt=dt)
        v.value = bm.ones(3) * i
        d.update()
    d.method = 'bogus'
    bp.share.save(i=4, t=0.4, dt=dt)
    with pytest.raises(ValueError):
        d.retrieve(2)


def test_register_delay_by_return_reuses_instance():
    """``register_delay_by_return`` adds + reuses an after-update delay."""
    n = bp.dyn.LifRef(4)
    d1 = register_delay_by_return(n)
    d2 = register_delay_by_return(n)
    assert isinstance(d1, Delay)
    assert d1 is d2  # second call reuses the registered instance


def test_runner_fun_inputs_deprecated():
    """The deprecated ``fun_inputs`` argument still drives inputs (367, 562)."""
    n = _net(3)

    def finp(shared):
        n.V += 0.1

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = bp.DSRunner(n, monitors=['V'], fun_inputs=finp, progress_bar=False)
        r.run(0.5)
    assert r.mon['V'].shape == (5, 3)


def test_vardelay_zero_delay_at_with_index():
    """A zero-delay entry returns the *indexed* current target value (325)."""
    v = bm.Variable(bm.zeros(4))
    d = VarDelay(v, time=1.0)
    d.register_entry('zero', delay_time=None)
    bp.share.save(i=0, t=0.0, dt=bm.get_dt())
    v.value = bm.arange(4).astype(bm.float_)
    assert float(np.asarray(d.at('zero', 1))) == pytest.approx(1.0)


def test_vardelay_axis_names_sharding():
    """A target with axis_names builds a sharded delay buffer (244-246)."""
    v = bm.Variable(bm.zeros(4), axis_names=('feat',))
    d = VarDelay(v, time=0.5)
    assert d.data.value.shape == (5, 4)


def test_init_delay_by_return_returninfo_axis_names():
    """ReturnInfo carrying axis_names builds a DataDelay (asserts ndim, 559)."""
    ri = ReturnInfo(size=(4,), batch_or_mode=bm.NonBatchingMode(),
                    data=bm.zeros, axis_names=('feat',))
    d = init_delay_by_return(ri)
    assert isinstance(d, DataDelay)


# ===========================================================================
# Extra coverage: input formatting / _f_ops helpers (direct unit tests)
# ===========================================================================

def test_check_and_format_inputs_none():
    """``inputs=None`` produces empty formatted-input buckets (line 79)."""
    from brainpy.runners import check_and_format_inputs
    n = _net(3)
    res = check_and_format_inputs(n, None)
    assert set(res) == {'fixed', 'iterated', 'functional', 'array'}
    # everything empty
    assert all(len(lst) == 0 for d in res.values() for lst in d.values())


def test_check_and_format_inputs_nonstr_target():
    """A non-str / non-Variable target raises in absolute access (line 122)."""
    from brainpy.runners import check_and_format_inputs
    from brainpy._errors import RunningError
    n = _net(3)
    with pytest.raises(RunningError):
        check_and_format_inputs(n, [(123, 1.0)])


def test_f_ops_unknown_operation_raises():
    """``_f_ops`` with an unknown operation raises (line 207)."""
    from brainpy.runners import _f_ops
    n = _net(3)
    with pytest.raises(ValueError):
        _f_ops('^', n.V, 1.0)


def test_f_ops_all_supported_operations():
    """Each supported ``_f_ops`` branch runs without error."""
    from brainpy.runners import _f_ops
    v = bm.Variable(bm.ones(3))
    _f_ops('=', v, 2.0)
    np.testing.assert_allclose(np.asarray(v.value), 2.0)
    _f_ops('+', v, 1.0)
    np.testing.assert_allclose(np.asarray(v.value), 3.0)
    _f_ops('-', v, 1.0)
    np.testing.assert_allclose(np.asarray(v.value), 2.0)
    _f_ops('*', v, 2.0)
    np.testing.assert_allclose(np.asarray(v.value), 4.0)
    _f_ops('/', v, 4.0)
    np.testing.assert_allclose(np.asarray(v.value), 1.0)
