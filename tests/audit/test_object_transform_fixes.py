# -*- coding: utf-8 -*-
"""Regression + coverage tests for the BrainPy v2.7.8 object-transform audit
(see ``docs/issues-found-20260618.md``).

This module exercises the fixes recorded in the audit for the
``brainpy/math/object_transform`` package:

* ``jit.py``        — H-01 (``cls_jit`` no longer corrupts NEGATIVE
  ``static_argnums``/``donate_argnums`` by shifting them, which previously
  double-marked ``self`` static), M-02 (``donate_argnums`` shifted by +1 so
  ``self`` is never donated), H-04 (``bm.jit(fn, dyn_vars=..., child_objs=...)``
  pops the legacy kwargs with a ``DeprecationWarning`` instead of forwarding
  them into brainstate and raising ``TypeError``).
* ``controls.py``   — H-02 (``cond``/``for_loop``/``scan``/``while_loop``
  accept a ``Variable``/``Array`` in ``operands``), H-03 (``for_loop(jit=False)``
  with a zero-length pytree operand returns ``[]`` instead of crashing),
  M-03 (``scan`` returns ``(carry, ys)``), M-05 (``ifelse`` builds mutually
  exclusive conditions), M-06 (``while_loop`` body returning ``None`` raises).
* ``function.py``   — ``Partial``/``to_object`` behaviour and L-04 (``function``
  emits a ``DeprecationWarning``).
* ``_utils.py``     — ``warp_to_no_state_input_output`` strips/restores states.
* ``variables.py``  — C-25 (``var_dict`` round-trips through ``jax.jit``),
  C-26 (``Variable`` keeps ``batch_axis``/``axis_names`` through
  flatten/unflatten, ``jit``, ``grad``, ``vmap``), H-06 (``Variable.value``
  setter accepts a ``brainstate.State`` and a float64 numpy array and
  canonicalizes), H-45 (``size_without_batch`` returns a shape tuple).
* ``base.py``       — H-05 (``.cpu()`` moves variables and injects no junk
  attributes), H-08 (``register_implicit_vars`` accepts ``var_list``/``var_dict``).
* ``naming.py``     — H-07 (creating + discarding many named objects does not
  raise ``UniqueNameError`` and the ``_name2id`` registry stays bounded).

All tests assert the CORRECT post-fix behaviour. They use tiny array sizes so
the whole module runs in a few seconds.
"""

import gc
import importlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainstate

import brainpy as bp
import brainpy.math as bm
from brainpy._errors import MathError, UniqueNameError

# NOTE: ``from brainpy.math.object_transform import jit`` would bind the *jit
# function* (re-exported in the package ``__init__``), not the submodule. Import
# the submodule explicitly so we can monkeypatch its module-level ``jit`` that
# ``cls_jit`` calls into.
jit_module = importlib.import_module('brainpy.math.object_transform.jit')
from brainpy.math.object_transform import naming
from brainpy.math.object_transform.collectors import ArrayCollector
from brainpy.math.object_transform.variables import (
    Variable, TrainVar, Parameter, VariableView, VarList, VarDict,
)
from brainpy.math.object_transform.base import BrainPyObject, FunAsObject
from brainpy.math.object_transform._utils import (
    warp_to_no_state_input_output, infer_dyn_vars, get_brainpy_object,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _Obj(bp.BrainPyObject):
    """A tiny BrainPyObject owning a single Variable."""

    def __init__(self, n=3):
        super().__init__()
        self.v = bm.Variable(jnp.ones(n))


def _capture_cls_jit_argnums(static_argnums=None, donate_argnums=None):
    """Apply ``cls_jit`` and capture the ``static_argnums``/``donate_argnums``
    that it forwards into the underlying ``jit`` call, *without* actually
    invoking ``brainstate.transform.jit`` (which rejects negative argnums).
    """
    captured = {}
    orig_jit = jit_module.jit

    def spy(*args, **kwargs):
        captured['static_argnums'] = kwargs.get('static_argnums')
        captured['donate_argnums'] = kwargs.get('donate_argnums')
        raise _Stop()

    class _Stop(Exception):
        pass

    jit_module.jit = spy
    try:
        try:
            jit_module.cls_jit(
                static_argnums=static_argnums,
                donate_argnums=donate_argnums,
            )(lambda self, *a, **k: a)
        except _Stop:
            pass
    finally:
        jit_module.jit = orig_jit
    return captured['static_argnums'], captured['donate_argnums']


# ===========================================================================
# jit.py
# ===========================================================================

def test_cls_jit_positive_static_argnums_shifted_once():
    """H-01: a positive user index N is shifted to N+1 (account for ``self``),
    and ``self`` (index 0) is marked static exactly once."""
    static, donate = _capture_cls_jit_argnums(static_argnums=1)
    assert static == (0, 2)
    assert donate == ()


def test_cls_jit_negative_static_argnums_not_corrupted():
    """H-01: the historical bug shifted ``-1`` to ``0`` and produced
    ``(0, 0)`` (``self`` marked static twice + wrong target). The fix leaves
    negative indices unshifted, so ``self`` (0) appears exactly once and the
    negative index is preserved -- NOT collapsed into a duplicate ``0``."""
    static, donate = _capture_cls_jit_argnums(static_argnums=-1)
    assert static == (0, -1)
    # the corrupting outcome would have been (0, 0); make that explicit.
    assert static != (0, 0)
    assert static.count(0) == 1


def test_cls_jit_list_static_argnums_dedup_and_shift():
    """H-01: list of positive indices are each shifted by +1, ``self`` is
    prepended, and duplicates are removed."""
    static, _ = _capture_cls_jit_argnums(static_argnums=[0, 2])
    assert static == (0, 1, 3)


def test_cls_jit_donate_argnums_shifted_so_self_not_donated():
    """M-02: ``donate_argnums`` is shifted by +1, so a user index 1 becomes 2
    and ``self`` (index 0) is never donated."""
    static, donate = _capture_cls_jit_argnums(static_argnums=[0, 2], donate_argnums=1)
    assert donate == (2,)
    static2, donate2 = _capture_cls_jit_argnums(donate_argnums=[3, 4])
    assert donate2 == (4, 5)
    # default static is just (self,)
    assert static2 == (0,)


def test_cls_jit_runs_on_bound_method_with_positive_static():
    """H-01 end-to-end: a bound method jitted with a positive ``static_argnums``
    runs and mutates the owned Variable correctly."""

    class Prog(bp.BrainPyObject):
        def __init__(self):
            super().__init__()
            self.b = bm.Variable(jnp.zeros(2))

        # user index 0 (``scale``, a hashable int) is static; after the +1 shift
        # it becomes index 1 in the bound signature ``(self, scale, x)``.
        @bm.cls_jit(static_argnums=0)
        def run(self, scale, x):
            self.b.value = self.b.value + scale * x
            return self.b.value

    p = Prog()
    out = p.run(2, jnp.ones(2))
    np.testing.assert_allclose(np.asarray(out), [2., 2.])


def test_cls_jit_invalid_argnums_type_raises():
    with pytest.raises(ValueError):
        bm.cls_jit(static_argnums=1.5)(lambda self: None)
    with pytest.raises(ValueError):
        bm.cls_jit(donate_argnums=1.5)(lambda self: None)


def test_jit_dyn_vars_child_objs_deprecation_pops_kwargs():
    """H-04: ``dyn_vars``/``child_objs`` are no longer forwarded to brainstate
    (which would raise ``TypeError``); they are popped with a one-time
    ``DeprecationWarning`` and the function still works."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fn = bm.jit(lambda x: x + 1, dyn_vars=[], child_objs=[])
        result = fn(jnp.asarray(1.0))
    assert float(result) == 2.0
    categories = [w.category for w in caught]
    assert sum(issubclass(c, DeprecationWarning) for c in categories) >= 2


def test_jit_on_object_method():
    """``bm.jit`` JIT-compiles a BrainPyObject bound method."""

    class Hello(bp.BrainPyObject):
        def __init__(self):
            super().__init__()
            self.a = bm.Variable(jnp.asarray(10.))

        def transform(self):
            self.a.value = self.a.value * 2
            return self.a.value

    h = Hello()
    jfn = bm.jit(h.transform)
    assert float(jfn()) == 20.0
    assert float(jfn()) == 40.0


def test_jit_decorator_form_on_pure_function():
    @bm.jit
    def selu(x, alpha=1.67, lmbda=1.05):
        return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

    out = selu(jnp.asarray([1.0, -1.0]))
    assert out.shape == (2,)


def test_jit_static_argnums_on_pure_function():
    @bm.jit(static_argnums=(1,))
    def f(x, n):
        return x + n

    assert float(f(jnp.asarray(1.0), 5)) == 6.0


# ===========================================================================
# controls.py
# ===========================================================================

def test_cond_accepts_variable_operand():
    """H-02: a Variable passed in ``operands`` of ``cond`` is unwrapped to a
    raw jax array and does not raise."""
    a = bm.Variable(bm.zeros(2))

    def true_f(x):
        a.value = a.value + x
        return a.value

    def false_f(x):
        return a.value

    bm.cond(True, true_f, false_f, bm.Variable(bm.ones(2)))
    np.testing.assert_allclose(np.asarray(a.value), [1., 1.])


def test_cond_accepts_array_operand_and_scalar_operand():
    out = bm.cond(True, lambda x: x + 1, lambda x: x - 1, bm.asarray([1., 2.]))
    np.testing.assert_allclose(np.asarray(out), [2., 3.])
    # scalar operand (wrapped into a tuple internally)
    out2 = bm.cond(False, lambda x: x + 1, lambda x: x - 1, 5.0)
    assert float(out2) == 4.0


def test_for_loop_accepts_variable_operand():
    """H-02: ``bm.for_loop(lambda x: x+1, bm.arange(1,5))`` works with a
    BrainPy Array operand."""
    out = bm.for_loop(lambda x: x + 1, bm.arange(1, 5))
    np.testing.assert_allclose(np.asarray(out).ravel(), [2, 3, 4, 5])


def test_for_loop_variable_state_accumulation():
    a = bm.Variable(bm.zeros(1))
    b = bm.Variable(bm.ones(1))

    def body(x):
        a.value += x
        b.value *= x
        return a.value

    hist = bm.for_loop(body, operands=bm.arange(1, 5))
    np.testing.assert_allclose(np.asarray(hist).ravel(), [1, 3, 6, 10])
    np.testing.assert_allclose(np.asarray(a.value), [10.])
    np.testing.assert_allclose(np.asarray(b.value), [24.])


def test_for_loop_multiple_operands():
    a = bm.Variable(bm.zeros(1))

    def body(x, y):
        a.value += x + y
        return a.value

    hist = bm.for_loop(body, operands=(bm.arange(1, 5), bm.arange(2, 6)))
    assert np.asarray(hist).shape == (4, 1)


def test_for_loop_jit_false_normal_path():
    a = bm.Variable(bm.zeros(1))

    def body(x):
        a.value += x
        return a.value

    hist = bm.for_loop(body, bm.arange(1., 4.), jit=False)
    np.testing.assert_allclose(np.asarray(hist).ravel(), [1, 3, 6])


def test_for_loop_jit_false_zero_length_pytree_returns_empty():
    """H-03: ``for_loop(jit=False)`` with a zero-length *pytree* (dict) operand
    must not crash on ``operands[0].shape`` -- the leading length is computed
    from ``jax.tree.leaves``. It falls back to JIT mode (UserWarning) and
    returns an empty result."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = bm.for_loop(lambda d: d['x'] + 1, {'x': bm.arange(0.)}, jit=False)
    assert len(np.asarray(out)) == 0
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_for_loop_progress_bar_variants():
    out = bm.for_loop(lambda x: x + 1, bm.arange(1, 4), progress_bar=True)
    assert np.asarray(out).shape == (3,)
    out2 = bm.for_loop(lambda x: x + 1, bm.arange(1, 4), progress_bar=2)
    assert np.asarray(out2).shape == (3,)


def test_for_loop_invalid_progress_bar_raises():
    with pytest.raises(TypeError):
        bm.for_loop(lambda x: x + 1, bm.arange(1, 4), progress_bar="bad")


def test_scan_accepts_variable_operand_and_returns_carry_ys():
    """H-02 + M-03: ``scan`` accepts a Variable in operands and returns the
    documented ``(final_carry, stacked_ys)`` two-tuple."""
    carry, ys = bm.scan(lambda c, x: (c + x, c), 0., bm.Variable(bm.arange(1., 4.)))
    assert float(carry) == 6.0
    np.testing.assert_allclose(np.asarray(ys).ravel(), [0., 1., 3.])


def test_while_loop_accepts_variable_operand():
    """H-02: ``while_loop`` accepts a Variable operand (it is unwrapped to a raw
    jax array rather than being rejected at brainstate cache-key time). The body
    returns the updated operands, preserving the single-operand tuple structure."""
    res = bm.while_loop(lambda x: (x + 1.,), lambda x: x < 3., (bm.Variable(jnp.asarray(0.)),))
    assert float(np.asarray(res[0])) == 3.0


def test_while_loop_state_mutation():
    a = bm.Variable(bm.zeros(1))
    b = bm.Variable(bm.ones(1))

    def cond_f(x, y):
        return x < 6.

    def body_f(x, y):
        a.value += x
        b.value *= y
        return x + b[0], y + 1.

    res = bm.while_loop(body_f, cond_f, operands=(1., 1.))
    assert len(res) == 2


def test_while_loop_body_returning_none_raises():
    """M-06: a ``while_loop`` body that returns ``None`` would freeze the carry
    and loop forever -- it must raise a clear ``ValueError`` instead."""

    def body(x):
        # returns None -> illegal
        pass

    with pytest.raises(ValueError):
        bm.while_loop(body, lambda x: x < 3., 0.)


def test_ifelse_callable_branches_mutually_exclusive():
    """M-05: ``ifelse`` resolves to the first matching branch (mutually
    exclusive conditions, default branch last)."""

    def f(a):
        return bm.ifelse(
            conditions=[a > 10, a > 5, a > 2, a > 0],
            branches=[lambda: 1, lambda: 2, lambda: 3, lambda: 4, lambda: 5],
        )

    assert int(f(11)) == 1
    assert int(f(7)) == 2
    assert int(f(1)) == 4
    assert int(f(-5)) == 5


def test_ifelse_non_callable_branches():
    def f(a):
        return bm.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
                         branches=[1, 2, 3, 4, 5])

    assert int(f(3)) == 3


def test_ifelse_with_operands_and_variable():
    out = bm.ifelse(
        conditions=[True],
        branches=[lambda x: x + 1, lambda x: x - 1],
        operands=bm.Variable(jnp.asarray(10.)),
    )
    assert float(out) == 11.0


# ===========================================================================
# _utils.py
# ===========================================================================

def test_warp_to_no_state_input_output_strips_and_restores():
    """``warp_to_no_state_input_output`` removes ``State`` wrappers from inputs
    and outputs, passing through plain jax arrays."""

    def fn(x):
        # the wrapper should have unwrapped the State to a jax array
        assert not isinstance(x, brainstate.State)
        return x * 2

    wrapped = warp_to_no_state_input_output(fn)
    out = wrapped(bm.Variable(jnp.asarray(3.)))
    assert not isinstance(out, brainstate.State)
    assert float(out) == 6.0


def test_warp_to_no_state_passes_missing_through():
    missing = brainstate.typing.Missing()
    assert warp_to_no_state_input_output(missing) is missing


def test_infer_dyn_vars_and_get_brainpy_object():
    obj = _Obj()
    dv = infer_dyn_vars(obj)
    assert len(dv) >= 1
    mapping = get_brainpy_object(obj)
    assert obj.name in mapping
    # bound method path
    bound = infer_dyn_vars(obj.vars)
    assert isinstance(bound, ArrayCollector)
    # non-object path returns empty collector / dict
    assert len(infer_dyn_vars(lambda: None)) == 0
    assert get_brainpy_object(lambda: None) == {}


# ===========================================================================
# function.py
# ===========================================================================

def test_partial_binds_positional_args():
    add = bm.Partial(lambda x, y: x + y, 1)
    assert add(2) == 3


def test_partial_keyword_override_and_is_brainpy_object():
    p = bm.Partial(lambda x, scale=1.: x * scale, scale=2.)
    assert isinstance(p, FunAsObject)
    assert float(p(3.)) == 6.0
    # call-time keyword overrides bound keyword
    assert float(p(3., scale=10.)) == 30.0


def test_partial_tracks_variables():
    sub = _Obj()
    p = bm.Partial(lambda: sub.v.value, child_objs=sub)
    np.testing.assert_allclose(np.asarray(p()), np.ones(3))
    assert sub.name in p.nodes()


def test_to_object_decorator_and_direct():
    sub = _Obj()
    obj = bm.to_object(lambda: sub.v.value, child_objs=sub)
    np.testing.assert_allclose(np.asarray(obj()), np.ones(3))

    @bm.to_object(child_objs=sub)
    def fn():
        return sub.v.value

    np.testing.assert_allclose(np.asarray(fn()), np.ones(3))


def test_to_object_requires_child_objs_when_f_given():
    with pytest.raises(ValueError):
        bm.to_object(lambda: 1.)


def test_function_is_deprecated():
    """L-04: ``function`` is deprecated in favour of ``to_object``."""
    sub = _Obj()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        obj = bm.function(lambda: sub.v.value, nodes=sub)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    np.testing.assert_allclose(np.asarray(obj()), np.ones(3))


# ===========================================================================
# variables.py
# ===========================================================================

def test_var_dict_round_trips_through_jax_jit():
    """C-25: ``VarDict.tree_unflatten`` no longer uses the removed ``jax.util``;
    a ``var_dict`` survives ``jax.jit``."""
    d = bm.var_dict({'a': bm.Variable(jnp.ones(2))})
    out = jax.jit(lambda dd: dd)(d)
    assert isinstance(out, VarDict)
    assert list(out.keys()) == ['a']
    np.testing.assert_allclose(np.asarray(out['a'].value), np.ones(2))


def test_var_dict_tree_map():
    d = bm.var_dict({'a': bm.Variable(jnp.ones(2)), 'b': bm.Variable(jnp.zeros(2))})
    leaves = jax.tree.leaves(d)
    assert len(leaves) == 2


def test_var_list_round_trips_through_pytree():
    vl = bm.var_list([bm.Variable(jnp.ones(1)), bm.Variable(jnp.zeros(2))])
    leaves, treedef = jax.tree.flatten(vl)
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert isinstance(rebuilt, VarList)
    assert len(rebuilt) == 2


def test_variable_keeps_metadata_through_flatten_unflatten():
    """C-26: ``batch_axis``/``axis_names`` survive a manual pytree round-trip."""
    v = bm.Variable(jnp.zeros((3, 4)), batch_axis=0, axis_names=('a', 'b'))
    assert v.batch_axis == 0
    assert v.axis_names == ('a', 'b')
    leaves, treedef = jax.tree.flatten(v)
    v2 = jax.tree.unflatten(treedef, leaves)
    assert isinstance(v2, Variable)
    assert v2.batch_axis == 0
    assert v2.axis_names == ('a', 'b')


def test_variable_keeps_metadata_through_jit():
    """C-26: metadata preserved through ``jax.jit``."""
    v = bm.Variable(jnp.zeros((3, 4)), batch_axis=0, axis_names=('a', 'b'))
    out = jax.jit(lambda x: x)(v)
    assert out.batch_axis == 0
    assert out.axis_names == ('a', 'b')


def test_variable_keeps_metadata_through_grad():
    """C-26: metadata preserved through ``jax.grad``."""
    v = bm.Variable(jnp.ones((3, 4)), batch_axis=0, axis_names=('a', 'b'))
    g = jax.grad(lambda x: jnp.sum(x.value ** 2))(v)
    assert isinstance(g, Variable)
    assert g.batch_axis == 0
    assert g.axis_names == ('a', 'b')


def test_variable_keeps_metadata_through_vmap():
    """C-26: metadata preserved through ``jax.vmap``."""
    v = bm.Variable(jnp.zeros((3, 4)), batch_axis=0, axis_names=('a', 'b'))
    out = jax.vmap(lambda x: x)(v)
    assert isinstance(out, Variable)
    assert out.batch_axis == 0


def test_variable_value_setter_accepts_brainstate_state():
    """H-06: ``Variable.value = some_State`` unwraps the state first."""
    v = bm.Variable(jnp.zeros(3, dtype=jnp.float32))
    st = brainstate.State(jnp.ones(3, dtype=jnp.float32))
    v.value = st
    np.testing.assert_allclose(np.asarray(v.value), np.ones(3))


def test_variable_value_setter_canonicalizes_float64_numpy():
    """H-06: assigning a float64 numpy array to a float32 Variable canonicalizes
    (converts) it rather than raising a spurious dtype ``MathError``."""
    v = bm.Variable(jnp.zeros(3, dtype=jnp.float32))
    arr = np.ones(3, dtype=np.float64)
    v.value = arr  # must not raise
    assert v.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(v.value), np.ones(3))


def test_variable_value_setter_accepts_brainpy_array():
    v = bm.Variable(jnp.zeros(3, dtype=jnp.float32))
    v.value = bm.asarray(np.arange(3), dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(v.value), [0., 1., 2.])


def test_variable_value_setter_shape_mismatch_raises():
    v = bm.Variable(jnp.zeros(3))
    with pytest.raises(MathError):
        v.value = jnp.zeros(4)


def test_size_without_batch_returns_shape_tuple():
    """H-45: ``size_without_batch`` returns a *shape tuple* (drops the batch
    axis), not an integer element count."""
    v = bm.Variable(jnp.zeros((3, 4)), batch_axis=0)
    assert v.size_without_batch == (4,)
    v2 = bm.Variable(jnp.zeros((3, 4)))
    assert v2.size_without_batch == (3, 4)


def test_variable_batch_size_and_axis():
    v = bm.Variable(jnp.zeros((5, 4)), batch_axis=0)
    assert v.batch_size == 5
    v2 = bm.Variable(jnp.zeros(4))
    assert v2.batch_size is None


def test_variable_batch_axis_immutable():
    v = bm.Variable(jnp.zeros((5, 4)), batch_axis=0)
    with pytest.raises(ValueError):
        v.batch_axis = 1
    with pytest.raises(ValueError):
        v.batch_size = 2


def test_variable_invalid_batch_axis_raises():
    with pytest.raises(MathError):
        bm.Variable(jnp.zeros(3), batch_axis=5)


def test_variable_init_from_size_and_hash():
    v = bm.Variable(4)
    assert v.shape == (4,)
    np.testing.assert_allclose(np.asarray(v.value), np.zeros(4))
    # identity hash
    assert hash(v) == id(v)
    s = {v, v}
    assert len(s) == 1


def test_trainvar_and_parameter_are_variables():
    tv = bm.TrainVar(jnp.zeros(2))
    par = bm.Parameter(jnp.ones(2))
    assert isinstance(tv, Variable)
    assert isinstance(par, Variable)


def test_variable_view_reads_and_writes_origin():
    origin = bm.Variable(jnp.arange(5.))
    view = bm.VariableView(origin, slice(None, 2, None))
    np.testing.assert_allclose(np.asarray(view.value), [0., 1.])
    view.value = jnp.asarray([10., 11.])
    np.testing.assert_allclose(np.asarray(origin.value)[:2], [10., 11.])
    assert 'VariableView' in repr(view)


def test_variable_view_requires_variable():
    with pytest.raises(ValueError):
        bm.VariableView(jnp.zeros(3), slice(None))


# ===========================================================================
# base.py
# ===========================================================================

def test_cpu_moves_variable_and_injects_no_junk_attrs():
    """H-05: ``.cpu()`` iterates real Variables and moves them; it must not add
    dict-valued junk attributes named after nodes."""
    obj = _Obj()
    before = set(obj.__dict__.keys())
    returned = obj.cpu()
    after = set(obj.__dict__.keys())
    assert after == before  # no junk attributes injected
    assert returned is obj
    assert isinstance(obj.v, Variable)
    np.testing.assert_allclose(np.asarray(obj.v.value), np.ones(3))


def test_to_moves_variable_to_device():
    obj = _Obj()
    dev = jax.devices('cpu')[0]
    obj.to(device=dev)
    assert isinstance(obj.v, Variable)


def test_register_implicit_vars_with_var_list():
    """H-08: ``register_implicit_vars(var_list([...]))`` flattens the container
    into the ``ArrayCollector`` (which only accepts plain Variables)."""
    obj = _Obj()
    obj.register_implicit_vars(bm.var_list([bm.Variable(jnp.zeros(2))]))
    assert len(obj.implicit_vars) == 1
    for v in obj.implicit_vars.values():
        assert isinstance(v, Variable)


def test_register_implicit_vars_with_var_dict():
    obj = _Obj()
    obj.register_implicit_vars(bm.var_dict({'x': bm.Variable(jnp.zeros(2)),
                                            'y': bm.Variable(jnp.zeros(2))}))
    assert len(obj.implicit_vars) == 2
    for v in obj.implicit_vars.values():
        assert isinstance(v, Variable)


def test_register_implicit_vars_plain_and_named():
    obj = _Obj()
    obj.register_implicit_vars(bm.Variable(jnp.zeros(1)), named=bm.Variable(jnp.zeros(1)))
    assert len(obj.implicit_vars) == 2


def test_register_implicit_vars_rejects_non_variable():
    obj = _Obj()
    with pytest.raises(ValueError):
        obj.register_implicit_vars(123)


def test_vars_collects_variables_and_containers():
    class Multi(bp.BrainPyObject):
        def __init__(self):
            super().__init__()
            self.a = bm.Variable(jnp.zeros(1))
            self.lst = bm.var_list([bm.Variable(jnp.zeros(1))])
            self.dct = bm.var_dict({'k': bm.Variable(jnp.zeros(1))})

    m = Multi()
    collected = m.vars()
    assert len(collected) == 3
    assert len(m.train_vars()) == 0


def test_nodes_and_state_dict_round_trip():
    class Parent(bp.BrainPyObject):
        def __init__(self):
            super().__init__()
            self.child = _Obj()
            self.w = bm.Variable(jnp.zeros(2))

    p = Parent()
    nodes = p.nodes()
    assert p.name in nodes
    sd = p.state_dict()
    assert isinstance(sd, dict)
    # load it back
    p.load_state_dict(sd, warn=False)


def test_brainpyobject_setattr_updates_variable_value():
    """Setting an attribute that is a Variable updates its value in place."""
    obj = _Obj(n=2)
    vid = id(obj.v)
    obj.v = jnp.asarray([5., 6.])
    assert id(obj.v) == vid  # same Variable object
    np.testing.assert_allclose(np.asarray(obj.v.value), [5., 6.])


def test_funasobject_call_and_repr():
    sub = _Obj()
    fo = FunAsObject(target=lambda: sub.v.value, child_objs=sub)
    np.testing.assert_allclose(np.asarray(fo()), np.ones(3))
    assert 'FunAsObject' in repr(fo)


def test_objecttransform_base():
    from brainpy.math.object_transform.base import ObjectTransform
    ot = ObjectTransform()
    assert repr(ot) == 'ObjectTransform'
    with pytest.raises(NotImplementedError):
        ot()


def test_node_list_and_node_dict():
    from brainpy.math.object_transform.base import node_list, node_dict
    nl = node_list([_Obj(), _Obj()])
    assert len(nl) == 2
    nd = node_dict({'a': _Obj()})
    assert 'a' in nd


def test_tracing_variable_raises_not_implemented():
    """L-06: ``tracing_variable`` is deprecated and always raises."""
    obj = _Obj()
    with pytest.raises(NotImplementedError):
        obj.tracing_variable('w', jnp.zeros, (2,))


# ===========================================================================
# naming.py
# ===========================================================================

def test_many_named_objects_do_not_raise_unique_name_error():
    """H-07: creating and discarding many named objects must not raise
    ``UniqueNameError`` from reused ids."""
    bm.clear_name_cache()
    for _ in range(300):
        _Obj()  # transient, immediately discarded
    gc.collect()
    # creating more after GC must still not collide
    keep = [_Obj() for _ in range(5)]
    assert len(keep) == 5


def test_name2id_registry_stays_bounded():
    """H-07: the ``_name2id`` registry prunes dead weak refs and stays bounded
    instead of growing unboundedly."""
    bm.clear_name_cache()
    for _ in range(400):
        _Obj()
    gc.collect()
    # all transient objects collected -> registry pruned back to (near) empty.
    assert len(naming._name2id) <= 5


def test_explicit_duplicate_name_raises_unique_name_error():
    bm.clear_name_cache()
    keep = _Obj()
    keep.name = 'my_unique_name'
    other = _Obj()
    with pytest.raises(UniqueNameError):
        other.name = 'my_unique_name'


def test_invalid_identifier_name_raises():
    obj = _Obj()
    with pytest.raises(bp.errors.BrainPyError):
        obj.name = '123 not valid'


def test_get_unique_name_increments():
    n1 = naming.get_unique_name('Foo')
    n2 = naming.get_unique_name('Foo')
    assert n1 != n2
    assert n1.startswith('Foo')


def test_clear_name_cache_warns_when_requested():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bm.clear_name_cache(ignore_warn=False)
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_stack_cache_helpers():
    def fn():
        return None

    naming.cache_stack(fn, [1, 2, 3])
    assert naming.get_stack_cache(fn) == [1, 2, 3]
    assert naming.get_stack_cache(lambda: None) is None
    naming.clear_stack_cache()
    assert naming.get_stack_cache(fn) is None


# ===========================================================================
# collectors (ArrayCollector) exercised via base/vars APIs
# ===========================================================================

def test_array_collector_basic_ops():
    ac = ArrayCollector()
    v = bm.Variable(jnp.zeros(1))
    ac['x'] = v
    assert len(ac.unique()) == 1
    assert 'x' in ac.dict()
    # subset by type
    tv = bm.TrainVar(jnp.zeros(1))
    ac['t'] = tv
    assert len(ac.subset(TrainVar)) == 1


# ===========================================================================
# Additional coverage for variables.py
# ===========================================================================

def test_variable_init_from_list_of_int_size():
    v = bm.Variable([2, 3])
    assert v.shape == (2, 3)


def test_variable_axis_names_inserts_batch_axis_name():
    """The axis_names insertion path: when ``len(axis_names) + 1 == ndim`` the
    batch-axis name is inserted at ``batch_axis``."""
    from brainpy.math.sharding import BATCH_AXIS
    v = bm.Variable(jnp.zeros((5, 4)), batch_axis=0, axis_names=('feat',))
    assert len(v.axis_names) == 2
    assert v.axis_names[0] == BATCH_AXIS
    assert v.axis_names[1] == 'feat'


def test_variable_value_setter_with_batch_axis():
    v = bm.Variable(jnp.zeros((5, 4)), batch_axis=0)
    v.value = jnp.ones((7, 4))  # different batch size OK
    assert v.shape == (7, 4)
    with pytest.raises(MathError):
        v.value = jnp.ones((5, 9))  # non-batch dim mismatch


def test_variable_value_setter_dtype_mismatch_raises():
    v = bm.Variable(jnp.zeros(3, dtype=jnp.float32))
    with pytest.raises(MathError):
        v.value = jnp.zeros(3, dtype=jnp.int32)


def test_var_list_append_rejects_non_variable():
    vl = bm.var_list()
    with pytest.raises(TypeError):
        vl.append(jnp.zeros(2))


def test_var_list_setitem_int_updates_value_slice_replaces():
    a = bm.Variable(jnp.zeros(1))
    b = bm.Variable(jnp.zeros(2))
    vl = bm.var_list([a, b])
    ids = (id(vl[0]), id(vl[1]))
    vl[0] = jnp.ones(1)  # updates value in place, keeps the same Variable
    assert id(vl[0]) == ids[0]
    np.testing.assert_allclose(np.asarray(vl[0].value), [1.])
    # slice assignment replaces entries
    vl[0:1] = [bm.Variable(jnp.zeros(1))]
    assert len(vl) == 2


def test_var_dict_update_existing_key_sets_value():
    d = bm.var_dict({'a': bm.Variable(jnp.zeros(2))})
    original_id = id(d['a'])
    d['a'] = jnp.ones(2)  # update value, keep the same Variable
    assert id(d['a']) == original_id
    np.testing.assert_allclose(np.asarray(d['a'].value), np.ones(2))


def test_var_dict_rejects_non_variable_element():
    with pytest.raises(TypeError):
        bm.var_dict({'a': jnp.zeros(2)})


def test_var_dict_update_from_tuple_and_kwargs():
    d = bm.var_dict(('a', bm.Variable(jnp.zeros(1))), b=bm.Variable(jnp.zeros(1)))
    assert set(d.keys()) == {'a', 'b'}


def test_variable_view_setter_shape_and_dtype_checks():
    origin = bm.Variable(jnp.arange(5.))
    view = bm.VariableView(origin, slice(None, 2, None))
    with pytest.raises(MathError):
        view.value = jnp.zeros(3)  # wrong shape


# ===========================================================================
# Additional coverage for base.py
# ===========================================================================

def test_relative_vars_and_nodes():
    class Parent(bp.BrainPyObject):
        def __init__(self):
            super().__init__()
            self.child = _Obj()
            self.w = bm.Variable(jnp.zeros(2))

    p = Parent()
    rel_vars = p.vars(method='relative')
    assert len(rel_vars) >= 2
    rel_nodes = p.nodes(method='relative')
    assert '' in rel_nodes


def test_vars_with_level_zero():
    class Parent(bp.BrainPyObject):
        def __init__(self):
            super().__init__()
            self.child = _Obj()
            self.w = bm.Variable(jnp.zeros(2))

    p = Parent()
    # level=0 -> only self's own variable, not the child's
    own = p.vars(level=0)
    assert len(own) == 1


def test_register_implicit_nodes_variants():
    parent = _Obj()
    parent.register_implicit_nodes(_Obj())
    parent.register_implicit_nodes([_Obj(), _Obj()])
    parent.register_implicit_nodes({'k': _Obj()})
    parent.register_implicit_nodes(named=_Obj())
    # nodes are keyed by name; distinct objects -> at least 5 distinct entries.
    assert len(parent.implicit_nodes) >= 5


def test_register_implicit_nodes_rejects_bad_type():
    parent = _Obj()
    with pytest.raises(ValueError):
        parent.register_implicit_nodes(123)


def test_register_implicit_vars_from_list_and_dict_args():
    obj = _Obj()
    obj.register_implicit_vars([bm.Variable(jnp.zeros(1)), bm.Variable(jnp.zeros(1))])
    obj.register_implicit_vars({'k': bm.Variable(jnp.zeros(1))})
    assert len(obj.implicit_vars) == 3
    with pytest.raises(ValueError):
        obj.register_implicit_vars([123])
    with pytest.raises(ValueError):
        obj.register_implicit_vars({'bad': 123})
    with pytest.raises(ValueError):
        obj.register_implicit_vars(named=123)


def test_node_dict_check_unique_raises_on_conflict():
    from brainpy.math.object_transform.base import NodeDict
    a, b = _Obj(), _Obj()
    nd = NodeDict(check_unique=True)
    nd['k'] = a
    nd['k'] = a  # same object, OK
    with pytest.raises(KeyError):
        nd['k'] = b  # different object under same key


def test_node_list_and_node_dict_in_find_nodes():
    from brainpy.math.object_transform.base import node_list, node_dict

    class Parent(bp.BrainPyObject):
        def __init__(self):
            super().__init__()
            self.lst = node_list([_Obj(), _Obj()])
            self.dct = node_dict({'a': _Obj()})

    p = Parent()
    nodes = p.nodes()
    # parent + 2 list children + 1 dict child
    assert len(nodes) >= 4
    rel_nodes = p.nodes(method='relative')
    assert len(rel_nodes) >= 4


def test_funasobject_repr_with_nodes():
    sub = _Obj()
    fo = FunAsObject(target=lambda: sub.v.value, child_objs=sub,
                     dyn_vars=bm.Variable(jnp.zeros(1)))
    r = repr(fo)
    assert 'FunAsObject' in r
    assert 'num_of_vars' in r


def test_save_and_load_state_methods():
    obj = _Obj(n=2)
    sd = obj.save_state()
    assert isinstance(sd, dict)
    missing, unexpected = obj.load_state(sd)
    assert missing == [] and unexpected == []


def test_vars_invalid_method_raises():
    obj = _Obj()
    with pytest.raises(ValueError):
        obj.nodes(method='bad')


def test_brainpyobject_tree_flatten_unflatten():
    class Mixed(bp.BrainPyObject):
        def __init__(self):
            super().__init__()
            self.v = bm.Variable(jnp.ones(2))   # dynamic
            self.lst = bm.var_list([bm.Variable(jnp.zeros(1))])  # dynamic container
            self.scalar = 7                       # static

    m = Mixed()
    leaves, treedef = jax.tree.flatten(m)
    assert len(leaves) >= 1
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert isinstance(rebuilt, Mixed)
    assert rebuilt.scalar == 7


def test_brainpyobject_setattr_method_bypasses_variable_update():
    obj = _Obj()
    # ``.setattr`` is the explicit object.__setattr__ wrapper
    obj.setattr('plain', 123)
    assert obj.plain == 123


def test_load_state_dict_v1_and_warnings():
    obj = _Obj(n=2)
    # build a v1-style flat state dict
    flat = {k: np.asarray(v.value) for k, v in obj.vars().items()}
    res = obj.load_state_dict(flat, compatible='v1', warn=False)
    assert res.missing_keys == [] and res.unexpected_keys == []
    # unexpected + missing keys trigger warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        obj.load_state_dict({'does_not_exist': np.zeros(2)}, compatible='v1', warn=True)
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_load_state_dict_invalid_compatible_raises():
    obj = _Obj()
    with pytest.raises(ValueError):
        obj.load_state_dict({}, compatible='v9')


def test_vars_exclude_types():
    class Holder(bp.BrainPyObject):
        def __init__(self):
            super().__init__()
            self.tv = bm.TrainVar(jnp.zeros(1))
            self.v = bm.Variable(jnp.zeros(1))

    h = Holder()
    # excluding TrainVar drops it from the collection
    kept = h.vars(exclude_types=(TrainVar,))
    assert all(not isinstance(v, TrainVar) for v in kept.values())


def test_unique_name_with_explicit_type():
    obj = _Obj()
    name = obj.unique_name(type_='CustomType')
    assert name.startswith('CustomType')


def test_node_dict_update_from_tuple():
    from brainpy.math.object_transform.base import NodeDict
    nd = NodeDict(('a', _Obj()))
    assert 'a' in nd


def test_brainpyobject_tree_flatten_unflatten_direct():
    """Exercise ``tree_flatten``/``tree_unflatten`` directly (the pytree
    registration is gated off by default, so ``jax.tree`` would treat the
    object as a leaf)."""
    class Mixed(bp.BrainPyObject):
        def __init__(self):
            super().__init__()
            self.v = bm.Variable(jnp.ones(2))
            self.scalar = 9

    m = Mixed()
    dynamic, aux = m.tree_flatten()
    assert len(dynamic) == 1
    rebuilt = Mixed.tree_unflatten(aux, dynamic)
    assert rebuilt.scalar == 9
    assert isinstance(rebuilt.v, Variable)


def test_relative_nodes_nested_hierarchy():
    """Cover the relative-method recursion that joins child paths."""
    class Leaf(bp.BrainPyObject):
        def __init__(self):
            super().__init__()
            self.w = bm.Variable(jnp.zeros(1))

    class Middle(bp.BrainPyObject):
        def __init__(self):
            super().__init__()
            self.leaf = Leaf()

    class Top(bp.BrainPyObject):
        def __init__(self):
            super().__init__()
            self.mid = Middle()

    t = Top()
    rel = t.nodes(method='relative')
    # joined paths like 'mid.leaf' appear
    assert any('.' in k for k in rel.keys() if k)
    rel_vars = t.vars(method='relative')
    assert len(rel_vars) >= 1


def test_cuda_tpu_raise_without_device():
    obj = _Obj()
    with pytest.raises(RuntimeError):
        obj.cuda()
    with pytest.raises(RuntimeError):
        obj.tpu()
