# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy/math/object_transform/collectors.py``.

Exercises the ``Collector`` / ``ArrayCollector`` (alias ``TensorCollector``) dict
subclasses:

* ``Collector.__setitem__`` conflict detection (same key, different value).
* ``replace`` (pop + reinsert).
* ``update`` from dict / list / tuple / kwargs and the assertion branch.
* ``__add__`` (merge) and ``__sub__`` (dict removal + list/tuple removal +
  error branches: removing missing keys, value mismatches, bad ``other`` type).
* the type-filtering helpers ``subset`` / ``not_subset`` / ``include`` /
  ``exclude`` and ``unique`` (dedup by ``id``).
* ``ArrayCollector.__setitem__`` value-type assertion + conflict detection,
  ``dict`` and ``data`` accessors, and the registered pytree flatten/unflatten.

Tiny inputs throughout; both happy and error paths are covered.

NOTE: two source lines stay uncovered because they are dead code -- the trailing
``else: raise ValueError`` in ``update`` (line 54) and ``else: raise KeyError``
in ``__sub__`` (line 124) sit *after* an ``isinstance`` ``assert`` / guard that
already rejects every non dict/list/tuple input, so the ``else`` can never run.
"""

import jax
import pytest

import brainpy.math as bm
from brainpy.math.object_transform.collectors import (
    Collector, ArrayCollector, TensorCollector,
)
from brainpy.math.object_transform.variables import Variable, TrainVar


# ---------------------------------------------------------------------------
# Collector.__setitem__ / replace
# ---------------------------------------------------------------------------

def test_setitem_same_value_no_conflict():
    c = Collector()
    obj = object()
    c['a'] = obj
    c['a'] = obj  # same id -> allowed (no error)
    assert c['a'] is obj


def test_setitem_conflict_raises():
    c = Collector()
    c['a'] = 1
    with pytest.raises(ValueError):
        c['a'] = 2  # different value for the same key


def test_replace_swaps_value():
    c = Collector()
    c['a'] = 1
    c.replace('a', 99)
    assert c['a'] == 99
    # replace removes then re-adds, so no conflict error
    c.replace('a', 100)
    assert c['a'] == 100


# ---------------------------------------------------------------------------
# Collector.update
# ---------------------------------------------------------------------------

def test_update_with_dict():
    c = Collector({'a': 1})
    ret = c.update({'b': 2, 'c': 3})
    assert ret is c               # update returns self
    assert c == {'a': 1, 'b': 2, 'c': 3}


def test_update_with_list_uses_var_keys():
    c = Collector({'x': 0})       # len(self) == 1
    c.update([10, 20])            # -> _var1, _var2 (offset by num=1)
    assert c['_var1'] == 10
    assert c['_var2'] == 20


def test_update_with_tuple_and_kwargs():
    c = Collector()
    c.update((5, 6), extra=7)
    assert c['_var0'] == 5
    assert c['_var1'] == 6
    assert c['extra'] == 7


def test_update_rejects_non_collection():
    c = Collector()
    with pytest.raises(AssertionError):
        c.update(123)             # not a dict/list/tuple


# ---------------------------------------------------------------------------
# Collector.__add__
# ---------------------------------------------------------------------------

def test_add_merges_two_collectors():
    a = Collector({'a': 1})
    b = Collector({'b': 2})
    merged = a + b
    assert isinstance(merged, Collector)
    assert merged == {'a': 1, 'b': 2}
    # originals untouched
    assert a == {'a': 1}
    assert b == {'b': 2}


# ---------------------------------------------------------------------------
# Collector.__sub__
# ---------------------------------------------------------------------------

def test_sub_with_dict_removes_matching_pair():
    obj = object()
    c = Collector({'a': obj, 'b': 2})
    res = c - {'a': obj}
    assert res == {'b': 2}


def test_sub_with_dict_value_mismatch_raises():
    c = Collector({'a': 1})
    with pytest.raises(ValueError):
        c - {'a': 999}            # same key, different value -> error


def test_sub_with_dict_missing_key_raises():
    c = Collector({'a': 1})
    with pytest.raises(ValueError):
        c - {'missing': 1}


def test_sub_with_list_of_keys():
    c = Collector({'a': 1, 'b': 2, 'c': 3})
    res = c - ['a', 'c']
    assert res == {'b': 2}


def test_sub_with_list_of_values_by_identity():
    shared = object()
    c = Collector({'a': shared, 'b': 2})
    # passing the value object removes every key mapping to that id
    res = c - [shared]
    assert res == {'b': 2}


def test_sub_with_list_missing_key_raises():
    c = Collector({'a': 1})
    with pytest.raises(ValueError):
        c - ['nope']


def test_sub_with_list_missing_value_raises():
    # P4-M3: removing a *value* object that is not present must raise the same
    # descriptive ValueError as the string-key path, not a bare KeyError(id).
    present = object()
    absent = object()
    c = Collector({'a': present})
    with pytest.raises(ValueError):
        c - [absent]


def test_sub_rejects_bad_type():
    c = Collector({'a': 1})
    with pytest.raises(ValueError):
        c - 5                     # not dict/tuple/list


# ---------------------------------------------------------------------------
# subset / not_subset / include / exclude / unique
# ---------------------------------------------------------------------------

def test_subset_and_not_subset_by_isinstance():
    v = Variable(bm.zeros(2))
    tv = TrainVar(bm.zeros(2))
    c = Collector({'v': v, 'tv': tv, 'n': 3})
    # TrainVar is a subclass of Variable
    var_sub = c.subset(Variable)
    assert set(var_sub.keys()) == {'v', 'tv'}
    train_sub = c.subset(TrainVar)
    assert set(train_sub.keys()) == {'tv'}
    non_var = c.not_subset(Variable)
    assert set(non_var.keys()) == {'n'}


def test_include_and_exclude_by_exact_class():
    v = Variable(bm.zeros(2))
    tv = TrainVar(bm.zeros(2))
    c = Collector({'v': v, 'tv': tv})
    # include matches exact class only (TrainVar is NOT Variable exactly)
    inc = c.include(Variable)
    assert set(inc.keys()) == {'v'}
    exc = c.exclude(Variable)
    assert set(exc.keys()) == {'tv'}


def test_unique_dedups_by_identity():
    shared = Variable(bm.zeros(2))
    other = Variable(bm.ones(2))
    c = Collector({'a': shared, 'b': shared, 'c': other})
    uniq = c.unique()
    # only one of the two keys pointing to `shared` survives, plus `other`
    assert len(uniq) == 2
    ids = {id(v) for v in uniq.values()}
    assert id(other) in ids
    assert id(shared) in ids


# ---------------------------------------------------------------------------
# ArrayCollector / TensorCollector
# ---------------------------------------------------------------------------

def test_tensor_collector_is_array_collector_alias():
    assert TensorCollector is ArrayCollector


def test_array_collector_setitem_requires_variable():
    ac = ArrayCollector()
    with pytest.raises(AssertionError):
        ac['x'] = 5               # not a Variable


def test_array_collector_setitem_same_value_ok():
    ac = ArrayCollector()
    v = Variable(bm.zeros(3))
    ac['v'] = v
    ac['v'] = v                   # same id, no conflict
    assert ac['v'] is v


def test_array_collector_setitem_conflict_raises():
    ac = ArrayCollector()
    ac['v'] = Variable(bm.zeros(3))
    with pytest.raises(ValueError):
        ac['v'] = Variable(bm.ones(3))   # different Variable, same key


def test_array_collector_dict_and_data():
    ac = ArrayCollector()
    v1 = Variable(bm.zeros(2))
    v2 = Variable(bm.ones(2))
    ac['v1'] = v1
    ac['v2'] = v2
    d = ac.dict()
    assert set(d.keys()) == {'v1', 'v2'}
    # dict() returns the underlying .value of each Variable
    assert bm.allclose(d['v1'], bm.zeros(2))
    data = ac.data()
    assert len(data) == 2
    assert bm.allclose(data[1], bm.ones(2))


def test_array_collector_pytree_roundtrip():
    ac = ArrayCollector()
    ac['v1'] = Variable(bm.zeros(2))
    ac['v2'] = Variable(bm.ones(2))
    leaves, treedef = jax.tree_util.tree_flatten(ac)
    assert len(leaves) == 2
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, ArrayCollector)
    assert set(rebuilt.keys()) == {'v1', 'v2'}
