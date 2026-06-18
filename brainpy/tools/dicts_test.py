# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy/tools/dicts.py``.

Exercises ``DotDict``:
- dot-access read/write, KeyError on missing attribute.
- ``copy`` (returns a DotDict), ``to_numpy``, ``update`` (returns self).
- ``__add__`` merge.
- ``__sub__`` with dict (success, mismatched-value raise, missing-key raise),
  with list/tuple of string keys and of value objects (id lookup), missing
  key in sequence raise, and the bad-type raise.
- ``subset`` (type filtering), ``unique`` (dedup by id), ``__hash__``.
- pytree registration round-trip via ``jax.tree_util``.
"""

import numpy as np
import pytest

from brainpy.tools.dicts import DotDict


# --------------------------------------------------------------------------
# basic dot access
# --------------------------------------------------------------------------

def test_dotdict_dot_access():
    d = DotDict({"a": 10, "b": 20})
    assert d.a == 10
    assert d["b"] == 20


def test_dotdict_missing_attr_raises():
    # NOTE: the class docstring claims accessing a missing key via attribute
    # raises KeyError, but because ``self.__dict__ = self`` routes attribute
    # lookups through dict, a missing attribute actually raises AttributeError.
    # Documents current behavior.
    d = DotDict({"a": 1})
    with pytest.raises(AttributeError):
        _ = d.c
    # subscript access on a missing key still raises KeyError as usual
    with pytest.raises(KeyError):
        _ = d["c"]


def test_dotdict_assign_new_attr():
    d = DotDict({"a": 1})
    d.c = 30
    assert d.c == 30
    assert d["c"] == 30


# --------------------------------------------------------------------------
# copy / to_numpy / update
# --------------------------------------------------------------------------

def test_dotdict_copy():
    d = DotDict({"a": 1})
    c = d.copy()
    assert isinstance(c, DotDict)
    assert c == d
    c.a = 99
    assert d.a == 1  # independent


def test_dotdict_to_numpy():
    d = DotDict({"a": [1, 2, 3]})
    d.to_numpy()
    assert isinstance(d["a"], np.ndarray)
    assert np.array_equal(d["a"], [1, 2, 3])


def test_dotdict_update_returns_self():
    d = DotDict({"a": 1})
    ret = d.update({"b": 2})
    assert ret is d
    assert d.b == 2


# --------------------------------------------------------------------------
# __add__
# --------------------------------------------------------------------------

def test_dotdict_add():
    d1 = DotDict({"a": 1})
    d2 = {"b": 2}
    merged = d1 + d2
    assert isinstance(merged, DotDict)
    assert merged.a == 1 and merged.b == 2
    # original unchanged
    assert "b" not in d1


# --------------------------------------------------------------------------
# __sub__ with dict
# --------------------------------------------------------------------------

def test_dotdict_sub_dict_success():
    val = object()
    d = DotDict({"a": val, "b": 2})
    out = d - {"a": val}
    assert "a" not in out
    assert "b" in out


def test_dotdict_sub_dict_value_mismatch_raises():
    d = DotDict({"a": object()})
    with pytest.raises(ValueError, match="two different values"):
        d - {"a": object()}


def test_dotdict_sub_dict_missing_key_raises():
    d = DotDict({"a": 1})
    with pytest.raises(ValueError, match="do not find it"):
        d - {"z": 1}


# --------------------------------------------------------------------------
# __sub__ with list/tuple
# --------------------------------------------------------------------------

def test_dotdict_sub_list_of_string_keys():
    d = DotDict({"a": 1, "b": 2, "c": 3})
    out = d - ["a", "b"]
    assert set(out.keys()) == {"c"}


def test_dotdict_sub_list_of_value_objects():
    v1 = object()
    v2 = object()
    d = DotDict({"a": v1, "b": v2})
    out = d - (v1,)
    # remove by id of value -> drops key 'a'
    assert "a" not in out
    assert "b" in out


def test_dotdict_sub_list_missing_key_raises():
    # 'z' is a string key not present -> raises in the final loop
    d = DotDict({"a": 1})
    with pytest.raises(ValueError, match="do not find it"):
        d - ["z"]


def test_dotdict_sub_bad_type_raises():
    d = DotDict({"a": 1})
    with pytest.raises(ValueError, match="Only support dict/tuple/list"):
        d - 123


# --------------------------------------------------------------------------
# subset / unique
# --------------------------------------------------------------------------

def test_dotdict_subset():
    d = DotDict({"i": 1, "s": "x", "f": 2.0, "j": 5})
    out = d.subset(int)
    # only ints (note: bool would also match but none present)
    assert set(out.keys()) == {"i", "j"}
    assert isinstance(out, DotDict)


def test_dotdict_unique():
    shared = object()
    d = DotDict({"a": shared, "b": shared, "c": object()})
    out = d.unique()
    # only one of the keys mapping to `shared` survives
    shared_keys = [k for k, v in out.items() if v is shared]
    assert len(shared_keys) == 1
    assert "c" in out


# --------------------------------------------------------------------------
# __hash__
# --------------------------------------------------------------------------

def test_dotdict_hash():
    d = DotDict({"a": 1, "b": 2})
    # hashable; consistent for equal contents
    assert hash(d) == hash(DotDict({"b": 2, "a": 1}))


# --------------------------------------------------------------------------
# pytree registration
# --------------------------------------------------------------------------

def test_dotdict_pytree_roundtrip():
    import jax
    d = DotDict({"a": np.array([1.0, 2.0]), "b": np.array([3.0])})
    leaves, treedef = jax.tree_util.tree_flatten(d)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, DotDict)
    assert set(rebuilt.keys()) == {"a", "b"}
    assert np.array_equal(rebuilt["a"], [1.0, 2.0])


def test_dotdict_pytree_map():
    import jax
    d = DotDict({"a": np.array([1.0]), "b": np.array([2.0])})
    out = jax.tree_util.tree_map(lambda x: x + 1, d)
    assert isinstance(out, DotDict)
    assert np.array_equal(out["a"], [2.0])
