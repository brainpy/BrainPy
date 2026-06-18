# -*- coding: utf-8 -*-
"""Coverage tests for :mod:`brainpy.mixin`.

Exercises the mixin classes and helpers in ``brainpy/mixin.py``:

- ``_get_bm`` lazy loader
- ``AlignPost.align_post_input_add`` (NotImplementedError stub)
- ``BindCondData.bind_cond`` / ``unbind_cond``
- ``ReturnInfo.get_data`` for callable + array data, batch modes, and error
- ``Container`` (``__getitem__``, ``__getattr__``, ``__repr__``,
  ``format_elements`` valid/invalid branches, ``add_elem``)
- ``TreeNode`` (``check_hierarchy`` / ``check_hierarchies`` valid + errors)
- ``SupportInputProj`` (``add_inp_fun``, ``get_inp_fun``,
  ``sum_current_inputs`` / ``sum_delta_inputs`` with/without label,
  ``cur_inputs`` property, deprecated ``sum_inputs``, error branches)
- the abstract stub mixins (``SupportReturnInfo``, ``SupportOnline``,
  ``SupportOffline``, ``SupportSTDP``).
"""
import warnings

import jax.numpy as jnp
import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy import mixin


# --------------------------------------------------------------------------- #
# _get_bm
# --------------------------------------------------------------------------- #
def test_get_bm_lazy_loader():
    m = mixin._get_bm()
    assert m is bm
    # second call hits cached branch
    assert mixin._get_bm() is m


# --------------------------------------------------------------------------- #
# AlignPost
# --------------------------------------------------------------------------- #
def test_align_post_not_implemented():
    class S(mixin.AlignPost):
        pass

    with pytest.raises(NotImplementedError):
        S().align_post_input_add(1.0)


# --------------------------------------------------------------------------- #
# BindCondData
# --------------------------------------------------------------------------- #
def test_bind_unbind_cond():
    class C(mixin.BindCondData):
        def __init__(self):
            self._conductance = None

    c = C()
    c.bind_cond(0.5)
    assert c._conductance == 0.5
    c.unbind_cond()
    assert c._conductance is None


# --------------------------------------------------------------------------- #
# ReturnInfo.get_data
# --------------------------------------------------------------------------- #
class TestReturnInfo:
    def test_callable_int_batch(self):
        ri = mixin.ReturnInfo(size=(3,), batch_or_mode=4, data=jnp.zeros)
        out = ri.get_data()
        assert out.shape == (4, 3)

    def test_callable_nonbatching_mode(self):
        ri = mixin.ReturnInfo(size=(3,), batch_or_mode=bm.NonBatchingMode(), data=jnp.zeros)
        out = ri.get_data()
        assert out.shape == (3,)

    def test_callable_batching_mode(self):
        ri = mixin.ReturnInfo(size=(3,), batch_or_mode=bm.BatchingMode(8), data=jnp.zeros)
        out = ri.get_data()
        assert out.shape == (8, 3)

    def test_callable_none_mode(self):
        ri = mixin.ReturnInfo(size=(5,), batch_or_mode=None, data=jnp.zeros)
        out = ri.get_data()
        assert out.shape == (5,)

    def test_array_data(self):
        arr = bm.ones(3)
        ri = mixin.ReturnInfo(size=(3,), data=arr)
        assert ri.get_data() is arr

    def test_bad_data(self):
        ri = mixin.ReturnInfo(size=(3,), data='not-array-or-callable')
        with pytest.raises(ValueError):
            ri.get_data()


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #
class TestContainer:
    def _make(self, **children):
        c = mixin.Container()
        c.children = dict(children)
        return c

    def test_getitem_ok(self):
        a = bp.dyn.Expon(1)
        c = self._make(a=a)
        assert c['a'] is a

    def test_getitem_missing(self):
        c = self._make()
        with pytest.raises(ValueError):
            c['missing']

    def test_getattr_child(self):
        a = bp.dyn.Expon(1)
        c = self._make(a=a)
        assert c.a is a

    def test_getattr_children_attr(self):
        c = self._make(a=bp.dyn.Expon(1))
        # access the raw 'children' attribute path
        assert 'a' in c.children

    def test_getattr_missing_falls_through(self):
        c = self._make()
        with pytest.raises(AttributeError):
            _ = c.nonexistent_attr

    def test_repr(self):
        c = self._make(a=bp.dyn.Expon(1))
        s = repr(c)
        assert 'Container' in s

    def test_format_elements_tuple_obj(self):
        c = mixin.Container()
        a = bp.dyn.Expon(1)
        res = c.format_elements(bp.DynamicalSystem, a)
        assert a in res.values()

    def test_format_elements_list(self):
        c = mixin.Container()
        a = bp.dyn.Expon(1)
        b = bp.dyn.Expon(1)
        res = c.format_elements(bp.DynamicalSystem, [a, b])
        assert len(res) == 2

    def test_format_elements_list_bad(self):
        c = mixin.Container()
        with pytest.raises(ValueError):
            c.format_elements(bp.DynamicalSystem, [123])

    def test_format_elements_dict_in_tuple(self):
        c = mixin.Container()
        a = bp.dyn.Expon(1)
        res = c.format_elements(bp.DynamicalSystem, {'x': a})
        assert res['x'] is a

    def test_format_elements_dict_in_tuple_bad(self):
        c = mixin.Container()
        with pytest.raises(ValueError):
            c.format_elements(bp.DynamicalSystem, {'x': 123})

    def test_format_elements_bad_tuple_elem(self):
        c = mixin.Container()
        with pytest.raises(ValueError):
            c.format_elements(bp.DynamicalSystem, 123)

    def test_format_elements_kwargs(self):
        c = mixin.Container()
        a = bp.dyn.Expon(1)
        res = c.format_elements(bp.DynamicalSystem, x=a)
        assert res['x'] is a

    def test_format_elements_kwargs_bad(self):
        c = mixin.Container()
        with pytest.raises(ValueError):
            c.format_elements(bp.DynamicalSystem, x=123)

    def test_add_elem(self):
        # DynSysGroup is a real Container subclass with a children dict
        net = bp.DynSysGroup()
        extra = bp.dyn.Expon(1)
        net.add_elem(extra=extra)
        assert extra in net.children.values()

    def test_format_elements_non_brainpyobject_name(self):
        # exercises __get_elem_name fallback (get_unique_name) for a tuple-typed
        # element that is NOT a BrainPyObject.
        c = mixin.Container()
        plain1 = object()
        plain2 = object()
        res = c.format_elements(object, plain1, plain2)
        assert len(res) == 2
        assert plain1 in res.values() and plain2 in res.values()
        # names auto-generated via get_unique_name('ContainerElem')
        assert all(k.startswith('ContainerElem') for k in res)


# --------------------------------------------------------------------------- #
# TreeNode
# --------------------------------------------------------------------------- #
class TestTreeNode:
    def test_check_hierarchy_ok(self):
        # leaf with master_type matching root
        class Root(bp.DynamicalSystem):
            def update(self, *a, **k):
                pass

        class Leaf(bp.DynamicalSystem, mixin.TreeNode):
            master_type = Root

            def update(self, *a, **k):
                pass

        node = Leaf()
        node.check_hierarchy(Root, node)  # issubclass(Root, Root) -> ok

    def test_check_hierarchy_missing_master_type(self):
        class Root(bp.DynamicalSystem):
            def update(self, *a, **k):
                pass

        tn = _bare_tree_node()
        leaf = object()  # no master_type
        with pytest.raises(ValueError):
            tn.check_hierarchy(Root, leaf)

    def test_check_hierarchy_type_mismatch(self):
        class RootA(bp.DynamicalSystem):
            def update(self, *a, **k):
                pass

        class RootB(bp.DynamicalSystem):
            def update(self, *a, **k):
                pass

        class Leaf(bp.DynamicalSystem, mixin.TreeNode):
            master_type = RootA

            def update(self, *a, **k):
                pass

        leaf = Leaf()
        with pytest.raises(TypeError):
            leaf.check_hierarchy(RootB, leaf)  # issubclass(RootB, RootA) -> False

    def test_check_hierarchies_seq_and_dict(self):
        class Root(bp.DynamicalSystem):
            def update(self, *a, **k):
                pass

        class Leaf(bp.DynamicalSystem, mixin.TreeNode):
            master_type = Root

            def update(self, *a, **k):
                pass

        l1, l2, l3 = Leaf(), Leaf(), Leaf()
        tn = _bare_tree_node()
        # positional DynamicalSystem leaf, a list, a dict, and named leaves
        tn.check_hierarchies(Root, l1, [l2], named=l3)

    def test_check_hierarchies_dict_leaf(self):
        # a dict positional leaf recurses via **leaf (named_leaves path)
        class Root(bp.DynamicalSystem):
            def update(self, *a, **k):
                pass

        class Leaf(bp.DynamicalSystem, mixin.TreeNode):
            master_type = Root

            def update(self, *a, **k):
                pass

        tn = _bare_tree_node()
        tn.check_hierarchies(Root, {'x': Leaf()})

    def test_check_hierarchies_bad_leaf_type(self):
        class Root(bp.DynamicalSystem):
            def update(self, *a, **k):
                pass

        tn = _bare_tree_node()
        with pytest.raises(ValueError):
            tn.check_hierarchies(Root, 123)

    def test_check_hierarchies_bad_named_leaf(self):
        class Root(bp.DynamicalSystem):
            def update(self, *a, **k):
                pass

        tn = _bare_tree_node()
        with pytest.raises(ValueError):
            tn.check_hierarchies(Root, named=123)


def _bare_tree_node():
    class TN(mixin.TreeNode):
        pass

    return TN()


# --------------------------------------------------------------------------- #
# SupportInputProj
# --------------------------------------------------------------------------- #
class _Proj(mixin.SupportInputProj):
    def __init__(self):
        self.current_inputs = {}
        self.delta_inputs = {}


class TestSupportInputProj:
    def test_add_inp_fun_not_callable(self):
        p = _Proj()
        with pytest.raises(TypeError):
            p.add_inp_fun('k', 123)

    def test_add_current_and_duplicate(self):
        p = _Proj()
        p.add_inp_fun('a', lambda: 1.0, category='current')
        assert 'a' in p.current_inputs
        with pytest.raises(ValueError):
            p.add_inp_fun('a', lambda: 1.0, category='current')

    def test_add_delta_and_duplicate(self):
        p = _Proj()
        p.add_inp_fun('b', lambda: 2.0, category='delta')
        assert 'b' in p.delta_inputs
        with pytest.raises(ValueError):
            p.add_inp_fun('b', lambda: 2.0, category='delta')

    def test_add_bad_category(self):
        p = _Proj()
        with pytest.raises(NotImplementedError):
            p.add_inp_fun('c', lambda: 1.0, category='unknown')

    def test_add_with_label(self):
        p = _Proj()
        p.add_inp_fun('x', lambda: 1.0, label='L', category='current')
        # label-prefixed key
        assert any(k.startswith('L // ') for k in p.current_inputs)

    def test_get_inp_fun_current(self):
        p = _Proj()
        f = lambda: 1.0
        p.add_inp_fun('a', f, category='current')
        assert p.get_inp_fun('a') is f

    def test_get_inp_fun_delta(self):
        p = _Proj()
        f = lambda: 1.0
        p.add_inp_fun('b', f, category='delta')
        assert p.get_inp_fun('b') is f

    def test_get_inp_fun_unknown(self):
        p = _Proj()
        with pytest.raises(ValueError):
            p.get_inp_fun('nope')

    def test_sum_current_inputs_no_label(self):
        p = _Proj()
        p.add_inp_fun('a', lambda: 1.0, category='current')
        p.add_inp_fun('b', lambda: 2.0, category='current')
        assert p.sum_current_inputs(init=0.0) == 3.0

    def test_sum_current_inputs_with_label(self):
        p = _Proj()
        p.add_inp_fun('a', lambda: 1.0, label='L', category='current')
        p.add_inp_fun('b', lambda: 2.0, category='current')  # no label
        # only the L-labelled function counts
        assert p.sum_current_inputs(init=0.0, label='L') == 1.0

    def test_sum_delta_inputs_no_label(self):
        p = _Proj()
        p.add_inp_fun('a', lambda: 4.0, category='delta')
        assert p.sum_delta_inputs(init=0.0) == 4.0

    def test_sum_delta_inputs_with_label(self):
        p = _Proj()
        p.add_inp_fun('a', lambda: 4.0, label='D', category='delta')
        p.add_inp_fun('b', lambda: 9.0, category='delta')
        assert p.sum_delta_inputs(init=0.0, label='D') == 4.0

    def test_cur_inputs_property(self):
        p = _Proj()
        assert p.cur_inputs is p.current_inputs

    def test_sum_inputs_deprecated(self):
        p = _Proj()
        p.add_inp_fun('a', lambda: 5.0, category='current')
        with pytest.warns(UserWarning, match='sum_current_inputs'):
            assert p.sum_inputs(init=0.0) == 5.0

    def test_input_label_helpers(self):
        assert mixin.SupportInputProj._input_label_start('L') == 'L // '
        assert mixin.SupportInputProj._input_label_repr('n') == 'n'
        assert mixin.SupportInputProj._input_label_repr('n', 'L') == 'L // n'


# --------------------------------------------------------------------------- #
# stub mixins
# --------------------------------------------------------------------------- #
class TestStubMixins:
    def test_support_return_info(self):
        class R(mixin.SupportReturnInfo):
            pass

        with pytest.raises(NotImplementedError):
            R().return_info()

    def test_support_auto_delay_inherits(self):
        assert issubclass(mixin.SupportAutoDelay, mixin.SupportReturnInfo)

    def test_support_online(self):
        class O(mixin.SupportOnline):
            pass

        with pytest.raises(NotImplementedError):
            O().online_init()
        with pytest.raises(NotImplementedError):
            O().online_fit('t', {})

    def test_support_offline(self):
        class F(mixin.SupportOffline):
            pass

        # offline_init is a no-op (pass)
        assert F().offline_init() is None
        with pytest.raises(NotImplementedError):
            F().offline_fit('t', {})

    def test_support_stdp(self):
        class S(mixin.SupportSTDP):
            pass

        with pytest.raises(NotImplementedError):
            S().stdp_update()
