# -*- coding: utf-8 -*-
"""Coverage tests for :mod:`brainpy.dynsys`.

Exercises the base dynamical-system machinery in ``brainpy/dynsys.py``:

- ``DelayRegister`` (``register_delay`` / ``get_delay_data`` /
  ``get_delay_var`` and the deprecated ``update_local_delays`` /
  ``reset_local_delays`` warnings).
- ``DynamicalSystem`` construction (mode validation, ``supported_modes``
  restriction), before/after update registration (add/get/has + dup &
  missing errors), ``__call__`` dispatch through before/after updates,
  ``step_run`` / ``jit_step_run``, ``mode`` getter/setter, ``reset`` /
  ``reset_state`` (incl. the compat deprecation path), the
  ``_compatible_update`` legacy ``tdi`` shims, ``register_local_delay`` /
  ``get_local_delay``, ``__repr__`` / ``__rrshift__``.
- ``DynSysGroup`` / ``Network`` update; ``Sequential`` indexing/slicing,
  ``update``, ``return_info`` error, ``__repr__``.
- ``Projection`` update (empty -> error), ``clear_input`` / ``reset_state``.
- ``Dynamic`` size validation, ``varshape`` / ``get_batch_shape``,
  ``init_param`` / ``init_variable``, ``__getitem__`` -> ``DynView``.
- ``DynView`` slicing (int/slice/iterable), ``SLICE_VARS`` path,
  batch-axis path, ``__setattr__`` / ``__getattribute__``, ``update`` error.
- ``_slice_to_num`` edge cases.
- the ``receive_/not_receive_`` update-input/output decorators.

NOTE: ``_slice_to_num`` (dynsys.py:949-981) is decorated with
``@tools.numba_jit``. When numba is installed it is compiled to a numba
``CPUDispatcher``, so its body executes in numba's runtime rather than under
CPython's line tracer -- coverage.py therefore reports lines 952-981 as
"missing" even though the function is exercised here. This is a coverage-tool
limitation, not a test gap.
"""
import warnings

import numpy as np
import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy import dynsys
from brainpy._errors import UnsupportedError, NoImplementationError
from brainpy.dynsys import (
    _slice_to_num,
    receive_update_output,
    not_receive_update_output,
    receive_update_input,
    not_receive_update_input,
    not_implemented,
)


# --------------------------------------------------------------------------- #
# helper systems
# --------------------------------------------------------------------------- #
class _Tiny(bp.DynamicalSystem):
    def __init__(self, mode=None):
        super().__init__(mode=mode)
        self.v = bm.Variable(bm.zeros(3))

    def update(self, x=None):
        self.v.value = self.v.value + (0. if x is None else x)
        return self.v.value


class _Pop(dynsys.Dynamic):
    def __init__(self, size, **kwargs):
        super().__init__(size, **kwargs)
        self.V = bm.Variable(bm.zeros(self.num))

    def update(self, x=None):
        return self.V.value


@pytest.fixture(autouse=True)
def _share_ctx():
    bp.share.save(i=0, t=0., dt=0.1)
    yield


# --------------------------------------------------------------------------- #
# DelayRegister
# --------------------------------------------------------------------------- #
class TestDelayRegister:
    def test_register_and_get_delay(self):
        lif = bp.dyn.Lif(5)
        name = lif.register_delay('out', 2, lif.spike)
        assert isinstance(name, str)
        data = lif.get_delay_data('out', name)
        assert bm.as_jax(data).shape == (5,)
        assert lif.get_delay_var('out') is not None

    def test_deprecated_update_reset_local_delays(self):
        lif = bp.dyn.Lif(5)
        with pytest.warns(DeprecationWarning):
            lif.update_local_delays()
        with pytest.warns(DeprecationWarning):
            lif.reset_local_delays()


# --------------------------------------------------------------------------- #
# DynamicalSystem construction & mode
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_bad_mode_type(self):
        with pytest.raises(ValueError):
            _Tiny(mode='not-a-mode')

    def test_supported_modes_ok(self):
        class Restricted(bp.DynamicalSystem):
            supported_modes = (bm.NonBatchingMode,)

            def update(self, x=None):
                return x

        # NonBatchingMode is parent-compatible -> ok
        Restricted(mode=bm.NonBatchingMode())

    def test_supported_modes_violation(self):
        class Restricted(bp.DynamicalSystem):
            supported_modes = (bm.BatchingMode,)

            def update(self, x=None):
                return x

        with pytest.raises(UnsupportedError):
            Restricted(mode=bm.NonBatchingMode())

    def test_mode_getter_setter(self):
        t = _Tiny()
        assert isinstance(t.mode, bm.Mode)
        t.mode = bm.NonBatchingMode()
        with pytest.raises(ValueError):
            t.mode = 'bad'

    def test_repr_and_rrshift(self):
        t = _Tiny()
        assert 'mode' in repr(t)
        out = 1.0 >> t  # __rrshift__
        assert bm.as_jax(out).shape == (3,)


# --------------------------------------------------------------------------- #
# before/after updates
# --------------------------------------------------------------------------- #
class TestBeforeAfterUpdates:
    def test_add_get_has(self):
        t = _Tiny()
        bef = lambda: None
        aft = lambda ret=None: None
        t.add_bef_update('b', bef)
        t.add_aft_update('a', aft)
        assert t.has_bef_update('b')
        assert t.has_aft_update('a')
        assert t.get_bef_update('b') is bef
        assert t.get_aft_update('a') is aft

    def test_duplicate_keys(self):
        t = _Tiny()
        t.add_bef_update('b', lambda: None)
        t.add_aft_update('a', lambda ret=None: None)
        with pytest.raises(KeyError):
            t.add_bef_update('b', lambda: None)
        with pytest.raises(KeyError):
            t.add_aft_update('a', lambda ret=None: None)

    def test_missing_keys(self):
        t = _Tiny()
        with pytest.raises(KeyError):
            t.get_bef_update('zzz')
        with pytest.raises(KeyError):
            t.get_aft_update('zzz')

    def test_call_dispatches_updates(self):
        t = _Tiny()
        seen = {}
        t.add_bef_update('b', lambda *a, **k: seen.__setitem__('bef', True))
        t.add_aft_update('a', lambda ret=None: seen.__setitem__('aft', ret))
        t(1.0)
        assert seen['bef'] is True
        assert bm.as_jax(seen['aft']).shape == (3,)

    def test_call_with_receive_input_and_not_receive_output(self):
        t = _Tiny()

        # before-update that *receives* the update input
        @receive_update_input
        class BefRecv:
            def __init__(self):
                self.got = None

            def __call__(self, *args, **kwargs):
                self.got = args

        # after-update that does *not* receive the update output
        @not_receive_update_output
        class AftNoRecv:
            def __init__(self):
                self.called = False

            def __call__(self):
                self.called = True

        bef = BefRecv()
        aft = AftNoRecv()
        t.add_bef_update('b', bef)
        t.add_aft_update('a', aft)
        t(2.0)
        assert bef.got == (2.0,)
        assert aft.called is True


# --------------------------------------------------------------------------- #
# step_run / jit_step_run / reset
# --------------------------------------------------------------------------- #
class TestRunReset:
    def test_step_run(self):
        t = _Tiny()
        out = t.step_run(0, 1.0)
        assert bm.allclose(bm.as_jax(out), bm.ones(3))

    def test_jit_step_run(self):
        t = _Tiny()
        out = t.jit_step_run(1, 1.0)
        assert bm.as_jax(out).shape == (3,)

    def test_reset(self):
        # _Tiny has no custom reset_state (default is a no-op), so reset()
        # traverses children and runs without error; state is left unchanged.
        t = _Tiny()
        t.update(5.0)
        t.reset()  # exercises DynamicalSystem.reset -> helpers.reset_state
        assert bm.as_jax(t.v.value).shape == (3,)

    def test_reset_state_compat_deprecation(self):
        # a node without a custom reset_state hits the compat/deprecation path
        class NoResetState(bp.DynamicalSystem):
            def __init__(self):
                super().__init__()
                self.v = bm.Variable(bm.zeros(3))

            def update(self, x=None):
                return self.v.value

        n = NoResetState()
        with pytest.warns(DeprecationWarning):
            n.reset_state()

    def test_clear_input_noop(self):
        t = _Tiny()
        assert t.clear_input() is None


# --------------------------------------------------------------------------- #
# _compatible_update legacy paths
# --------------------------------------------------------------------------- #
class TestCompatibleUpdate:
    def test_tdi_signature_called_with_dict(self):
        class WithTdi(bp.DynamicalSystem):
            def update(self, tdi, x=None):
                return x

        w = WithTdi()
        with pytest.warns(UserWarning):
            assert w(dict(t=0., i=0), 5.0) == 5.0

    def test_tdi_signature_called_without_dict(self):
        class WithTdi(bp.DynamicalSystem):
            def update(self, sh, x=None):
                return x

        w = WithTdi()
        with pytest.warns(UserWarning):
            # called positionally without a leading dict -> share.get_shargs() shim
            assert w(7.0) == 7.0

    def test_no_tdi_signature_called_with_dict(self):
        class NoTdi(bp.DynamicalSystem):
            def update(self, x=None):
                return x

        n = NoTdi()
        with pytest.warns(UserWarning):
            assert n(dict(t=0.), 9.0) == 9.0

    def test_plain_update(self):
        t = _Tiny()
        # the modern path: update(x) called as update(x)
        out = t(1.0)
        assert bm.as_jax(out).shape == (3,)

    def test_tdi_in_kwargs(self):
        # update(tdi, x) defined; called with the tdi-named param as a kwarg
        # and no positional args -> branch at dynsys.py:417-423.
        class A(bp.DynamicalSystem):
            def update(self, tdi, x=None):
                return x

        a = A()
        with pytest.warns(UserWarning):
            assert a(tdi=dict(t=0.), x=5) == 5

    def test_tdi_signature_no_positional_no_tdi_kwarg(self):
        # update(tdi, x) defined; called with no positional and tdi NOT in
        # kwargs -> share.get_shargs() shim, branch at dynsys.py:424-430.
        class B(bp.DynamicalSystem):
            def update(self, tdi, x=None):
                return x

        b = B()
        with pytest.warns(UserWarning):
            assert b(x=9) == 9

    def test_bind_typeerror_args0_dict(self):
        # update(x) requires one positional; calling with (dict, extra) makes
        # bind() raise TypeError and args[0] is a dict -> share.save shim path
        # (dynsys.py:433-447).
        class A(bp.DynamicalSystem):
            def update(self, x):
                return x

        a = A()
        with pytest.warns(UserWarning):
            assert a(dict(t=0.), 5) == 5

    def test_bind_typeerror_args0_not_dict(self):
        # update(shared) requires one positional named 'shared' (not tdi/sh/sha);
        # called with no args -> bind() fails, args empty -> get_shargs() shim
        # injects the shared dict (dynsys.py:448-458).
        class B(bp.DynamicalSystem):
            def update(self, shared):
                return type(shared).__name__

        b = B()
        with pytest.warns(UserWarning):
            assert b() == 'DotDict'

    def test_scalar_dict_inner_bind_failure(self):
        # update(a, b): outer bind(dict, 5) succeeds (a=dict, b=5) and args[0] is
        # a scalar dict, so the inner bind(args[1:]=(5,)) is attempted and FAILS
        # (b missing) -> ``except TypeError: pass`` -> falls through to the plain
        # call update(dict, 5) (dynsys.py:460-464, 475).
        class M(bp.DynamicalSystem):
            def update(self, a, b):
                return (a, b)

        m = M()
        a, b = m(dict(t=0.), 5)
        assert a == {'t': 0.} and b == 5

    def test_update_x_default_called_with_tdi(self):
        # update(x=None) defined; called with a scalar dict that is treated as
        # shared args, then update() runs with x defaulting to None
        # (dynsys.py:465-474).
        class A(bp.DynamicalSystem):
            def update(self, x=None):
                return x

        a = A()
        with pytest.warns(UserWarning):
            assert a(dict(t=0.5)) is None

    def test_base_update_not_implemented(self):
        # the un-overridden DynamicalSystem.update raises (dynsys.py:271).
        class Bare(bp.DynamicalSystem):
            pass

        with pytest.raises(NotImplementedError):
            Bare().update()

    def test_get_update_fun(self):
        # exercises _get_update_fun (dynsys.py:502).
        t = _Tiny()
        fn = t._get_update_fun()
        assert callable(fn)


# --------------------------------------------------------------------------- #
# register_local_delay / get_local_delay
# --------------------------------------------------------------------------- #
class TestLocalDelay:
    def test_register_get_local_delay(self):
        lif = bp.dyn.Lif(10)
        lif.register_local_delay('spike', 'd1', delay_time=10.)
        data = lif.get_local_delay('spike', 'd1')
        assert bm.allclose(data, bm.zeros(10))

    def test_register_local_delay_bad_var(self):
        lif = bp.dyn.Lif(10)
        with pytest.raises(AttributeError):
            lif.register_local_delay('nonexistent_var', 'd', 10.)


# --------------------------------------------------------------------------- #
# DynSysGroup / Network
# --------------------------------------------------------------------------- #
class TestGroups:
    def test_dynsysgroup_update(self):
        net = bp.DynSysGroup(n1=bp.dyn.Lif(5), n2=bp.dyn.Lif(5))
        net.update()

    def test_network_update(self):
        net = bp.Network(bp.dyn.Lif(3))
        net.update()

    def test_group_with_projection_node(self):
        # a group containing a Projection node and a Dynamic node exercises the
        # projection / dynamics / other-type iteration branches (583, 591).
        class MyProj(bp.Projection):
            def __init__(self):
                super().__init__()
                self.child = bp.dyn.Lif(5)

        net = bp.DynSysGroup(proj=MyProj(), neu=bp.dyn.Lif(5))
        net.update()

    def test_group_with_other_type_node(self):
        # a DynamicalSystem node that is neither Dynamic nor Projection hits the
        # "other types" iteration branch (dynsys.py:591).
        class Other(bp.DynamicalSystem):
            def __init__(self):
                super().__init__()
                self.v = bm.Variable(bm.zeros(3))

            def update(self, x=None):
                return self.v.value

        net = bp.DynSysGroup(other=Other())
        net.update()


# --------------------------------------------------------------------------- #
# Sequential
# --------------------------------------------------------------------------- #
class TestSequential:
    def _make(self):
        return bp.Sequential(
            l1=bp.layers.Dense(4, 3),
            l2=bm.relu,
            l3=bp.layers.Dense(3, 2),
        )

    def test_forward(self):
        seq = self._make()
        out = seq(bm.random.rand(2, 4))
        assert out.shape == (2, 2)

    def test_index_str(self):
        seq = self._make()
        assert type(seq['l1']).__name__ == 'Dense'

    def test_index_str_missing(self):
        seq = self._make()
        with pytest.raises(KeyError):
            seq['nope']

    def test_index_int(self):
        seq = self._make()
        assert type(seq[0]).__name__ == 'Dense'

    def test_index_slice(self):
        seq = self._make()
        sub = seq[0:2]
        assert isinstance(sub, bp.Sequential)

    def test_index_tuple(self):
        seq = self._make()
        sub = seq[(0, 2)]
        assert isinstance(sub, bp.Sequential)

    def test_index_bad_type(self):
        seq = self._make()
        with pytest.raises(KeyError):
            seq[1.5]

    def test_repr(self):
        seq = self._make()
        assert 'Sequential' in repr(seq)

    def test_return_info_unsupported(self):
        seq = self._make()
        with pytest.raises(UnsupportedError):
            seq.return_info()

    def test_return_info_success(self):
        # last node is an Expon, which implements return_info (SupportAutoDelay).
        seq = bp.Sequential(l1=bp.dyn.Expon(5))
        info = seq.return_info()
        assert info is not None


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
class TestProjection:
    def test_empty_update_raises(self):
        class EmptyProj(bp.Projection):
            pass

        ep = EmptyProj()
        with pytest.raises(ValueError):
            ep.update()

    def test_clear_input_and_reset(self):
        class EmptyProj(bp.Projection):
            pass

        ep = EmptyProj()
        assert ep.clear_input() is None
        assert ep.reset_state() is None

    def test_update_with_children(self):
        # a Projection with a child DynamicalSystem iterates and updates it
        # (dynsys.py:705-709).
        class ProjWithChild(bp.Projection):
            def __init__(self):
                super().__init__()
                self.child = bp.dyn.Lif(5)

        p = ProjWithChild()
        p.update()


# --------------------------------------------------------------------------- #
# Dynamic
# --------------------------------------------------------------------------- #
class TestDynamic:
    def test_size_empty_list(self):
        with pytest.raises(ValueError):
            _Pop([])

    def test_size_non_int_elem(self):
        with pytest.raises(ValueError):
            _Pop(['a'])

    def test_size_float(self):
        with pytest.raises(ValueError):
            _Pop(3.5)

    def test_size_int(self):
        p = _Pop(10)
        assert p.size == (10,)
        assert p.num == 10
        assert p.varshape == (10,)

    def test_keep_size(self):
        p = _Pop((2, 3), keep_size=True)
        assert p.varshape == (2, 3)

    def test_get_batch_shape(self):
        p = _Pop(5)
        assert p.get_batch_shape() == (5,)
        assert p.get_batch_shape(4) == (4, 5)

    def test_init_param(self):
        p = _Pop(5)
        par = p.init_param(1.0)
        assert par == 1.0

    def test_init_variable(self):
        p = _Pop(5)
        var = p.init_variable(bm.zeros, None)
        assert var.shape == (5,)

    def test_update_not_implemented(self):
        class Bare(dynsys.Dynamic):
            pass

        b = Bare(3)
        with pytest.raises(NotImplementedError):
            b.update()

    def test_repr(self):
        p = _Pop(5)
        assert 'size' in repr(p)

    def test_clear_input(self):
        p = _Pop(5)
        assert p.clear_input() is None


# --------------------------------------------------------------------------- #
# DynView
# --------------------------------------------------------------------------- #
class TestDynView:
    def test_slice_view(self):
        p = _Pop(10)
        view = p[2:5]
        assert isinstance(view, dynsys.DynView)
        assert view.size == (3,)

    def test_int_view(self):
        p = _Pop(10)
        view = p[3]
        assert view.size == (1,)

    def test_iterable_index(self):
        # the iterable must be one axis of the index tuple; an int/slice is
        # auto-wrapped but a list is not, so wrap it explicitly.
        p = _Pop(10)
        view = p[[0, 2, 4],]
        assert view.size == (3,)

    def test_too_long_index(self):
        p = _Pop(10)
        with pytest.raises(ValueError):
            p[1:2, 1:2]

    def test_bad_target(self):
        with pytest.raises(TypeError):
            dynsys.DynView(target=object(), index=slice(0, 1))

    def test_non_iterable_index_elem(self):
        p = _Pop(10)
        with pytest.raises(TypeError):
            dynsys.DynView(p, (object(),))

    def test_update_raises(self):
        p = _Pop(10)
        view = p[0:5]
        with pytest.raises(NoImplementationError):
            view.update()

    def test_reset_state_noop(self):
        p = _Pop(10)
        view = p[0:5]
        assert view.reset_state() is None

    def test_repr(self):
        p = _Pop(10)
        view = p[0:5]
        assert 'target' in repr(view)

    def test_setattr_slice_var(self):
        p = _Pop(10)
        view = p[0:5]
        view.V = bm.ones(5)
        assert bm.allclose(bm.as_jax(view.V), bm.ones(5))

    def test_slice_vars_attribute_path(self):
        class PopS(dynsys.Dynamic):
            slice_vars = ['V']

            def __init__(self, size):
                super().__init__(size)
                self.V = bm.Variable(bm.zeros(self.num))

            def update(self, x=None):
                return self.V.value

        ps = PopS(10)
        view = ps[0:5]
        assert view.size == (5,)

    def test_batch_axis_view(self):
        class PopB(dynsys.Dynamic):
            def __init__(self, size, mode):
                super().__init__(size, mode=mode)
                self.V = bm.Variable(bm.zeros((1, self.num)), batch_axis=0)

            def update(self, x=None):
                return self.V.value

        pb = PopB(10, mode=bm.BatchingMode(1))
        view = pb[0:5]
        assert view.size == (5,)


# --------------------------------------------------------------------------- #
# _slice_to_num
# --------------------------------------------------------------------------- #
class TestSliceToNum:
    def test_basic(self):
        assert _slice_to_num(slice(0, 5, None), 10) == 5

    def test_none_start_stop(self):
        assert _slice_to_num(slice(None, None, None), 10) == 10

    def test_negative_start_stop(self):
        assert _slice_to_num(slice(-3, -1, None), 10) == 2

    def test_step_gt_one(self):
        assert _slice_to_num(slice(0, 10, 2), 10) == 5

    def test_negative_step(self):
        assert _slice_to_num(slice(9, 0, -2), 10) == 5

    def test_zero_step(self):
        with pytest.raises(ValueError):
            _slice_to_num(slice(0, 5, 0), 10)


# --------------------------------------------------------------------------- #
# decorators
# --------------------------------------------------------------------------- #
class TestDecorators:
    def test_not_receive_then_receive_output(self):
        class Foo:
            pass

        not_receive_update_output(Foo)
        assert hasattr(Foo, '_not_receive_update_output')
        receive_update_output(Foo)
        assert not hasattr(Foo, '_not_receive_update_output')

    def test_receive_then_not_receive_input(self):
        class Foo:
            pass

        receive_update_input(Foo)
        assert hasattr(Foo, '_receive_update_input')
        not_receive_update_input(Foo)
        assert not hasattr(Foo, '_receive_update_input')

    def test_receive_output_idempotent(self):
        # receive_update_output on a class without the marker is a no-op
        class Foo:
            pass

        assert receive_update_output(Foo) is Foo

    def test_not_implemented_decorator(self):
        @not_implemented
        def fn():
            return 1

        assert fn._not_implemented is True
        assert fn() == 1
