# -*- coding: utf-8 -*-
"""Coverage tests for :mod:`brainpy.check`.

Exhaustively exercises the validation utilities in ``brainpy/check.py``,
covering both the valid (return) paths and every ``raise`` /
``ValueError`` / ``TypeError`` / ``NotImplementedError`` branch.

Targets: ``is_checking``/``turn_on``/``turn_off``, shape helpers
(``is_shape_consistency``, ``is_shape_broadcastable``,
``check_shape_except_batch``, ``check_shape``), the type validators
(``is_dict_data``, ``is_callable``, ``is_initializer``, ``is_connector``,
``is_sequence``, ``is_float``, ``is_integer``, ``is_string``,
``is_subclass``, ``is_instance``, ``is_elem_or_seq_or_dict``,
``is_all_vars``, ``is_all_objs``), ``serialize_kwargs``, and the JIT error
helpers (``jit_error``, ``jit_error_checking``, ``jit_error_checking_no_args``).
"""
import jax
import numpy as np
import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy import check as checking


# --------------------------------------------------------------------------- #
# on/off toggles
# --------------------------------------------------------------------------- #
class TestToggles:
    def test_turn_on_off(self):
        original = checking.is_checking()
        try:
            checking.turn_off()
            assert checking.is_checking() is False
            checking.turn_on()
            assert checking.is_checking() is True
        finally:
            if original:
                checking.turn_on()
            else:
                checking.turn_off()


# --------------------------------------------------------------------------- #
# is_shape_consistency / is_shape_broadcastable
# --------------------------------------------------------------------------- #
class TestShapeConsistency:
    def test_consistent_no_free_axes(self):
        # all shapes identical -> ok, returns None when not asking for format
        out = checking.is_shape_consistency([(2, 3), (2, 3)])
        assert out is None

    def test_inconsistent_dims(self):
        with pytest.raises(ValueError):
            checking.is_shape_consistency([(2, 3), (2, 3, 4)])

    def test_inconsistent_shapes(self):
        with pytest.raises(ValueError):
            checking.is_shape_consistency([(2, 3), (2, 4)])

    def test_free_axes_int_format(self):
        unique, free = checking.is_shape_consistency(
            [(2, 3), (2, 5)], free_axes=1, return_format_shapes=True
        )
        assert unique == (2,)
        assert free == (3, 5)

    def test_free_axes_seq_format(self):
        unique, free = checking.is_shape_consistency(
            [(2, 3, 4), (2, 7, 4)], free_axes=[1], return_format_shapes=True
        )
        assert unique == (2, 4)
        assert free == ((3,), (7,))

    def test_free_axes_none_format(self):
        unique, free = checking.is_shape_consistency(
            [(2, 3), (2, 3)], free_axes=None, return_format_shapes=True
        )
        assert unique == (2, 3)
        assert free is None

    def test_negative_free_axis(self):
        # negative axis resolves to dims[0] + axis
        out = checking.is_shape_consistency([(2, 3), (2, 5)], free_axes=-1)
        assert out is None

    def test_inconsistent_with_free_axes(self):
        with pytest.raises(ValueError):
            checking.is_shape_consistency([(2, 3), (4, 3)], free_axes=[1])

    def test_bad_free_axes_type(self):
        with pytest.raises(ValueError):
            checking.is_shape_consistency([(2, 3), (2, 3)], free_axes='bad')

    def test_broadcastable_ok(self):
        # (3,) padded to (1, 3); with free axis 0 the remaining (3,) match.
        out = checking.is_shape_broadcastable([(3,), (2, 3)], free_axes=[0])
        assert out is None

    def test_broadcastable_inconsistent(self):
        # without a free axis, padded (1,3) vs (2,3) are inconsistent -> raise
        with pytest.raises(ValueError):
            checking.is_shape_broadcastable([(3,), (2, 3)], free_axes=())


# --------------------------------------------------------------------------- #
# check_shape_except_batch
# --------------------------------------------------------------------------- #
class TestCheckShapeExceptBatch:
    def test_ok(self):
        assert checking.check_shape_except_batch((10, 2, 3), (5, 2, 3)) is True

    def test_dim_mismatch_raise(self):
        with pytest.raises(ValueError):
            checking.check_shape_except_batch((10, 2), (5, 2, 3))

    def test_dim_mismatch_bool(self):
        assert checking.check_shape_except_batch((10, 2), (5, 2, 3), mode='bool') is False

    def test_shape_mismatch_raise(self):
        with pytest.raises(ValueError):
            checking.check_shape_except_batch((10, 2, 3), (5, 2, 4))

    def test_shape_mismatch_bool(self):
        assert checking.check_shape_except_batch((10, 2, 3), (5, 2, 4), mode='bool') is False


# --------------------------------------------------------------------------- #
# check_shape
# --------------------------------------------------------------------------- #
class TestCheckShape:
    def test_dict_input(self):
        free, fixed = checking.check_shape({'a': (1, 2, 3), 'b': (10, 2, 4)}, free_axes=-1)
        assert free == [3, 4]
        assert fixed == [10, 2]

    def test_bad_all_shapes_type(self):
        with pytest.raises(ValueError):
            checking.check_shape(123, free_axes=-1)

    def test_tuple_free_axes(self):
        free, fixed = checking.check_shape([(1, 2, 3), (10, 2, 4)], free_axes=(2,))
        assert free == [[3], [4]]

    def test_incompatible(self):
        with pytest.raises(ValueError):
            checking.check_shape([(2, 3), (4, 3)], free_axes=[1])


# --------------------------------------------------------------------------- #
# is_dict_data
# --------------------------------------------------------------------------- #
class TestIsDictData:
    def test_none_allowed(self):
        assert checking.is_dict_data(None, allow_none=True) is None

    def test_none_not_allowed(self):
        with pytest.raises(ValueError):
            checking.is_dict_data(None, allow_none=False)

    def test_not_a_dict(self):
        with pytest.raises(ValueError):
            checking.is_dict_data([1, 2], allow_none=False)

    def test_valid(self):
        d = {'a': 1.0}
        assert checking.is_dict_data(d, key_type=str, val_type=(int, float)) is d

    def test_bad_key(self):
        with pytest.raises(ValueError):
            checking.is_dict_data({1: 1.0}, key_type=str, val_type=float, name='d')

    def test_bad_value(self):
        with pytest.raises(ValueError):
            checking.is_dict_data({'a': 'x'}, key_type=str, val_type=float)


# --------------------------------------------------------------------------- #
# is_callable
# --------------------------------------------------------------------------- #
class TestIsCallable:
    def test_valid(self):
        f = lambda x: x
        assert checking.is_callable(f) is f

    def test_none_allowed(self):
        assert checking.is_callable(None, allow_none=True) is None

    def test_none_not_allowed(self):
        with pytest.raises(ValueError):
            checking.is_callable(None, name='fn', allow_none=False)

    def test_not_callable(self):
        with pytest.raises(ValueError):
            checking.is_callable(123)


# --------------------------------------------------------------------------- #
# is_initializer
# --------------------------------------------------------------------------- #
class TestIsInitializer:
    def test_none_allowed(self):
        assert checking.is_initializer(None, allow_none=True) is None

    def test_none_not_allowed(self):
        with pytest.raises(ValueError):
            checking.is_initializer(None, allow_none=False)

    def test_initializer_instance(self):
        ini = bp.init.ZeroInit()
        assert checking.is_initializer(ini) is ini

    def test_array(self):
        arr = bm.zeros(3)
        assert checking.is_initializer(arr) is arr

    def test_callable(self):
        f = lambda shape: bm.zeros(shape)
        assert checking.is_initializer(f) is f

    def test_invalid(self):
        with pytest.raises(ValueError):
            checking.is_initializer('not-an-init')


# --------------------------------------------------------------------------- #
# is_connector
# --------------------------------------------------------------------------- #
class TestIsConnector:
    def test_none_allowed(self):
        assert checking.is_connector(None, allow_none=True) is None

    def test_none_not_allowed(self):
        with pytest.raises(ValueError):
            checking.is_connector(None, allow_none=False)

    def test_connector_instance(self):
        c = bp.conn.All2All()
        assert checking.is_connector(c) is c

    def test_array(self):
        arr = bm.zeros(3)
        assert checking.is_connector(arr) is arr

    def test_callable(self):
        f = lambda: None
        assert checking.is_connector(f) is f

    def test_invalid(self):
        with pytest.raises(ValueError):
            checking.is_connector('not-a-conn')


# --------------------------------------------------------------------------- #
# is_sequence
# --------------------------------------------------------------------------- #
class TestIsSequence:
    def test_none_allowed(self):
        assert checking.is_sequence(None, allow_none=True) is None

    def test_none_not_allowed(self):
        with pytest.raises(ValueError):
            checking.is_sequence(None, allow_none=False)

    def test_not_sequence(self):
        with pytest.raises(ValueError):
            checking.is_sequence(123)

    def test_valid(self):
        seq = [1, 2, 3]
        assert checking.is_sequence(seq, elem_type=int) is seq

    def test_bad_elem(self):
        with pytest.raises(ValueError):
            checking.is_sequence([1, 'x'], elem_type=int)


# --------------------------------------------------------------------------- #
# is_float
# --------------------------------------------------------------------------- #
class TestIsFloat:
    def test_none_allowed(self):
        assert checking.is_float(None, allow_none=True) is None

    def test_none_not_allowed(self):
        with pytest.raises(ValueError):
            checking.is_float(None, allow_none=False)

    def test_valid_float(self):
        assert checking.is_float(1.5) == 1.5

    def test_int_allowed(self):
        assert checking.is_float(3, allow_int=True) == 3

    def test_int_not_allowed(self):
        with pytest.raises(ValueError):
            checking.is_float(3, allow_int=False)

    def test_numpy_floating(self):
        assert checking.is_float(np.float32(2.0), allow_int=False) == np.float32(2.0)

    def test_bad_type(self):
        with pytest.raises(ValueError):
            checking.is_float('x')

    def test_min_bound_ok(self):
        assert checking.is_float(5.0, min_bound=1.0) == 5.0

    def test_min_bound_branch(self):
        # P14-H2: eager out-of-bound checks now raise (previously the
        # jit_error_checking_no_args pure_callback path silently accepted them).
        with pytest.raises(Exception):
            checking.is_float(0.5, min_bound=1.0, name='v')

    def test_max_bound_ok(self):
        assert checking.is_float(5.0, max_bound=10.0) == 5.0

    def test_max_bound_branch(self):
        # P14-H2: eager out-of-bound checks now raise.
        with pytest.raises(Exception):
            checking.is_float(20.0, max_bound=10.0, name='v')


# --------------------------------------------------------------------------- #
# is_integer
# --------------------------------------------------------------------------- #
class TestIsInteger:
    def test_none_allowed(self):
        assert checking.is_integer(None, allow_none=True) is None

    def test_none_not_allowed(self):
        with pytest.raises(ValueError):
            checking.is_integer(None, allow_none=False)

    def test_valid_int(self):
        assert checking.is_integer(3) == 3

    def test_numpy_integer(self):
        assert checking.is_integer(np.int32(4)) == np.int32(4)

    def test_array_scalar_int(self):
        arr = np.array(5, dtype=np.int32)
        assert checking.is_integer(arr) is arr

    def test_array_non_int(self):
        arr = np.array([1, 2], dtype=np.int32)  # ndim != 0
        with pytest.raises(ValueError):
            checking.is_integer(arr)

    def test_bad_type(self):
        with pytest.raises(ValueError):
            checking.is_integer('x')

    def test_min_bound_ok(self):
        assert checking.is_integer(5, min_bound=1) == 5

    def test_min_bound_branch(self):
        # P14-H2: eager out-of-bound checks now raise.
        with pytest.raises(Exception):
            checking.is_integer(0, min_bound=1, name='v')

    def test_max_bound_ok(self):
        assert checking.is_integer(5, max_bound=10) == 5

    def test_max_bound_branch(self):
        # P14-H2: eager out-of-bound checks now raise.
        with pytest.raises(Exception):
            checking.is_integer(20, max_bound=10, name='v')


# --------------------------------------------------------------------------- #
# is_string
# --------------------------------------------------------------------------- #
class TestIsString:
    def test_none_allowed(self):
        assert checking.is_string(None, allow_none=True) is None

    def test_none_not_allowed(self):
        with pytest.raises(ValueError):
            checking.is_string(None, allow_none=False)

    def test_valid(self):
        assert checking.is_string('a', candidates=['a', 'b']) == 'a'

    def test_not_candidate(self):
        with pytest.raises(ValueError):
            checking.is_string('z', candidates=['a', 'b'])

    def test_no_candidates(self):
        assert checking.is_string('whatever') == 'whatever'


# --------------------------------------------------------------------------- #
# serialize_kwargs
# --------------------------------------------------------------------------- #
class TestSerializeKwargs:
    def test_none(self):
        assert checking.serialize_kwargs(None) == str({})

    def test_sorted(self):
        out = checking.serialize_kwargs({'b': 1, 'a': 2})
        assert out == str({'a': 2, 'b': 1})

    def test_bad_value_type(self):
        with pytest.raises(ValueError):
            checking.serialize_kwargs({'a': [1, 2, 3]})


# --------------------------------------------------------------------------- #
# is_subclass
# --------------------------------------------------------------------------- #
class TestIsSubclass:
    # NOTE: is_subclass checks ``issubclass(supported_type, type(instance))``.
    # i.e. the instance's class must be an *ancestor* (parent) of the supported
    # type(s), the opposite direction of isinstance. So an instance of a base
    # class passes the check for supported sub-classes.
    def test_single_type(self):
        class A: pass
        class B(A): pass
        a = A()
        # issubclass(B, type(a)=A) is True -> passes
        assert checking.is_subclass(a, B) is a

    def test_list_of_types_ok(self):
        class A: pass
        class B(A): pass
        class C(A): pass
        a = A()
        # any(issubclass(B, A), issubclass(C, A)) -> True
        assert checking.is_subclass(a, [B, C]) is a

    def test_bad_supported_types_container(self):
        class A: pass
        with pytest.raises(TypeError):
            checking.is_subclass(A(), 123)

    def test_bad_supported_types_elem(self):
        class A: pass
        with pytest.raises(TypeError):
            checking.is_subclass(A(), [A, 'not-a-type'])

    def test_not_supported(self):
        class A: pass
        class B: pass
        # issubclass(B, type(A())=A) is False -> NotImplementedError
        with pytest.raises(NotImplementedError):
            checking.is_subclass(A(), [B], name='thing')


# --------------------------------------------------------------------------- #
# is_instance
# --------------------------------------------------------------------------- #
class TestIsInstance:
    def test_ok(self):
        x = 5
        assert checking.is_instance(x, int) == 5

    def test_default_name(self):
        with pytest.raises(NotImplementedError):
            checking.is_instance('s', int)

    def test_named(self):
        with pytest.raises(NotImplementedError):
            checking.is_instance('s', int, name='value')


# --------------------------------------------------------------------------- #
# is_elem_or_seq_or_dict
# --------------------------------------------------------------------------- #
class TestIsElemOrSeqOrDict:
    def test_none(self):
        assert checking.is_elem_or_seq_or_dict(None, int, out_as='tuple') == ()

    def test_single_elem_tuple(self):
        out = checking.is_elem_or_seq_or_dict(5, int, out_as='tuple')
        assert out == (5,)

    def test_single_elem_list(self):
        out = checking.is_elem_or_seq_or_dict(5, int, out_as='list')
        assert out == [5]

    def test_seq(self):
        out = checking.is_elem_or_seq_or_dict([1, 2], int, out_as='tuple')
        assert out == (1, 2)

    def test_seq_bad_elem(self):
        with pytest.raises(ValueError):
            checking.is_elem_or_seq_or_dict([1, 'x'], int, out_as='tuple')

    def test_dict(self):
        out = checking.is_elem_or_seq_or_dict({'a': 1, 'b': 2}, int, out_as='dict')
        assert out == {'a': 1, 'b': 2}

    def test_dict_bad_value(self):
        with pytest.raises(ValueError):
            checking.is_elem_or_seq_or_dict({'a': 'x'}, int, out_as='dict')

    def test_unsupported_target(self):
        with pytest.raises(ValueError):
            checking.is_elem_or_seq_or_dict(3.5, int, out_as='tuple')

    def test_out_as_none(self):
        targets = [1, 2]
        assert checking.is_elem_or_seq_or_dict(targets, int, out_as=None) is targets

    def test_bad_out_as(self):
        with pytest.raises(AssertionError):
            checking.is_elem_or_seq_or_dict([1], int, out_as='bad')


# --------------------------------------------------------------------------- #
# is_all_vars / is_all_objs
# --------------------------------------------------------------------------- #
class TestIsAllVarsObjs:
    def test_all_vars(self):
        v = bm.Variable(bm.zeros(3))
        out = checking.is_all_vars([v], out_as='tuple')
        assert out == (v,)

    def test_all_vars_bad(self):
        with pytest.raises(ValueError):
            checking.is_all_vars([1, 2], out_as='tuple')

    def test_all_objs(self):
        obj = bp.dyn.Expon(1)
        out = checking.is_all_objs([obj], out_as='list')
        assert out == [obj]

    def test_all_objs_bad(self):
        with pytest.raises(ValueError):
            checking.is_all_objs([1], out_as='tuple')


# --------------------------------------------------------------------------- #
# jit error helpers
# --------------------------------------------------------------------------- #
class TestJitErrors:
    # P14-H2: for a *concrete* predicate these helpers now raise synchronously
    # (previously the ``jax.pure_callback`` path silently swallowed the raise).
    # Under tracing they keep the deferred ``cond`` + ``pure_callback`` path.
    def test_no_args_pred_false(self):
        # pred False -> false branch only, no raise
        checking.jit_error_checking_no_args(False, ValueError('boom'))

    def test_no_args_pred_true(self):
        # concrete True -> raises synchronously
        with pytest.raises(ValueError):
            checking.jit_error_checking_no_args(True, ValueError('boom'))

    def test_no_args_under_jit(self):
        # tracer predicate -> deferred, no raise at trace time
        @jax.jit
        def f(x):
            checking.jit_error_checking_no_args(x > 1.0, ValueError('boom'))
            return x

        assert float(f(0.0)) == 0.0

    def test_no_args_bad_err(self):
        with pytest.raises(AssertionError):
            checking.jit_error_checking_no_args(False, 'not-an-exception')

    def test_jit_error_false(self):
        def err_fun(arg):
            raise ValueError('boom')

        # pred False -> never invokes err_fun
        checking.jit_error(False, err_fun, bm.as_jax(bm.zeros(2)))

    def test_jit_error_true(self):
        def err_fun(arg):
            raise ValueError('boom')

        # concrete True -> raises synchronously
        with pytest.raises(ValueError):
            checking.jit_error(True, err_fun, bm.as_jax(bm.zeros(2)))

    def test_jit_error_true_tuple_arg(self):
        def err_fun(arg):
            raise ValueError('boom')

        # concrete True with a tuple err_arg -> raises synchronously
        with pytest.raises(ValueError):
            checking.jit_error(True, err_fun, (bm.as_jax(bm.zeros(2)), bm.as_jax(bm.ones(3))))

    def test_alias(self):
        assert checking.jit_error_checking is checking.jit_error
