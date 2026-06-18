# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy/math/surrogate/_utils.py``.

Exercises the surrogate-gradient helper machinery:

* ``get_default`` (None vs provided).
* ``make_return`` (scalar-with-default path) -- and PINS the tuple-with-default
  defect (see NOTE below).
* ``_get_args`` -- POSITIONAL_OR_KEYWORD, VAR_POSITIONAL, and the three
  ``UnsupportedError`` branches (KEYWORD_ONLY / POSITIONAL_ONLY / VAR_KEYWORD).
* ``VJPCustom`` -- construction with and without ``statics`` (the
  ``itertools.product`` caching path + ``_str_static_arg``), the
  ``statics``-without-``defaults`` ``KeyError`` guard, and ``__call__`` covering
  positional / kwarg / default argument resolution, the ``Array`` -> ``.value``
  unwrap, and the missing-arg / unknown-kwarg error branches.
* the public ``vjp_custom`` decorator (docstring propagation + gradient flow).

NOTE: the final ``else: raise UnsupportedError()`` in ``_get_args`` (line 68) is
dead code -- ``inspect.Parameter.kind`` only ever takes the five values already
handled by the preceding branches, so the catch-all ``else`` is unreachable.
"""

import jax
import jax.numpy as jnp
import pytest

import brainpy.math as bm
from brainpy._errors import UnsupportedError
from brainpy.math.surrogate._utils import (
    get_default, make_return, _get_args, VJPCustom, vjp_custom,
)


# ---------------------------------------------------------------------------
# get_default
# ---------------------------------------------------------------------------

def test_get_default_none_returns_fallback():
    val, provided = get_default(None, 42)
    assert val == 42
    assert provided is False


def test_get_default_value_kept():
    val, provided = get_default(7, 42)
    assert val == 7
    assert provided is True


# ---------------------------------------------------------------------------
# make_return
# ---------------------------------------------------------------------------

def test_make_return_scalar_with_default():
    # scalar r -> wrapped in a list; truthy default appends a None
    assert make_return(5, True, False) == (5, None)
    # no truthy args -> just the wrapped scalar
    assert make_return(5) == (5,)


def test_make_return_tuple_no_default():
    # tuple/list r with no truthy default flows through cleanly
    assert make_return((1, 2)) == (1, 2)
    assert make_return([1, 2], False) == (1, 2)


def test_make_return_tuple_with_default_is_buggy():
    # NOTE: DEFECT in make_return -- when ``r`` is a tuple/list AND any
    # ``args`` entry is truthy, the function does ``tuple(r) += [None]`` which
    # raises ``TypeError: can only concatenate tuple (not "list") to tuple``.
    # It should append ``(None,)`` (a tuple) instead of ``[None]`` (a list).
    # Pinning the current broken behaviour so a future fix is detected.
    with pytest.raises(TypeError):
        make_return([1, 2], True)


# ---------------------------------------------------------------------------
# _get_args
# ---------------------------------------------------------------------------

def test_get_args_positional_and_varargs():
    def f(a, b, *rest):
        pass
    assert _get_args(f) == ['a', 'b', '*rest']


def test_get_args_keyword_only_unsupported():
    def f(a, *, b):
        pass
    with pytest.raises(UnsupportedError):
        _get_args(f)


def test_get_args_positional_only_unsupported():
    def f(a, /, b):
        pass
    with pytest.raises(UnsupportedError):
        _get_args(f)


def test_get_args_var_keyword_unsupported():
    def f(a, **kw):
        pass
    with pytest.raises(UnsupportedError):
        _get_args(f)


# ---------------------------------------------------------------------------
# VJPCustom construction
# ---------------------------------------------------------------------------

def test_vjpcustom_no_statics_basic():
    @vjp_custom(['x', 'y'], defaults=dict(alpha=2.0))
    def f(x, y, alpha):
        def grad(dz):
            return dz * alpha, dz, None
        return x * y, grad

    # positional + default alpha resolution
    out = float(f(jnp.asarray(3.0), jnp.asarray(4.0)))
    assert out == pytest.approx(12.0)
    # gradient flows; dx == dz*alpha == 1*2 == 2
    gx = jax.grad(lambda x, y: f(x, y))(jnp.asarray(3.0), jnp.asarray(4.0))
    assert float(gx) == pytest.approx(2.0)


def test_vjpcustom_named_positional_arg_via_kwarg():
    # supply a declared positional arg (in ``self.args``) by keyword -> exercises
    # the ``args.append(kwargs.pop(k))`` resolution branch.
    @vjp_custom(['x', 'y'], defaults={})
    def f(x, y):
        def grad(dz):
            return dz, dz
        return x + y, grad

    assert float(f(jnp.asarray(3.0), y=jnp.asarray(4.0))) == pytest.approx(7.0)


def test_vjpcustom_default_passed_as_kwarg():
    @vjp_custom(['x'], defaults=dict(alpha=2.0))
    def f(x, alpha):
        def grad(dz):
            return dz * alpha, None
        return x * alpha, grad

    # override the default via kwarg -> exercises the kwargs.pop branch
    assert float(f(jnp.asarray(3.0), alpha=10.0)) == pytest.approx(30.0)


def test_vjpcustom_with_statics_caches_per_choice():
    # statics produce an itertools.product over their choices; each choice gets
    # its own cached custom_gradient keyed by _str_static_arg.
    @vjp_custom(['x'], defaults=dict(mode=0), statics=dict(mode=[0, 1]))
    def f(x, mode):
        def grad(dz):
            return dz * (mode + 1), None
        return x * (mode + 1), grad

    assert float(f(jnp.asarray(2.0))) == pytest.approx(2.0)            # default mode=0
    assert float(f(jnp.asarray(2.0), mode=1)) == pytest.approx(4.0)    # kwarg mode=1
    assert float(f(jnp.asarray(2.0), 1)) == pytest.approx(4.0)         # positional mode=1


def test_vjpcustom_array_argument_is_unwrapped():
    # passing a brainpy Array exercises the ``isinstance(v, Array): v = v.value`` line
    @vjp_custom(['x', 'y'], defaults={})
    def f(x, y):
        def grad(dz):
            return dz, dz
        return x + y, grad

    out = float(f(bm.asarray(3.0), bm.asarray(4.0)))
    assert out == pytest.approx(7.0)


def test_vjpcustom_statics_without_default_raises():
    def fn(x, mode):
        return x, (lambda dz: (dz, None))
    with pytest.raises(KeyError):
        VJPCustom(fn, ['x'], defaults={}, statics=dict(mode=[0]))


def test_vjpcustom_missing_required_arg_raises():
    @vjp_custom(['x', 'y'], defaults={})
    def f(x, y):
        def grad(dz):
            return dz, dz
        return x + y, grad

    with pytest.raises(ValueError):
        f(jnp.asarray(1.0))   # y not provided


def test_vjpcustom_unknown_kwargs_raises():
    @vjp_custom(['x', 'y'], defaults={})
    def f(x, y):
        def grad(dz):
            return dz, dz
        return x + y, grad

    with pytest.raises(KeyError):
        f(jnp.asarray(1.0), jnp.asarray(2.0), bogus=3)


def test_vjp_custom_propagates_docstring():
    @vjp_custom(['x'], defaults={})
    def f(x):
        """my doc"""
        def grad(dz):
            return (dz,)
        return x, grad

    assert f.__doc__ == "my doc"
