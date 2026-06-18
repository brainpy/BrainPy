# -*- coding: utf-8 -*-
"""Supplementary coverage tests for ``brainpy/tools/functions.py``.

The pre-existing ``functions_test.py`` only covers the ``compose``/``pipe``
happy path. This file targets the remaining lines:

- ``identity``.
- ``instanceproperty`` used both with and without an explicit ``fget`` (the
  ``partial`` branch), and ``InstanceProperty`` access as a class attribute
  (returns ``classval``) vs an instance attribute, plus ``__reduce__``.
- ``Compose``: ``__call__``, ``__getstate__``/``__setstate__`` (pickle
  round-trip), ``__doc__`` (success and the AttributeError fallback),
  ``__name__`` (success and fallback), ``__repr__``, ``__eq__`` (equal,
  not-equal, NotImplemented), ``__ne__`` (incl. NotImplemented passthrough),
  ``__hash__``, ``__get__`` as a bound method, ``__signature__`` and
  ``__wrapped__``.
- ``compose`` with zero args (-> identity), one arg (passthrough), many args.
- ``pipe`` ordering.
"""

import inspect
import pickle

import pytest

from brainpy.tools import functions as F


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------

def test_identity():
    assert F.identity(3) == 3
    obj = object()
    assert F.identity(obj) is obj


# --------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# --------------------------------------------------------------------------

def test_instanceproperty_class_vs_instance():
    class MyClass:
        """The class docstring"""

        @F.instanceproperty(classval=__doc__)
        def __doc__(self):
            return "An object docstring"

        @F.instanceproperty
        def val(self):
            return 42

    # Accessed on the class -> classval
    assert MyClass.__doc__ == "The class docstring"
    assert MyClass.val is None
    # Accessed on the instance -> fget result
    obj = MyClass()
    assert obj.__doc__ == "An object docstring"
    assert obj.val == 42


def test_instanceproperty_partial_branch():
    # Calling instanceproperty with fget=None returns a partial.
    deferred = F.instanceproperty(classval="cv")
    assert callable(deferred)
    prop = deferred(lambda self: 7)
    assert isinstance(prop, F.InstanceProperty)
    assert prop.classval == "cv"


def test_instanceproperty_reduce():
    prop = F.instanceproperty(lambda self: 1, classval="cv")
    cls, state = prop.__reduce__()
    assert cls is F.InstanceProperty
    # state is (fget, fset, fdel, __doc__, classval)
    assert state[-1] == "cv"
    assert len(state) == 5


# --------------------------------------------------------------------------
# Compose basics
# --------------------------------------------------------------------------

def _inc(i):
    return i + 1


def _double(i):
    return 2 * i


def test_compose_call():
    c = F.Compose((str, _inc))
    # compose(str, inc)(3) == str(inc(3)) == '4'
    assert c(3) == "4"


def test_compose_getstate_setstate_pickle():
    c = F.compose(str, _inc)
    state = c.__getstate__()
    assert state == (c.first, c.funcs)

    # round-trip via pickle exercises __getstate__/__setstate__
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3) == "4"

    # manual setstate
    c2 = F.Compose((str, _inc))
    c2.__setstate__((_inc, (_double,)))
    assert c2.first is _inc
    assert c2.funcs == (_double,)


def test_compose_doc_success():
    c = F.compose(str, _inc)
    doc = c.__doc__
    assert doc.startswith("lambda *args, **kwargs:")
    assert "_inc" in doc and "str" in doc


def test_compose_doc_attributeerror_fallback():
    # A callable without __name__ triggers the AttributeError fallback in __doc__
    class NoName:
        def __call__(self, x):
            return x

    c = F.Compose((NoName(), _inc))
    assert c.__doc__ == "A composition of functions"


def test_compose_name_success():
    # compose(str, _inc) -> first=_inc, funcs=(str,); __name__ joins the
    # reversed (first,)+funcs == (str, _inc) with "_of_".
    c = F.compose(str, _inc)
    assert c.__name__ == "str_of__inc"


def test_compose_name_attributeerror_fallback():
    class NoName:
        def __call__(self, x):
            return x

    c = F.Compose((NoName(), _inc))
    # fallback returns the class name
    assert c.__name__ == "Compose"


def test_compose_repr():
    c = F.compose(str, _inc)
    r = repr(c)
    assert r.startswith("Compose(")


def test_compose_eq_and_ne():
    a = F.compose(str, _inc)
    b = F.compose(str, _inc)
    diff = F.compose(str, _double)
    assert a == b
    assert a != diff
    assert not (a != b)


def test_compose_eq_notimplemented():
    a = F.compose(str, _inc)
    # __eq__ against a non-Compose returns NotImplemented -> Python falls back,
    # so equality with an unrelated object is False.
    assert (a == 123) is False
    # __ne__ passes through the NotImplemented from __eq__
    assert (a != 123) is True


def test_compose_hash():
    a = F.compose(str, _inc)
    b = F.compose(str, _inc)
    assert hash(a) == hash(b)
    # usable as a dict key
    assert {a: 1}[b] == 1


def test_compose_get_bound_method():
    # When Compose is a class attribute, __get__ binds it like a method.
    class Holder:
        transform = F.compose(str, _inc)

    h = Holder()
    # accessed on instance -> bound MethodType, self passed as first arg.
    # first(self) would be _inc(h) which fails; instead exercise class access.
    assert Holder.transform is Holder.__dict__["transform"]
    bound = Holder.transform.__get__(h, Holder)
    from types import MethodType
    assert isinstance(bound, MethodType)


def test_compose_signature():
    def f(a) -> str:
        return str(a)

    def g(x: int) -> int:
        return x

    c = F.compose(f, g)
    sig = c.__signature__
    assert isinstance(sig, inspect.Signature)
    # __signature__ uses base = signature(first) (=g, params x:int) and the
    # return_annotation from funcs[-1] (=f -> str).
    assert sig.return_annotation is str
    assert "x" in sig.parameters


def test_compose_wrapped():
    c = F.compose(str, _inc)
    # __wrapped__ returns the `first` function
    assert c.__wrapped__ is c.first


# --------------------------------------------------------------------------
# compose / pipe top-level
# --------------------------------------------------------------------------

def test_compose_no_args_returns_identity():
    assert F.compose() is F.identity


def test_compose_single_arg_passthrough():
    assert F.compose(_inc) is _inc


def test_compose_many_args():
    c = F.compose(str, _inc, _double)
    # str(inc(double(3))) = str(inc(6)) = str(7) = '7'
    assert c(3) == "7"


def test_pipe_ordering():
    p = F.pipe(_double, str)
    # pipe(double, str)(3) == str(double(3)) == '6'
    assert p(3) == "6"
