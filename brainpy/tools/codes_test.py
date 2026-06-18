# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy/tools/codes.py``.

Exercises:
- ``repr_dict``.
- ``repr_object`` for: a BrainPyObject instance, plain callables (with and
  without keyword defaults), ``functools.partial`` (the ``x = x.func`` unwrap
  loop and the no-``__name__``/has-``__class__`` fallback), and non-callable
  objects (single & multi-line repr).
- ``repr_context``.
- ``copy_doc``.
- ``code_lines_to_func`` for both the success path and the compile-error path.
- ``get_identifiers`` (with & without numbers), ``indent``, ``deindent``
  (all branches: explicit num_tabs, auto-detect, blank input, docstring mode,
  docstring with <2 lines), ``word_replace`` (exclude_dot True/False).
- ``change_func_name``, ``is_lambda_function``, ``get_func_source`` (with &
  without a decorator prefix), and ``get_main_code`` for: None, lambda,
  unparsable lambda, a normal function, an unparsable normal function, and a
  non-callable (the ValueError).
"""

import functools

import pytest

from brainpy.tools import codes as C


# --------------------------------------------------------------------------
# repr_dict
# --------------------------------------------------------------------------

def test_repr_dict():
    assert C.repr_dict({"a": 1, "b": 2}) == "a=1, b=2"
    assert C.repr_dict({}) == ""


# --------------------------------------------------------------------------
# repr_object
# --------------------------------------------------------------------------

def test_repr_object_brainpy_object():
    from brainpy.math import BrainPyObject
    obj = BrainPyObject()
    out = C.repr_object(obj)
    assert isinstance(out, str)
    # repr_object delegates straight to repr() for BrainPyObject instances.
    assert out == repr(obj)


def test_repr_object_callable_no_defaults():
    def f(x):
        return x

    assert C.repr_object(f) == "f"


def test_repr_object_callable_with_defaults():
    def f(x, y=10, z=20):
        return x

    out = C.repr_object(f)
    assert out.startswith("f(*, ")
    assert "y=10" in out
    assert "z=20" in out


def test_repr_object_partial_unwrap():
    def base(a, b):
        return a + b

    p = functools.partial(base, 1)
    out = C.repr_object(p)
    # partial has no __name__, loop unwraps to .func -> 'base'
    assert out == "base"


def test_repr_object_callable_class_fallback():
    # A callable instance with neither __name__ nor .func -> class-name fallback
    class CallMe:
        def __call__(self):
            return 1

    out = C.repr_object(CallMe())
    assert out == "CallMe"


def test_repr_object_noncallable_single_line():
    assert C.repr_object(123) == "123"


def test_repr_object_noncallable_multiline():
    # A non-callable whose repr genuinely spans multiple lines: continuation
    # lines are indented by two spaces.
    class MultiLine:
        def __repr__(self):
            return "first\nsecond\nthird"

    out = C.repr_object(MultiLine())
    assert out == "first\n  second\n  third"


# --------------------------------------------------------------------------
# repr_context
# --------------------------------------------------------------------------

def test_repr_context():
    out = C.repr_context("a\nb\nc", ">>")
    assert out == "a\n>>b\n>>c"


# --------------------------------------------------------------------------
# copy_doc
# --------------------------------------------------------------------------

def test_copy_doc():
    def src():
        """Source docstring."""

    @C.copy_doc(src)
    def dst():
        pass

    assert dst.__doc__ == "Source docstring."


# --------------------------------------------------------------------------
# code_lines_to_func
# --------------------------------------------------------------------------

def test_code_lines_to_func_success():
    import sys as _sys
    scope = {"sys": _sys}
    code, func = C.code_lines_to_func(
        ["return a + b"], "myfunc", ["a", "b"], scope
    )
    assert "def myfunc(a, b):" in code
    assert func(2, 3) == 5


def test_code_lines_to_func_runtime_error_reports_line():
    import sys as _sys
    scope = {"sys": _sys}
    code, func = C.code_lines_to_func(
        ["return 1 / 0"], "boom", [], scope, remind="check division"
    )
    with pytest.raises(ValueError, match="Error occurred in line"):
        func()


def test_code_lines_to_func_compile_error():
    scope = {}
    # invalid python body -> compile fails -> wrapped ValueError
    with pytest.raises(ValueError, match="Compilation function error"):
        C.code_lines_to_func(["this is not valid python $$"], "bad", [], scope)


# --------------------------------------------------------------------------
# get_identifiers
# --------------------------------------------------------------------------

def test_get_identifiers_basic():
    expr = "3-a*_b+c5+8+f(A - .3e-10, tau_2)*17"
    ids = C.get_identifiers(expr)
    assert sorted(ids) == ["A", "_b", "a", "c5", "f", "tau_2"]


def test_get_identifiers_with_numbers():
    expr = "3-a*_b+c5+8+f(A - .3e-10, tau_2)*17"
    ids = C.get_identifiers(expr, include_numbers=True)
    assert "17" in ids
    assert "8" in ids
    assert ".3e-10" in ids


def test_get_identifiers_strips_keywords():
    ids = C.get_identifiers("a and b or not True")
    assert "and" not in ids
    assert "True" not in ids
    assert {"a", "b"}.issubset(ids)


# --------------------------------------------------------------------------
# indent / deindent
# --------------------------------------------------------------------------

def test_indent_default_and_custom():
    assert C.indent("a\nb") == "    a\n    b"
    assert C.indent("a\nb", num_tabs=2, tab="-") == "--a\n--b"


def test_deindent_auto_detect():
    text = "    a\n      b"
    assert C.deindent(text) == "a\n  b"


def test_deindent_explicit_num_tabs():
    text = "    a\n    b"
    assert C.deindent(text, num_tabs=1) == "a\nb"


def test_deindent_blank_lines_only():
    # only-blank lines -> indent_level 0 (empty line_seq branch)
    text = "\n\n"
    assert C.deindent(text) == "\n\n"


def test_deindent_docstring_short():
    # docstring mode with <2 lines returns text unchanged
    text = "single"
    assert C.deindent(text, docstring=True) == "single"


def test_deindent_docstring_mode():
    text = "first\n    second\n    third"
    out = C.deindent(text, docstring=True)
    assert out == "first\nsecond\nthird"


def test_deindent_converts_tabs():
    text = "\tindented"
    out = C.deindent(text)
    assert "\t" not in out


# --------------------------------------------------------------------------
# word_replace
# --------------------------------------------------------------------------

def test_word_replace_basic():
    expr = "a*_b+c5+8+f(A)"
    assert C.word_replace(expr, {"a": "banana", "f": "func"}) == "banana*_b+c5+8+func(A)"


def test_word_replace_exclude_dot():
    # with exclude_dot True (default), 'x' in 'obj.x' is NOT replaced
    out = C.word_replace("x + obj.x", {"x": "Y"}, exclude_dot=True)
    assert out == "Y + obj.x"


def test_word_replace_include_dot():
    out = C.word_replace("x + obj.x", {"x": "Y"}, exclude_dot=False)
    assert out == "Y + obj.Y"


# --------------------------------------------------------------------------
# change_func_name / is_lambda_function
# --------------------------------------------------------------------------

def test_change_func_name():
    def f():
        return 1

    g = C.change_func_name(f, "renamed")
    assert g.__name__ == "renamed"


def test_is_lambda_function():
    assert C.is_lambda_function(lambda x: x) is True

    def f():
        return 1

    assert C.is_lambda_function(f) is False


# --------------------------------------------------------------------------
# get_func_source
# --------------------------------------------------------------------------

def test_get_func_source_plain():
    def f(x):
        return x

    src = C.get_func_source(f)
    assert src.startswith("def f")


def test_get_func_source_decorated():
    def deco(fn):
        return fn

    @deco
    def g(x):
        return x

    src = C.get_func_source(g)
    # leading decorator stripped, starts at 'def '
    assert src.startswith("def g")


# --------------------------------------------------------------------------
# get_main_code
# --------------------------------------------------------------------------

def test_get_main_code_none():
    assert C.get_main_code(None) == ""


def test_get_main_code_lambda():
    out = C.get_main_code(lambda x: x + 1)
    assert out.strip().startswith("return")
    assert "x + 1" in out or "x+1" in out


def test_get_main_code_lambda_unparsable():
    # provide a codes string with no ':' -> raises
    with pytest.raises(ValueError, match="Can not parse function"):
        C.get_main_code(lambda x: x, codes="lambda x x")


def test_get_main_code_normal_function():
    def f(a, b):
        c = a + b
        return c

    src = C.get_func_source(f)
    out = C.get_main_code(f, codes=src)
    assert "return c" in out
    assert "c = a + b" in out


def test_get_main_code_normal_function_unparsable():
    # codes string has no '):' line -> raises ValueError
    with pytest.raises(ValueError, match="Can not parse function"):
        C.get_main_code(lambda: 0, codes="no_colon_here\nstill_none")  # noqa


def test_get_main_code_non_callable():
    with pytest.raises(ValueError, match="Unknown function type"):
        C.get_main_code(123)
