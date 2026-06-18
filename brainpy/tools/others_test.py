# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy/tools/others.py``.

Exercises:
- ``one_of``: default fall-through, single choice, custom names, and the
  multiple-choice ValueError.
- ``replicate``: scalar / str, length-1 sequence, exact-length sequence, and
  the mismatched-length TypeError.
- ``not_customized`` decorator marker.
- ``size2num``: int, numpy integer, tuple/list product, and the unsupported
  type ValueError.
- ``to_size``: tuple/list, int, numpy integer, None, and the ValueError.
- ``timeout``: wraps a fast function and returns its result without firing.
"""

import numpy as np
import pytest

from brainpy.tools import others as O


# --------------------------------------------------------------------------
# one_of
# --------------------------------------------------------------------------

def test_one_of_default_when_all_none():
    assert O.one_of("default", None, None) == "default"


def test_one_of_single_choice():
    assert O.one_of("default", None, 5, None) == 5


def test_one_of_custom_names_in_error():
    with pytest.raises(ValueError, match="Provide one of"):
        O.one_of("default", 1, 2, names=["first", "second"])


def test_one_of_multiple_choices_raises():
    with pytest.raises(ValueError):
        O.one_of(None, 1, 2)


# --------------------------------------------------------------------------
# replicate
# --------------------------------------------------------------------------

def test_replicate_scalar():
    assert O.replicate(3, 4, "x") == (3, 3, 3, 3)


def test_replicate_string_treated_as_scalar():
    assert O.replicate("ab", 2, "x") == ("ab", "ab")


def test_replicate_length_one_sequence():
    assert O.replicate([7], 3, "x") == (7, 7, 7)


def test_replicate_exact_length_sequence():
    assert O.replicate([1, 2, 3], 3, "x") == (1, 2, 3)


def test_replicate_mismatch_raises():
    with pytest.raises(TypeError, match="must be a scalar or sequence"):
        O.replicate([1, 2], 3, "myparam")


# --------------------------------------------------------------------------
# not_customized
# --------------------------------------------------------------------------

def test_not_customized():
    @O.not_customized
    def f():
        return 1

    assert f.not_customized is True


# --------------------------------------------------------------------------
# size2num
# --------------------------------------------------------------------------

def test_size2num_int():
    assert O.size2num(5) == 5


def test_size2num_numpy_integer():
    assert O.size2num(np.int32(7)) == 7


def test_size2num_tuple_and_list():
    assert O.size2num((2, 3, 4)) == 24
    assert O.size2num([2, 5]) == 10


def test_size2num_bad_type_raises():
    with pytest.raises(ValueError, match="Do not support type"):
        O.size2num("nope")


# --------------------------------------------------------------------------
# to_size
# --------------------------------------------------------------------------

def test_to_size_tuple_and_list():
    assert O.to_size((2, 3)) == (2, 3)
    assert O.to_size([4, 5]) == (4, 5)


def test_to_size_int():
    assert O.to_size(6) == (6,)


def test_to_size_numpy_integer():
    assert O.to_size(np.int64(8)) == (8,)


def test_to_size_none():
    assert O.to_size(None) is None


def test_to_size_bad_type_raises():
    with pytest.raises(ValueError, match="Cannot make a size"):
        O.to_size("bad")


# --------------------------------------------------------------------------
# timeout
# --------------------------------------------------------------------------

def test_timeout_returns_result_for_fast_function():
    @O.timeout(5)
    def fast(a, b):
        return a + b

    assert fast(2, 3) == 5


def test_timeout_propagates_exception_and_cancels_timer():
    @O.timeout(5)
    def boom():
        raise RuntimeError("inner error")

    with pytest.raises(RuntimeError, match="inner error"):
        boom()
