# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy/tools/progress.py``.

Exercises:
- ``func_dump`` / ``func_load`` round-trips (with/without defaults & closures,
  list-packed inputs, ascii vs raw_unicode_escape decode fallback).
- ``Progbar`` construction and the ``update`` rendering paths for verbose
  modes 0/1/2, target-known vs target-unknown, stateful vs averaged metrics,
  ETA formatting branches (seconds / minutes / hours), finalize branch, the
  dynamic-display vs newline branch and the interval-skip early return.
- ``Progbar.add``, ``_format_time`` (s / ms / us), ``_estimate_step_duration``
  (current==0, current==1, current>1) and ``_update_stateful_metrics``.
- ``make_batches``, ``slice_arrays`` (every branch incl. the ValueError),
  ``to_list``, ``to_snake_case`` (incl. private "_" prefix), the
  ``check_for_unexpected_keys`` raise branch, ``validate_kwargs`` raise branch,
  ``default`` / ``is_default``, ``populate_dict_with_module_objects``,
  ``LazyLoader`` lazy import, and ``print_msg`` (both line_break branches).
"""

import sys
import types

import numpy as np
import pytest

from brainpy.tools import progress as P


# --------------------------------------------------------------------------
# func_dump / func_load
# --------------------------------------------------------------------------

def test_func_dump_load_simple_roundtrip():
    def f(x):
        return x + 1

    code, defaults, closure = P.func_dump(f)
    assert isinstance(code, str)
    assert defaults is None
    assert closure is None
    g = P.func_load(code, defaults, closure)
    assert g(10) == 11


def test_func_dump_load_with_defaults_and_closure():
    captured = 7

    def make(base):
        def f(x, y=3):
            return x + y + base + captured

        return f

    f = make(100)
    code, defaults, closure = P.func_dump(f)
    assert defaults == (3,)
    # closure captures both `base` and `captured`
    assert closure is not None
    g = P.func_load(code, defaults, closure)
    assert g(1) == 1 + 3 + 100 + 7


def test_func_load_unpacks_tuple_and_list_defaults():
    def f(x, y=5):
        return x * y

    code, defaults, closure = P.func_dump(f)
    # packed as a single tuple argument -> exercises the unpack branch
    g = P.func_load((code, defaults, closure))
    assert g(2) == 10

    # list-defaults are coerced to tuple
    g2 = P.func_load((code, list(defaults), closure))
    assert g2(2) == 10


def test_func_load_raw_unicode_escape_fallback():
    # A string that cannot be ascii/base64-decoded triggers the except branch.
    def f():
        return 42

    code, defaults, closure = P.func_dump(f)
    # decode normally first to obtain real bytes, then feed raw bytes string
    # that fails base64 -> falls back to raw_unicode_escape.
    raw = code.encode("ascii")
    import codecs
    real_bytes = codecs.decode(raw, "base64")
    fallback_str = real_bytes.decode("raw_unicode_escape")
    g = P.func_load(fallback_str, defaults, closure)
    assert g() == 42


def _make_cell(val):
    def inner():
        return val

    return inner.__closure__[0]


def test_func_load_closure_already_cell():
    # When a closure entry is ALREADY a cell object, ensure_value_to_cell
    # returns it unchanged (the ``return value`` branch).
    def make():
        w = 5

        def g():
            return w

        return g

    g = make()
    code, defaults, _ = P.func_dump(g)
    cell = _make_cell(123)
    loaded = P.func_load(code, defaults, (cell,))
    assert loaded() == 123


def test_ensure_value_to_cell_non_cell_value():
    # When a closure entry is a plain value (not a cell), func_load wraps it.
    def make():
        z = 99

        def f():
            return z

        return f

    f = make()
    code, defaults, closure = P.func_dump(f)
    # closure here is a tuple of raw values (99,), not cell objects
    assert closure == (99,)
    g = P.func_load(code, defaults, closure)
    assert g() == 99


# --------------------------------------------------------------------------
# Progbar
# --------------------------------------------------------------------------

def test_progbar_verbose0_no_output(capsys):
    pb = P.Progbar(target=5, verbose=0)
    pb.update(1, values=[("loss", 0.5)])
    pb.update(5, values=[("loss", 0.1)])
    # verbose 0 prints nothing
    assert capsys.readouterr().out == ""


def test_progbar_verbose1_target_known(capsys):
    pb = P.Progbar(target=4, verbose=1, interval=0.0)
    pb.update(1, values=[("loss", 1.0)])
    pb.update(2, values=[("loss", 0.5)])
    pb.update(4, values=[("loss", 0.001)], finalize=True)
    out = capsys.readouterr().out
    assert "4/4" in out
    assert "loss" in out


def test_progbar_verbose1_small_avg_uses_scientific(capsys):
    pb = P.Progbar(target=2, verbose=1, interval=0.0)
    # very small averaged metric -> abs(avg) <= 1e-3 branch (scientific format)
    pb.update(1, values=[("tiny", 1e-9)])
    pb.update(2, values=[("tiny", 1e-9)], finalize=True)
    out = capsys.readouterr().out
    assert "tiny" in out
    assert "e-" in out  # scientific notation


def test_progbar_verbose1_target_unknown(capsys):
    pb = P.Progbar(target=None, verbose=1, interval=0.0)
    pb.update(1, values=[("metric", 2.0)])
    pb.update(3, values=[("metric", 4.0)])
    out = capsys.readouterr().out
    assert "Unknown" in out


def test_progbar_verbose1_interval_skip_returns_early(capsys):
    pb = P.Progbar(target=100, verbose=1, interval=1000.0)
    # First update always renders (last_update == 0). Drain it.
    pb.update(1, values=[("loss", 1.0)])
    capsys.readouterr()
    # Second update within the (huge) interval & not finalize -> early return.
    pb.update(2, values=[("loss", 1.0)])
    assert capsys.readouterr().out == ""


def test_progbar_verbose1_eta_minutes_and_hours(monkeypatch, capsys):
    # Force a huge per-step duration so ETA spans hours then minutes.
    pb = P.Progbar(target=1000, verbose=1, interval=0.0)
    monkeypatch.setattr(pb, "_estimate_step_duration", lambda current, now: 100.0)
    pb.update(1)  # eta = 100 * 999 -> hours branch
    out1 = capsys.readouterr().out
    assert ":" in out1  # h:mm:ss format

    pb2 = P.Progbar(target=10, verbose=1, interval=0.0)
    monkeypatch.setattr(pb2, "_estimate_step_duration", lambda current, now: 10.0)
    pb2.update(1)  # eta = 10 * 9 = 90 -> minutes branch
    out2 = capsys.readouterr().out
    assert ":" in out2


def test_progbar_verbose1_eta_seconds(monkeypatch, capsys):
    pb = P.Progbar(target=10, verbose=1, interval=0.0)
    monkeypatch.setattr(pb, "_estimate_step_duration", lambda current, now: 1.0)
    pb.update(1)  # eta = 1 * 9 = 9s -> seconds branch
    out = capsys.readouterr().out
    assert "ETA" in out


def test_progbar_non_dynamic_display_uses_newline(monkeypatch, capsys):
    pb = P.Progbar(target=3, verbose=1, interval=0.0)
    # Force the non-dynamic branch (message += "\n")
    pb._dynamic_display = False
    pb.update(1, values=[("loss", 1.0)])
    pb.update(3, values=[("loss", 0.5)], finalize=True)
    out = capsys.readouterr().out
    assert "\n" in out


def test_progbar_prev_total_width_padding(monkeypatch, capsys):
    # Exercise the prev_total_width > total_width padding branch.
    pb = P.Progbar(target=10, verbose=1, interval=0.0)
    pb._dynamic_display = True
    pb.update(1, values=[("a_metric_with_long_name", 1.0)])
    # Inflate prev width so the next, shorter render triggers padding
    pb._total_width = 10_000
    pb.update(2, values=[("a", 1.0)])
    out = capsys.readouterr().out
    assert out  # produced something


def test_progbar_verbose2_finalize(capsys):
    pb = P.Progbar(target=3, verbose=2, interval=0.0)
    # non-finalize updates in verbose 2 print nothing
    pb.update(1, values=[("loss", 1.0)])
    assert capsys.readouterr().out == ""
    # finalize triggers the verbose-2 render including epoch timing
    pb.update(3, values=[("loss", 0.5)], finalize=True)
    out = capsys.readouterr().out
    assert "3/3" in out
    assert "loss" in out


def test_progbar_verbose2_small_avg_scientific(capsys):
    pb = P.Progbar(target=2, verbose=2, interval=0.0)
    pb.update(1, values=[("tiny", 1e-9)])
    pb.update(2, values=[("tiny", 1e-9)], finalize=True)
    out = capsys.readouterr().out
    assert "e-" in out


def test_progbar_stateful_metrics(capsys):
    pb = P.Progbar(target=3, verbose=1, interval=0.0, stateful_metrics=["acc"])
    pb.update(1, values=[("acc", 0.9), ("loss", 1.0)])
    pb.update(3, values=[("acc", 0.95), ("loss", 0.5)], finalize=True)
    out = capsys.readouterr().out
    assert "acc" in out


def test_progbar_add(capsys):
    pb = P.Progbar(target=4, verbose=1, interval=0.0)
    pb.add(2, values=[("loss", 1.0)])
    assert pb._seen_so_far == 2
    pb.add(2, values=[("loss", 0.5)])
    assert pb._seen_so_far == 4
    assert capsys.readouterr().out  # produced output


def test_progbar_format_time_branches():
    pb = P.Progbar(target=1, verbose=0)
    assert "s/step" in pb._format_time(2.0, "step")
    assert "s/step" in pb._format_time(0.0, "step")
    assert "ms/step" in pb._format_time(0.01, "step")
    assert "us/step" in pb._format_time(1e-5, "step")


def test_progbar_estimate_step_duration_branches():
    pb = P.Progbar(target=10, verbose=0)
    now = pb._start
    # current == 0 -> returns 0
    assert pb._estimate_step_duration(0, now) == 0
    # current == 1 -> sets _time_after_first_step
    d1 = pb._estimate_step_duration(1, now + 1.0)
    assert d1 >= 0
    assert pb._time_after_first_step is not None
    # current > 1 with _time_after_first_step set -> other branch
    d2 = pb._estimate_step_duration(3, now + 5.0)
    assert d2 >= 0


def test_progbar_estimate_step_duration_current_gt1_no_first_step():
    pb = P.Progbar(target=10, verbose=0)
    now = pb._start
    # _time_after_first_step is None and current>1 -> fallback to simple calc
    d = pb._estimate_step_duration(5, now + 10.0)
    assert d == pytest.approx(2.0, rel=0.5)


def test_progbar_update_stateful_metrics():
    pb = P.Progbar(target=1, verbose=0)
    pb._update_stateful_metrics(["a", "b"])
    assert {"a", "b"}.issubset(pb.stateful_metrics)


def test_progbar_verbose1_non_list_value_branch(capsys):
    # The verbose-1 render has an else branch for when ``_values[k]`` is not a
    # list (it normally always is). Force a scalar entry to exercise it.
    pb = P.Progbar(target=2, verbose=1, interval=0.0)
    pb._values_order.append("custom")
    pb._values["custom"] = "raw_value"
    pb.update(1)
    out = capsys.readouterr().out
    assert "custom" in out
    assert "raw_value" in out


def test_progbar_value_base_accumulation():
    # The averaging accumulator: same key seen twice updates running sums.
    pb = P.Progbar(target=4, verbose=0)
    pb.update(1, values=[("loss", 2.0)])
    pb.update(3, values=[("loss", 4.0)])
    # _values[loss] = [sum, count]
    total, count = pb._values["loss"]
    assert count >= 1
    assert total >= 0


# --------------------------------------------------------------------------
# make_batches / slice_arrays / to_list
# --------------------------------------------------------------------------

def test_make_batches():
    assert P.make_batches(10, 3) == [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert P.make_batches(0, 3) == []


def test_slice_arrays_none():
    assert P.slice_arrays(None) == [None]


def test_slice_arrays_start_list_with_stop_raises():
    with pytest.raises(ValueError, match="stop argument has to be None"):
        P.slice_arrays([np.arange(5)], start=[0, 1], stop=3)


def test_slice_arrays_list_of_arrays_with_index_array():
    arrs = [np.arange(5), np.arange(5) * 2]
    idx = np.array([0, 2, 4])
    out = P.slice_arrays(arrs, start=idx)
    assert np.array_equal(out[0], [0, 2, 4])
    assert np.array_equal(out[1], [0, 4, 8])


def test_slice_arrays_list_with_none_entry_and_index_list():
    arrs = [np.arange(5), None]
    out = P.slice_arrays(arrs, start=[0, 1])
    assert np.array_equal(out[0], [0, 1])
    assert out[1] is None


def test_slice_arrays_list_with_slice():
    arrs = [np.arange(5), None, 123]
    out = P.slice_arrays(arrs, start=1, stop=3)
    assert np.array_equal(out[0], [1, 2])
    assert out[1] is None
    # 123 has no __getitem__ -> None
    assert out[2] is None


def test_slice_arrays_single_array_index_array():
    arr = np.arange(5)
    idx = np.array([1, 3])
    out = P.slice_arrays(arr, start=idx)
    assert np.array_equal(out, [1, 3])


def test_slice_arrays_single_array_index_list():
    arr = np.arange(5)
    out = P.slice_arrays(arr, start=[0, 2])
    assert np.array_equal(out, [0, 2])


def test_slice_arrays_single_array_slice():
    # NOTE: for a single (non-list) array, an integer `start` has no __len__
    # and no __getitem__, so this hits the final ``return [None]`` branch
    # rather than slicing. Documents current behavior.
    arr = np.arange(5)
    out = P.slice_arrays(arr, start=1, stop=3)
    assert out == [None]


def test_slice_arrays_single_array_subscriptable_start():
    # A start object that has __getitem__ but no __len__ exercises the
    # ``return arrays[start:stop]`` branch on the single-array path. We give
    # `start` an __index__ so the slice is valid, and a custom `arrays`
    # whose __getitem__ records the slice it received.
    start_key = type("StartKey", (), {"__getitem__": lambda self, k: k})()
    assert hasattr(start_key, "__getitem__")
    assert not hasattr(start_key, "__len__")

    class Arr:
        def __getitem__(self, sl):
            return ("sliced", sl.start, sl.stop)

    out = P.slice_arrays(Arr(), start=start_key, stop=4)
    # The single-array branch performs ``arrays[start:stop]``.
    assert out[0] == "sliced"
    assert out[1] is start_key
    assert out[2] == 4


def test_slice_arrays_single_non_subscriptable_returns_none_list():
    # start is None and no __getitem__ on arrays -> [None]
    assert P.slice_arrays(123) == [None]


def test_to_list():
    assert P.to_list([1, 2]) == [1, 2]
    assert P.to_list(5) == [5]


# --------------------------------------------------------------------------
# to_snake_case
# --------------------------------------------------------------------------

def test_to_snake_case_public():
    assert P.to_snake_case("MyClassName") == "my_class_name"
    assert P.to_snake_case("HTTPServer") == "http_server"


def test_to_snake_case_private_prefix():
    # name starting with "_" gets a "private" prefix
    assert P.to_snake_case("_Hidden").startswith("private")


# --------------------------------------------------------------------------
# check_for_unexpected_keys / validate_kwargs
# --------------------------------------------------------------------------

def test_check_for_unexpected_keys_ok():
    P.check_for_unexpected_keys("cfg", {"a": 1}, ["a", "b"])  # no raise


def test_check_for_unexpected_keys_raises():
    with pytest.raises(ValueError, match="Unknown entries"):
        P.check_for_unexpected_keys("cfg", {"x": 1}, ["a"])


def test_validate_kwargs_ok():
    P.validate_kwargs({"a": 1}, {"a", "b"})  # no raise


def test_validate_kwargs_raises():
    with pytest.raises(TypeError):
        P.validate_kwargs({"z": 1}, {"a"})


# --------------------------------------------------------------------------
# default / is_default
# --------------------------------------------------------------------------

def test_default_and_is_default():
    @P.default
    def m():
        return 1

    assert P.is_default(m) is True

    def n():
        return 2

    assert P.is_default(n) is False


# --------------------------------------------------------------------------
# populate_dict_with_module_objects
# --------------------------------------------------------------------------

def test_populate_dict_with_module_objects():
    mod = types.ModuleType("fake_mod")
    mod.keep_me = lambda: 1
    mod.skip_me = 42
    target = {}
    P.populate_dict_with_module_objects(target, [mod], callable)
    assert "keep_me" in target
    assert "skip_me" not in target


# --------------------------------------------------------------------------
# LazyLoader
# --------------------------------------------------------------------------

def test_lazy_loader_loads_on_attr_access():
    g = {}
    loader = P.LazyLoader("json_alias", g, "json")
    # Attribute access triggers the import + parent globals injection.
    dumps = loader.dumps
    assert callable(dumps)
    assert dumps([1, 2]) == "[1, 2]"
    # parent globals now contain the real module
    assert "json_alias" in g
    assert g["json_alias"].__name__ == "json"


# --------------------------------------------------------------------------
# print_msg
# --------------------------------------------------------------------------

def test_print_msg_line_break(capsys):
    P.print_msg("hello")
    out = capsys.readouterr().out
    assert out == "hello\n"


def test_print_msg_no_line_break(capsys):
    P.print_msg("hello", line_break=False)
    out = capsys.readouterr().out
    assert out == "hello"
