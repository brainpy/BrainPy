# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Line-coverage tests for ``brainpy/running/runner.py``.

Exercises the base :class:`~brainpy.running.runner.Runner` construction logic:

- ``jit`` argument normalisation (bool / dict / invalid).
- Sequence-form monitor parsing (str, (str,), (str, int), (str, list),
  (str, None)) and target resolution (own attribute, nested attribute,
  ``name2node`` lookup).
- Dict-form monitor parsing (Variable, str, (var, index), callable, len-1
  tuple) and target resolution.
- Deprecated ``fun_monitors`` merge path.
- Every ``MonitorError`` / ``RunningError`` / ``ValueError`` branch in the
  monitor/jit parsing helpers.
- ``__del__`` cleanup.

The target model is a tiny hand-rolled :class:`~brainpy.BrainPyObject` so the
tests stay fast and dependency-free.
"""

import warnings

import numpy as np
import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy._errors import MonitorError, RunningError
from brainpy.running import constants as C
from brainpy.running.runner import Runner


class _Sub(bp.BrainPyObject):
    """A leaf node carrying a single monitorable variable."""

    def __init__(self):
        super().__init__()
        self.V = bm.Variable(bm.zeros(3))


class _KeyErrorNode(bp.BrainPyObject):
    """A node whose attribute access raises ``KeyError`` on a missing name.

    Used to drive the ``except KeyError`` branch of the multi-segment
    monitor-path resolver in ``_find_seq_monitor_targets``.
    """

    def __init__(self):
        super().__init__()
        self.V = bm.Variable(bm.zeros(3))

    def __getattr__(self, item):
        if item.startswith('__'):
            raise AttributeError(item)
        raise KeyError(item)


class _MidParent(bp.BrainPyObject):
    """A parent whose child raises ``KeyError`` for unknown attributes."""

    def __init__(self):
        super().__init__()
        self.mid = _KeyErrorNode()


class _Target(bp.BrainPyObject):
    """A small target model with a nested child and own variables."""

    def __init__(self):
        super().__init__()
        self.sub = _Sub()
        self.V = bm.Variable(bm.zeros(5))
        self.spike = bm.Variable(bm.zeros(5))


@pytest.fixture()
def target():
    return _Target()


# --------------------------------------------------------------------------- #
# jit normalisation
# --------------------------------------------------------------------------- #

def test_jit_bool(target):
    r = Runner(target, monitors=None, jit=True, progress_bar=False)
    assert r.jit == {C.PREDICT_PHASE: True}


def test_jit_false(target):
    r = Runner(target, monitors=None, jit=False, progress_bar=False)
    assert r.jit == {C.PREDICT_PHASE: False}


def test_jit_dict_fills_predict_default(target):
    # dict without an explicit predict phase -> predict defaults to True
    r = Runner(target, monitors=None, jit={'train': False}, progress_bar=False)
    assert r.jit['train'] is False
    assert r.jit[C.PREDICT_PHASE] is True


def test_jit_dict_with_predict(target):
    r = Runner(target, monitors=None, jit={C.PREDICT_PHASE: False}, progress_bar=False)
    assert r.jit[C.PREDICT_PHASE] is False


def test_jit_invalid_type_raises(target):
    with pytest.raises(ValueError):
        Runner(target, monitors=None, jit='not-a-jit', progress_bar=False)


# --------------------------------------------------------------------------- #
# default / None monitors
# --------------------------------------------------------------------------- #

def test_monitors_none(target):
    r = Runner(target, monitors=None, progress_bar=False, jit=False)
    assert r._monitors == {}
    assert isinstance(r.mon, bp.tools.DotDict)
    assert r.progress_bar is False
    assert r._pbar is None
    assert r.numpy_mon_after_run is True


def test_monitors_invalid_type_raises(target):
    with pytest.raises(MonitorError):
        Runner(target, monitors=42, progress_bar=False, jit=False)


# --------------------------------------------------------------------------- #
# sequence monitors (list / tuple) - parsing + target resolution
# --------------------------------------------------------------------------- #

def test_seq_monitor_all_forms(target):
    r = Runner(
        target,
        monitors=['V',                 # bare str
                  ('spike',),           # len-1 tuple
                  ('V', 1),             # (str, int) -> wrapped index array
                  ('spike', [1, 2, 3]),  # (str, list) -> asarray index
                  ('V', None)],         # (str, None) -> no index
        progress_bar=False,
        jit=False,
    )
    # all monitor keys are present and resolve to (variable, index) tuples
    assert set(r._monitors.keys()) == {'V', 'spike'}
    for key, (var, idx) in r._monitors.items():
        assert isinstance(var, bm.Variable)


def test_seq_monitor_int_index_wrapped_to_array(target):
    r = Runner(target, monitors=[('V', 2)], progress_bar=False, jit=False)
    _, idx = r._monitors['V']
    assert np.asarray(bm.as_jax(idx)).tolist() == [2]


def test_seq_monitor_nested_attribute(target):
    r = Runner(target, monitors=['sub.V'], progress_bar=False, jit=False)
    var, _ = r._monitors['sub.V']
    assert var is target.sub.V


def test_seq_monitor_name2node_resolution(target):
    # a dotted key whose head is not a direct attribute but the *name* of a node
    sub_name = target.sub.name
    r = Runner(target, monitors=[f'{sub_name}.V'], progress_bar=False, jit=False)
    var, _ = r._monitors[f'{sub_name}.V']
    assert var is target.sub.V


def test_seq_monitor_missing_simple_var_raises(target):
    with pytest.raises(RunningError):
        Runner(target, monitors=['does_not_exist'], progress_bar=False, jit=False)


def test_seq_monitor_missing_node_raises(target):
    with pytest.raises(MonitorError):
        Runner(target, monitors=['ghost.V'], progress_bar=False, jit=False)


def test_seq_monitor_keyerror_midpath_raises():
    # 'mid' is a real attribute, so resolution walks splits[:-1]; the second
    # segment access raises KeyError -> wrapped as MonitorError.
    parent = _MidParent()
    with pytest.raises(MonitorError):
        Runner(parent, monitors=['mid.missing.V'], progress_bar=False, jit=False)


def test_seq_monitor_len3_tuple_raises(target):
    with pytest.raises(MonitorError):
        Runner(target, monitors=[('V', 1, 2)], progress_bar=False, jit=False)


def test_seq_monitor_nonstr_first_elem_raises(target):
    with pytest.raises(MonitorError):
        Runner(target, monitors=[(1, 2)], progress_bar=False, jit=False)


def test_seq_monitor_bad_element_type_raises(target):
    with pytest.raises(MonitorError):
        Runner(target, monitors=[42], progress_bar=False, jit=False)


# --------------------------------------------------------------------------- #
# dict monitors - parsing + target resolution
# --------------------------------------------------------------------------- #

def test_dict_monitor_variable_value(target):
    r = Runner(target, monitors={'a': target.V}, progress_bar=False, jit=False)
    var, idx = r._monitors['a']
    assert var is target.V and idx is None


def test_dict_monitor_str_value_not_resolved():
    # NOTE: DEFECT - dict-form *string* monitors are NOT resolved to their
    # target Variable. ``_format_dict_monitors`` wraps a string value 'V' into
    # the tuple ('V', None); by the time it reaches
    # ``_find_dict_monitor_targets`` the value is a tuple, so the
    # ``isinstance(_mon, str)`` resolution branch (runner.py lines ~241-266) is
    # never taken and the value falls through to the ``else`` branch which
    # stores it verbatim. The recorded "variable" is therefore the literal
    # string 'V', not ``target.V``. (Sequence-form monitors resolve correctly.)
    target = _Target()
    r = Runner(target, monitors={'a': 'V'}, progress_bar=False, jit=False)
    var, idx = r._monitors['a']
    assert var == 'V'  # the bug: a string, not the Variable
    assert idx is None


def test_dict_monitor_str_value_nested_not_resolved():
    # NOTE: DEFECT (same root cause as above) - a dotted string value is also
    # stored verbatim and never resolved to ``target.sub.V``.
    target = _Target()
    r = Runner(target, monitors={'a': 'sub.V'}, progress_bar=False, jit=False)
    var, idx = r._monitors['a']
    assert var == 'sub.V'
    assert idx is None


def test_dict_monitor_var_index_tuple(target):
    r = Runner(target, monitors={'a': (target.spike, 0)}, progress_bar=False, jit=False)
    _, idx = r._monitors['a']
    assert np.asarray(bm.as_jax(idx)).tolist() == [0]


def test_dict_monitor_var_list_index(target):
    r = Runner(target, monitors={'a': (target.V, [1, 2])}, progress_bar=False, jit=False)
    _, idx = r._monitors['a']
    assert np.asarray(bm.as_jax(idx)).tolist() == [1, 2]


def test_dict_monitor_len1_tuple(target):
    r = Runner(target, monitors={'a': (target.V,)}, progress_bar=False, jit=False)
    var, idx = r._monitors['a']
    assert var is target.V and idx is None


def test_dict_monitor_callable_value(target):
    fn = lambda tdi: target.V[:2]
    r = Runner(target, monitors={'a': fn}, progress_bar=False, jit=False)
    assert r._monitors['a'] is fn


def test_dict_monitor_str_missing_var_not_validated():
    # NOTE: DEFECT (same root cause) - because dict string monitors are never
    # resolved, an *invalid* variable name like 'nope' is silently accepted and
    # stored verbatim instead of raising RunningError (contrast with the
    # sequence-form which validates and raises). See
    # ``test_dict_monitor_str_value_not_resolved``.
    target = _Target()
    r = Runner(target, monitors={'a': 'nope'}, progress_bar=False, jit=False)
    assert r._monitors['a'] == ('nope', None)


def test_dict_monitor_nonstr_key_raises(target):
    with pytest.raises(MonitorError):
        Runner(target, monitors={1: 'V'}, progress_bar=False, jit=False)


def test_dict_monitor_bad_value_type_raises(target):
    with pytest.raises(MonitorError):
        Runner(target, monitors={'a': 42}, progress_bar=False, jit=False)


def test_dict_monitor_len3_tuple_raises(target):
    with pytest.raises(MonitorError):
        Runner(target, monitors={'a': (target.V, 1, 2)}, progress_bar=False, jit=False)


def test_dict_monitor_tuple_bad_first_elem_raises(target):
    with pytest.raises(MonitorError):
        Runner(target, monitors={'a': (42, 1)}, progress_bar=False, jit=False)


# --------------------------------------------------------------------------- #
# deprecated fun_monitors
# --------------------------------------------------------------------------- #

def test_fun_monitors_deprecated_merge(target):
    def fm(t, dt):
        return target.V

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        r = Runner(target, monitors={'a': target.V}, fun_monitors={'fm': fm},
                   progress_bar=False, jit=False)
    assert r._monitors['fm'] is fm
    assert any(issubclass(w.category, UserWarning) for w in caught)


# --------------------------------------------------------------------------- #
# direct helper type guards (defensive branches)
# --------------------------------------------------------------------------- #

def test_format_seq_monitors_type_guard(target):
    r = Runner(target, monitors=None, progress_bar=False, jit=False)
    with pytest.raises(TypeError):
        r._format_seq_monitors({'not': 'a-seq'})


def test_format_dict_monitors_type_guard(target):
    r = Runner(target, monitors=None, progress_bar=False, jit=False)
    with pytest.raises(TypeError):
        r._format_dict_monitors(['not', 'a', 'dict'])


def test_find_seq_monitor_targets_type_guard(target):
    r = Runner(target, monitors=None, progress_bar=False, jit=False)
    with pytest.raises(TypeError):
        r._find_seq_monitor_targets({'not': 'a-seq'})


def test_find_dict_monitor_targets_type_guard(target):
    r = Runner(target, monitors=None, progress_bar=False, jit=False)
    with pytest.raises(TypeError):
        r._find_dict_monitor_targets(['not', 'a', 'dict'])


def test_find_dict_monitor_targets_bare_string_branch():
    # NOTE: DEFECT - the ``isinstance(_mon, str)`` branch in
    # ``_find_dict_monitor_targets`` is dead under normal use (see
    # ``test_dict_monitor_str_value_not_resolved``) AND broken: it does
    # ``key, index = _mon[0], _mon[1]`` which takes the first *two characters*
    # of the string rather than treating the whole string as the variable name.
    # We call the helper directly to exercise this branch. With the bare string
    # 'Vx', key='V' (a real single-char variable) and index='x'.
    target = _Target()
    r = Runner(target, monitors=None, progress_bar=False, jit=False)
    out = r._find_dict_monitor_targets({'k': 'Vx'})
    var, index = out['V']  # NOTE: keyed by 'V' (first char), not 'k'
    assert var is target.V
    assert index == 'x'  # NOTE: second char taken as the "index"


def test_find_dict_monitor_targets_bare_string_missing_var_raises():
    # NOTE: DEFECT - same broken branch: a string whose first character is not
    # an attribute of the target raises RunningError.
    target = _Target()
    r = Runner(target, monitors=None, progress_bar=False, jit=False)
    with pytest.raises(RunningError):
        r._find_dict_monitor_targets({'k': 'zz'})


# --------------------------------------------------------------------------- #
# __del__ cleanup
# --------------------------------------------------------------------------- #

def test_del_cleans_up(target):
    r = Runner(target, monitors=['V'], progress_bar=False, jit=False)
    r.mon['extra'] = bm.zeros(2)
    # should not raise; clears mon dict and instance attributes
    r.__del__()
    assert not hasattr(r, 'target')
