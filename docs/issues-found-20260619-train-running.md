# Audit — brainpy/train + brainpy/running (2026-06-19)

Reviewer slice **P15**. Scope: `brainpy/train/{_utils,back_propagation,base,offline,online}.py`,
`brainpy/running/{constants,jax_multiprocessing,native_multiprocessing,pathos_multiprocessing,runner}.py`.

Severity: Critical = silently wrong / crash in default usage; High = wrong in realistic
cases / broken public API; Medium = edge/fragility/perf; Low = style/docs (recorded only).

---

### P15-H1 — `Runner.__init__` mutates the caller's `jit` dict and corrupts subclass jit config  [High]
- File: brainpy/running/runner.py:101
- Category: correctness / api-drift
- What: When `jit` is a dict, `self._origin_jit = jit` stores a *reference* to the
  caller's dict, then `self.jit[C.PREDICT_PHASE] = jit.pop(C.PREDICT_PHASE, True)`
  mutates that same dict in place, removing the `'predict'` key.
- Why it's a bug: Two failures.
  (1) The caller's dict is silently mutated (surprising side effect; M-32 in the
      2026-06-18 audit).
  (2) More seriously: subclasses (`DSTrainer`, `BPTrainer`) read
      `self._origin_jit.get(c.PREDICT_PHASE, True)` *after* `Runner.__init__` has
      popped the key. So a user who passes `jit={'predict': False, 'fit': True}`
      gets `self.jit['predict'] = True` (the default), the opposite of what was
      requested — predict is JIT-compiled when the user explicitly disabled it.
- Repro:
  ```python
  import brainpy as bp, brainpy.math as bm
  d = {'predict': False, 'fit': True}
  net = ... # any BatchingMode DynamicalSystem
  tr = bp.BPTT(net, loss_fun='mean_squared_error', jit=d)
  assert 'predict' in d          # FAILS: key was popped
  assert tr.jit['predict'] is False  # FAILS: reads True
  ```
- Fix: operate on a copy: `jit = dict(jit)` before popping; build `self.jit` from the
  copy so `self._origin_jit` (still the original) keeps its `'predict'` key for the
  subclasses to read.
- Tests: test_jit_dict_does_not_mutate_caller, test_jit_dict_predict_false_respected_by_subclass (runner_coverage_test.py)
- Status: fixed

---

### P15-H2 — dict-form string monitors are never resolved to their Variable (silent wrong monitor + no validation)  [High]
- File: brainpy/running/runner.py:240-269 (`_find_dict_monitor_targets`), :178 (`_format_dict_monitors`)
- Category: correctness
- What: `_format_dict_monitors` wraps a string monitor value `'V'` into the tuple
  `('V', None)`. By the time it reaches `_find_dict_monitor_targets` the value is a
  *tuple*, so the `isinstance(_mon, str)` resolution branch is never entered and the
  value falls through to `else: monitors[_key] = _mon`, storing the unresolved
  `('V', None)`. The stored "variable" is the literal string `'V'`, not `target.V`.
  An invalid name (`'nope'`) is also accepted without validation.
- Why it's a bug: `monitors={'a': 'V'}` is documented public API
  (Runner docstring: "A dict with the explicit monitor target"). At run time
  `_step_func_monitor` unpacks `(variable, idx) = ('V', None)` and evaluates
  `'V'.value` → `AttributeError`, or stores garbage. The sequence form
  (`monitors=['V']`) resolves correctly, so the two paths silently disagree.
  Two existing tests (`test_dict_monitor_str_value_not_resolved`,
  `test_dict_monitor_str_missing_var_not_validated`) explicitly document this as a
  DEFECT.
- Repro:
  ```python
  r = Runner(target, monitors={'a': 'V'}, jit=False, progress_bar=False)
  var, idx = r._monitors['a']
  assert var is target.V   # FAILS: var == 'V' (a string)
  ```
- Fix: in `_find_dict_monitor_targets`, take the resolution branch when the value is
  a `(name_str, index)` tuple (i.e. `isinstance(_mon, (tuple, list)) and
  isinstance(_mon[0], str)`), resolving the dotted name exactly like the sequence
  resolver, and validating unknown names (raises `RunningError`/`MonitorError`).
  Variables/callables/(Variable, idx) tuples still pass through unchanged.
  Additionally: the resolution branch keyed the result by the *variable name*
  (`monitors[key]`) instead of the user-chosen monitor key (`monitors[_key]`), so
  even if it had fired, `runner.mon['a']` would have been stored under `'V'`.
  Fixed to key by `_key`.
- Tests: test_dict_monitor_str_value_resolved, test_dict_monitor_str_value_nested_resolved,
  test_dict_monitor_str_missing_var_raises, test_dict_monitor_str_with_index_resolved
  (runner_coverage_test.py); updated the two prior DEFECT tests.
- Status: fixed

---

### P15-H3 — `jax_parallelize_map` crashes concatenating a trailing partial chunk across devices  [High]
- File: brainpy/running/jax_multiprocessing.py:139-160
- Category: correctness / perf
- What: With `n` devices and `num_parallel == n`, a task count not divisible by `n`
  produces a final chunk of size `< n`. `pmap` re-traces fine for the smaller chunk
  and shards its output on only the first `k` devices, but the closing
  `bm.concatenate(res, axis=0)` (the `clear_buffer=False` branch) then tries to
  concatenate arrays that live on *different device subsets* → JAX raises
  `ValueError: Received incompatible devices for jitted computation`.
- Why it's a bug: This is the documented multi-device use case
  ("set host device count by `brainpy.math.set_host_device_count(n)`"). Any
  `num_tasks % num_parallel != 0` crashes the run after all the compute is done.
- Repro (4 host devices):
  ```python
  # XLA_FLAGS=--xla_force_host_platform_device_count=4
  jax_parallelize_map(lambda x: x * 2.0, [np.arange(6.0)], num_parallel=4)
  # ValueError: Received incompatible devices for jitted computation
  ```
- Fix: gather each chunk's result to host (`jax.device_get`) before stacking, so the
  final concatenation operates on host arrays (no device-placement conflict), for both
  the `clear_buffer` and non-`clear_buffer` branches. Returns `bm.asarray` of the
  concatenation to preserve the JAX-array contract of the non-buffer branch.
- Tests: test_parallelize_map_partial_chunk (jax_multiprocessing_test.py, skipped if <2 devices),
  test_parallelize_map_single_device, test_vectorize_map_partial_chunk,
  test_vectorize_map_dict_args, test_map_length_mismatch_raises
- Status: fixed

---

### P15-M1 — `process_pool_lock` mutates the caller's parameter dicts with the lock  [Medium]
- File: brainpy/running/native_multiprocessing.py:110
- Category: edge/error
- What: For dict-form params, `net_params.update(lock=lock)` mutates the caller's
  dict in place, injecting a `Manager().Lock()` into it.
- Why it's a bug: The caller's `all_params` list is silently altered; re-running with
  the same params list now carries a stale, possibly cross-pool lock, and the dict
  now contains a non-picklable-in-some-contexts manager proxy the user never put
  there. Side-effecting a caller-owned container is a correctness/ergonomics trap.
- Repro (static): pass `all_params=[{'a': 1}]`; after the call the dict is
  `{'a': 1, 'lock': <lock>}`.
- Fix: build a shallow copy `{**net_params, 'lock': lock}` and submit that.
- Tests: test_process_pool_lock_does_not_mutate_caller_dict (native_multiprocessing_coverage_test.py)
- Status: fixed

---

### P15-M2 — BPTT loss uses unpinned `self.i0` for time indices; wrong absolute time when `reset_state=False`  [Medium]
- File: brainpy/train/back_propagation.py:522-523
- Category: correctness / edge
- What: `_step_func_loss` builds `indices = np.arange(self.i0, self.i0 + num_step)`.
  In the BPTT/BPFF fit loop, `i0` is reset to 0 by `reset_state()` only when
  `reset_state=True` (the default). With `reset_state=False` (continuing a stateful
  model across batches), `i0` is never advanced by the fit loop (`_predict` does not
  touch `i0`), so every batch re-uses indices starting at the same stale `i0`,
  giving a wrong/constant absolute `t`/`i` to time-dependent inputs and monitors.
- Why it's a bug: `reset_state=False` continuation is a realistic recurrent-training
  pattern. The absolute step index fed to `share['i']`/`share['t']` is then wrong.
- Repro: static (requires a model whose `update` reads `share['t']`).
- Fix: recorded only. A correct fix needs the fit loop to advance `i0` per batch
  (cross-cutting with `runners.py`, out of clean scope for `_step_func_loss` alone),
  and risks changing the default-path semantics. The grad/loss windows are
  self-consistent within a batch; only cross-batch `reset_state=False` continuation
  is affected. Left to a focused follow-up to avoid altering the common path.
- Tests: none
- Status: recorded-only

---

### P15-L1 — `jax_vectorize_map` builds `vmap_func` twice  [Low]
- File: brainpy/running/jax_multiprocessing.py:71-73
- Category: style/perf
- What: `vmap_func = vmap(func)` is created once before the loop, then inside the loop
  `run_f = vmap(func) if clear_buffer else vmap_func` rebuilds `vmap(func)` every
  iteration when `clear_buffer=True`. The eager pre-build at line 71 is wasted when
  `clear_buffer=True`, and the per-iteration rebuild is only needed because buffers
  are cleared between chunks.
- Why it's a bug: minor wasted tracing; not a correctness issue.
- Fix: recorded only (Low).
- Status: recorded-only

---

### P15-L2 — `BPTrainer` docstring typos / `loss_auto_run` undocumented  [Low]
- File: brainpy/train/back_propagation.py:50,66,90
- Category: style/docs
- What: "supervised trasks", "dyamical systems", `loss_auto_run` documented as
  "pass", duplicate inline comment "# loss auxiliary" on `loss_auto_run`.
- Fix: recorded only (Low).
- Status: recorded-only

---

### P15-L3 — `OfflineTrainer._fun_train` progress-bar callback ignores `progress_bar` toggling under jit  [Low]
- File: brainpy/train/offline.py:240-241
- Category: style/perf
- What: `jax.debug.callback(lambda *args: self._pbar.update(), ())` updates the bar
  once per train node. Minor: the lambda discards its args and closes over `self`.
- Fix: recorded only (Low).
- Status: recorded-only

---

## Summary
- Critical: 0
- High: 3 (P15-H1, P15-H2, P15-H3) — all fixed
- Medium: 2 (P15-M1 fixed; P15-M2 recorded-only, cross-cutting)
- Low: 3 (recorded only)
