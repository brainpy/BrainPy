# Audit findings — top-level glue (P14) — 2026-06-19

Scope: brainpy/{dynsys,runners,delay,mixin,transform,context,check,measure,helpers,
checkpoints,_errors,deprecations,types,visualization,channels,neurons,synapses,
synouts,synplast,layers,rates}.py (+ co-located *_test.py).

Branch: fix/audit-20260619-toplevel-glue

Counts: Critical 1, High 2, Medium 1, Low 4. Fixed 4 (C1, H1, H2 + the M2 test-fragility);
recorded-only 4 (M2 source-level, L1, L2, L3, L4).

| ID | Severity | File | Status |
|----|----------|------|--------|
| P14-C1 | Critical | runners.py:661 | fixed |
| P14-H1 | High | runners.py:496 | fixed |
| P14-H2 | High | check.py:610,629 | fixed |
| P14-M2 | Medium | runners.py:628 | recorded (source); test fragility fixed |
| P14-L4 | Low | measure.py:88 | recorded-only |
| P14-L1 | Low | dynsys.py:579 | recorded-only |
| P14-L2 | Low | mixin.py:258 | recorded-only |
| P14-L3 | Low | check.py:87 | recorded-only |

---

### P14-C1 — `DSRunner(memory_efficient=True)` returns `None` instead of the model outputs  [Critical]
- File: brainpy/runners.py:661-668 (`_fun_predict`, memory-efficient branch)
- Category: correctness
- What: The per-step output accumulation is doubly broken.
  ```python
  outs = None
  for i in range(indices.shape[0]):
      out, _ = run_fun(indices[i], *tree_map(lambda a: a[i], inputs))
      if outs is None:
          outs = tree_map(lambda a: [], out)             # (1) empty list is NOT a leaf
      outs = tree_map(lambda a, o: o.append(a), out, outs)  # (2) list.append() returns None
  outs = tree_map(lambda a: bm.as_jax(a), outs)
  ```
  (1) `tree_map(lambda a: [], out)` maps each leaf to an empty list `[]`; but jax treats `[]`
  as an empty *container node*, not a leaf, so `outs` ends up as an empty pytree.
  (2) `o.append(a)` returns `None`, so `tree_map(..., out, outs)` rebinds `outs` to a tree of
  `None` on every iteration. The final return value of `.run()/.predict()` is `None`/garbage.
- Why it's a bug: A `memory_efficient=True` run silently returns the wrong output (`None`) for
  the model's `update()` return value in the default, documented usage. Monitors happen to be
  collected separately via `jax.debug.callback`, so the existing tests (which only check
  `runner.mon[...]`) never caught it. The standard (`memory_efficient=False`) path returns the
  outputs stacked along a leading time axis; the two paths must agree.
- Repro:
  ```python
  class DS(bp.DynamicalSystem):
      def __init__(self):
          super().__init__(); self.i = bm.Variable(bm.zeros(1))
      def update(self):
          self.i += 1.; return self.i.value
  out_n = bp.DSRunner(DS(), dt=1., progress_bar=False, memory_efficient=False).run(5.)  # [1,2,3,4,5]
  out_m = bp.DSRunner(DS(), dt=1., progress_bar=False, memory_efficient=True).run(5.)   # [None]  <-- BUG
  ```
- Fix: Accumulate per-step outputs in a flat Python list and stack along a new leading axis
  (matching `bm.for_loop`'s time-major stacking):
  ```python
  outs = []
  for i in range(indices.shape[0]):
      out, _ = run_fun(indices[i], *tree_map(lambda a: a[i], inputs))
      outs.append(out)
  if len(outs) == 0:
      return None, None
  outs = tree_map(lambda *os: bm.as_jax(jnp.stack([bm.as_jax(o) for o in os])), *outs)
  return outs, None
  ```
- Tests: dyn_runner_test.py::TestMemoryEfficient::test_output_matches_normal_scalar,
  test_output_matches_normal_pytree, test_output_empty (in `dyn_runner_test.py`).
- Status: fixed

---

### P14-H1 — `DSRunner(memory_efficient=True)` returns monitors with a different axis order than the standard path (batching mode)  [High]
- File: brainpy/runners.py:496-499 (`predict`, memory-efficient monitor finalisation)
- Category: correctness
- What: The memory-efficient path appends per-step monitor values on the host, so they are
  always stacked time-major ``(T, ...)``. The standard path re-orders monitors (and outputs) to
  batch-major ``(B, T, ...)`` for a ``BatchingMode`` target with ``data_first_axis='B'`` (see
  ``_predict`` :542-544). The memory-efficient finalisation did `self.mon[key] = np.asarray(...)`
  with **no** re-ordering, so for batched models (where `data_first_axis` defaults to `'B'`) the
  monitor layout silently depends on the `memory_efficient` flag.
- Why it's a bug: `memory_efficient` is documented as a memory/perf toggle; it must not change
  the shape/orientation of the returned monitors. A user toggling it gets `(T, B, F)` vs
  `(B, T, F)` monitors — silently wrong indexing downstream. The model *outputs* were already
  re-ordered correctly by `_predict`, so outputs and monitors disagreed.
- Repro:
  ```python
  class Net(bp.DynamicalSystem):
      def __init__(self):
          super().__init__(mode=bm.BatchingMode(4))
          self.n = bp.dyn.LifRef(3, mode=bm.BatchingMode(4))
      def update(self, inp):
          self.n(inp); return self.n.V.value
  inp = bm.ones((4, 8, 3)) * 2.0   # (B, T, F), data_first_axis defaults to 'B'
  r_n = bp.DSRunner(Net(), monitors=['n.V'], memory_efficient=False, progress_bar=False); r_n.run(inputs=inp)
  r_m = bp.DSRunner(Net(), monitors=['n.V'], memory_efficient=True,  progress_bar=False); r_m.run(inputs=inp)
  r_n.mon['n.V'].shape  # (4, 8, 3)
  r_m.mon['n.V'].shape  # (8, 4, 3)   <-- BUG, time/batch swapped
  ```
- Fix: in the memory-efficient monitor finalisation, when the target is in `BatchingMode` and
  `data_first_axis == 'B'`, `np.moveaxis(arr, 0, 1)` each monitor (ndim >= 2) to make it
  batch-major, mirroring `_predict`.
- Tests: dyn_runner_test.py::TestMemoryEfficient::test_batched_monitor_axis_matches_normal
- Status: fixed

---

### P14-H2 — `is_float`/`is_integer` `min_bound`/`max_bound` checks never raise (eager validation is a silent no-op)  [High]
- File: brainpy/check.py:629-649 (`jit_error_checking_no_args`); also `jit_error` :610-623
- Category: edge/error
- What: `is_float`/`is_integer` route their bound checks through
  `jit_error_checking_no_args(value < min_bound, ValueError(...))`, which used
  `jax.lax.cond(..., lambda: jax.pure_callback(true_err_fun, None), lambda: None)`. For a
  *concrete* (eager) predicate, `jax.pure_callback` only raises when the staged computation is
  executed — under an eager `cond` the callback's exception is never surfaced synchronously, so
  the function returns normally. (`true_err_fun(arg, transforms)` also had the wrong arity for a
  no-operand callback.) Result: every eager out-of-bound check is a no-op.
- Why it's a bug: parameter validation across the codebase silently accepts invalid values, e.g.
  `bp.dnn.Dropout(prob=2.0)` / `prob=-1`, negative frequencies, non-positive integer sizes,
  `gamma`/`weight_decay` outside `[0,1]`, etc. (28+ call sites use these bounds). Constructors
  that are documented to reject bad input quietly accept it, deferring failures to confusing
  downstream errors or wrong results.
- Repro:
  ```python
  bp.check.is_float(2.0, 'x', max_bound=1.0)   # returns 2.0 instead of raising
  bp.check.is_integer(0, 'n', min_bound=1)     # returns 0 instead of raising
  ```
- Fix: in both `jit_error_checking_no_args` and `jit_error`, add a concrete-predicate fast path:
  if `pred` is not a `jax.core.Tracer`, evaluate `bool(np.asarray(pred))` and raise/call the
  error directly; otherwise keep the deferred `cond`+`pure_callback` path for in-jit signalling.
  Also fixed `true_err_fun` to accept `*args`.
- Tests: check_test.py::TestBoundChecks (9 tests: eager raise for min/max on float & int,
  within-bounds OK, concrete-True/False, and no-raise-at-trace under jit). Updated the
  pre-existing check_coverage_test.py tests that *pinned the buggy no-raise behavior*
  (TestIsFloat/TestIsInteger min/max bound branches; TestJitErrors pred_true / jit_error_true /
  jit_error_true_tuple_arg) to assert the corrected raising semantics.
- Status: fixed

---

### P14-L4 — `firing_rate(..., numpy=False)` JIT claim holds only for static window length  [Low]
- File: brainpy/measure.py:88-96
- Category: edge/error
- What: The docstring promises `numpy=False` makes `firing_rate` JIT-compilable
  ("If ``False``, this function can be JIT compiled."). With `numpy=False` the function uses
  `jnp.convolve`, which is fine, but `width1 = int(width/2/dt)*2+1` requires `width`/`dt` to be
  concrete python numbers — that is the normal case and works. This is NOT a correctness bug;
  the H-43 normalization fix is already present (line 95) and verified. Recorded as Low/no-fix
  observation only that the JIT claim holds only when window length is static.
- Why it's a bug: documentation nuance, not a functional defect.
- Repro: static
- Fix: recorded only
- Tests: none
- Status: recorded-only

---

### P14-M2 — `DSRunner` permanently mutates the global `dt` (state leak)  [Medium]
- File: brainpy/runners.py:628 (`_step_func_predict` → `share.save(..., dt=self.dt)`)
- Category: edge/error
- What: Every `DSRunner.run()` writes `dt` into the *global* brainstate environment via
  `share.save(dt=self.dt)` and never restores it. So `bp.DSRunner(model, dt=1.).run(...)` leaves
  the process-wide `bm.get_dt()` permanently at `1.0`, silently changing the behavior of any
  subsequently-created object that reads the default `dt` (e.g. `VarDelay(v, time=0.5)` then
  computes `int(0.5/1.0) == 0` steps instead of `5`).
- Why it's a bug: A per-runner `dt` should be scoped to that runner, not bleed into global state.
  This is a real cross-object coupling that surfaced as a test-ordering fragility (a delay test
  asserting `max_length == 5` failed after a `dt=1.` runner ran earlier in the same process).
- Repro:
  ```python
  bm.get_dt()                                   # 0.1
  bp.DSRunner(model, dt=1., progress_bar=False).run(5.)
  bm.get_dt()                                   # 1.0   <-- leaked
  ```
- Fix: Source-level fix is out of strict scope (changing the global-`dt` contract risks breaking
  the many call sites that intentionally rely on `DSRunner(dt=...)` making `share['dt']` visible
  in `update()`), so the runner behavior is **left as-is and recorded**. Fixed the resulting
  *test* fragility in scope: `dyn_runner_test.py` now snapshots the default `dt` at import and
  restores it in `tearDown` (`_DtRestoreMixin`) so the suite is order-independent.
- Tests: dyn_runner_test.py `_DtRestoreMixin` (TestDSRunner, TestMemoryEfficient teardown).
- Status: recorded-only (source); test fragility fixed

---

### P14-L1 — `DynSysGroup.update` recomputes the full node collection every step  [Low]
- File: brainpy/dynsys.py:579-591
- Category: perf
- What: `self.nodes(level=1, ...).subset(...).unique().not_subset(DynView)` is recomputed on
  every `update()` call (and again `.subset(Projection)`, `.subset(Dynamic)`, etc.). Under a
  `for_loop`/`jit` scan this is traced once, so runtime cost is amortized, but eager
  (`jit=False`) runs pay it every step.
- Why it's a bug: perf-only; correctness unaffected.
- Fix: recorded only (would require caching node partitions, which risks staleness when children
  are mutated; out of proportion to benefit).
- Status: recorded-only

---

### P14-L2 — `Container.__getattr__` can mask real `AttributeError`s with confusing messages  [Low]
- File: brainpy/mixin.py:258-267
- Category: edge/error
- What: `__getattr__` falls back to `super().__getattribute__(item)` for non-child attributes,
  which re-raises `AttributeError` but loses the original context when the missing attribute is
  computed lazily.
- Why it's a bug: cosmetic / debuggability only.
- Fix: recorded only
- Status: recorded-only

---

### P14-L3 — `check.is_shape_consistency` asserts on the wrong variable in its loop  [Low]
- File: brainpy/check.py:87-89, 106-108
- Category: style
- What: Inside `for shape in shapes:` the assertion checks `isinstance(shapes, (tuple, list))`
  (the outer container) rather than `shape` (the element). The intended per-element type check
  is effectively a no-op. Harmless because the outer check already ran, but dead/misleading.
- Why it's a bug: dead assertion; no behavioral impact.
- Fix: recorded only
- Status: recorded-only

---

## Cross-check against dev/issues-found-20260618.md (top-level entries)

- C-22 (`DSRunner(memory_efficient=True)` non-functional, `'dict' has no attribute 'shape'`):
  the monitor-callback crash is **already fixed** in the current code (jax.debug.callback path,
  runners.py:640-650). However the *output accumulation* in the same path was still broken —
  captured as **P14-C1** above and fixed here.
- H-43 (`measure.firing_rate` normalization): **already fixed** (measure.py:95). Verified
  (constant 100 Hz spike train → mean rate 100.0).
- H-44 (`VarDelay(target, time=T>0)` reads `self.data` before assignment): **already fixed**
  (delay.py:254-258, `self.data = None` set unconditionally). Verified.
- H-45 (`DataDelay.reset_state(batch_size)` → `size_without_batch` TypeError): root cause is in
  `math/object_transform/variables.py` (out of scope); **already fixed** there
  (`size_without_batch` now returns `(10,)` for a `(4,10)` batched Variable). Verified.

left_unfixed_chm: none in scope. H-45's fix lives outside this scope (variables.py) and is
already applied upstream.
