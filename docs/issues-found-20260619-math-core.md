# BrainPy math-core fresh audit — 2026-06-19 (P2)

Scope: `brainpy/math/_utils.py`, `datatypes.py`, `defaults.py`, `environment.py`,
`modes.py`, `ndarray.py`, `scales.py`, `sharding.py`, `others.py`, `remove_vmap.py`
(+ co-located `*_test.py`).

Environment: jax 0.10.2, brainstate 0.5.1, brainunit 0.5.1, brainevent 0.1.0,
braintools 0.1.10 (CPU-only). `import brainpy` works, so findings tagged
`[verified]` were reproduced at runtime.

This is a fresh pass. The fixes recorded in `dev/issues-found-20260618.md`
(C-10, M-07, M-08, M-09, M-10, H-10, H-11, H-12, H-14, H-15, L-02, L-03) are all
present in the current tree and were re-verified as correct; they are **not**
re-reported here. The findings below are new.

## Summary counts
- Critical: 0
- High: 1
- Medium: 1
- Low: 4
- Fixed: 2 (the High + the Medium)
- Recorded-only: 4 (all Low)

---

### P2-H1 — `ShardedArray` pytree round-trip drops `_keep_sharding` → `AttributeError` under every JAX transform  [High]
- File: brainpy/math/ndarray.py:228-288 (root cause: inherited `Array.tree_unflatten` at :110-119; `_keep_sharding` introduced at :242)
- Category: correctness / api-drift
- What: `ShardedArray` adds the slot `_keep_sharding` (set only in `__init__`) but
  reuses the base `Array.tree_flatten`/`tree_unflatten`. `tree_flatten` returns
  `aux_data=None` and `tree_unflatten` reconstructs via `object.__new__(cls)`
  setting only `_value`. So after any pytree round-trip the reconstructed
  `ShardedArray` has no `_keep_sharding` attribute, and its `value` getter
  (which reads `self._keep_sharding`) raises
  `AttributeError: 'ShardedArray' object has no attribute '_keep_sharding'`.
- Why it's a bug: JAX flattens/unflattens pytree leaves on essentially every
  transform boundary (`jit`, `vmap`, `scan`/`for_loop`, `grad`, `tree_map`,
  `eval_shape`). `ShardedArray` is a registered pytree node and is the wrapper
  `brainpy.math.sharding._device_put` returns (so `partition`/`partition_by_*`/
  `device_mesh` all hand back `ShardedArray`s). Passing such an array into any
  jitted/vmapped function — the entire point of sharding — crashes. The
  `keep_sharding=False` option was also silently lost (reset to the default).
- Repro (verified):
  ```python
  import jax, jax.numpy as jnp
  from brainpy.math.ndarray import ShardedArray
  jax.jit(lambda x: x.value + 1.)(ShardedArray(jnp.arange(3.)))
  # AttributeError: 'ShardedArray' object has no attribute '_keep_sharding'
  ```
- Fix: Added `ShardedArray.tree_flatten` (returns `(self._value,), self._keep_sharding`
  — flattens the raw `_value` to avoid running `with_sharding_constraint` during
  the abstract flatten step) and `ShardedArray.tree_unflatten` (reconstructs
  `_value` and restores `_keep_sharding` from `aux_data`, defaulting to `True`).
- Tests: `test_shardedarray_pytree_round_trip_preserves_value_and_keep_sharding`,
  `test_shardedarray_works_under_jit`, `test_shardedarray_works_under_vmap`
  (in `math_core_fixes_test.py`).
- Status: fixed

---

### P2-M1 — `remove_diag` crashes with an opaque error on tall (m > n) matrices  [Medium]
- File: brainpy/math/others.py:80-102
- Category: edge/error
- What: The docstring claims support for any `(M, N)` matrix returning
  `(M, N-1)`, but the off-diagonal index construction is inconsistent for
  `m > n`: `rows = np.repeat(np.arange(m), n - 1)` yields `m*(n-1)` indices
  while `cols` is taken from `~np.eye(m, n)` which has `m*n - n` `True` entries.
  When `m > n` these counts differ and the advanced-index gather raises an
  opaque `ValueError: Incompatible shapes for broadcasting`.
- Why it's a bug: `remove_diag` removes element `[i, i]` from each row, which is
  only well-defined when every row owns a diagonal element, i.e. `m <= n`. The
  historical implementation (boolean-mask + reshape) also failed for `m > n`,
  just at the reshape step — so this was never supported, but the new error
  message is misleading and hard to diagnose.
- Repro (verified): `remove_diag(jnp.arange(12).reshape(4, 3))` → broadcasting
  `ValueError` referencing internal gather shapes.
- Fix: Added an explicit guard that raises a clear `ValueError` (matching the
  existing `ndim` guard style) explaining the `m <= n` requirement, before the
  gather. The `m <= n` path is unchanged.
- Tests: `test_remove_diag_square_and_wide`,
  `test_remove_diag_tall_raises_clear_error`, `test_remove_diag_still_rejects_non_2d`.
- Status: fixed

---

### P2-L1 — `IdScaling._reject_overrides` raises a confusing truth-value error for array `bias`/`scale`  [Low]
- File: brainpy/math/scales.py:87-98
- Category: edge/error
- What: `_reject_overrides` does `if bias is not None and bias != 0.` / `scale != 1.`.
  When called with a non-scalar array `bias`/`scale`, `bias != 0.` is an array
  and the `and`/`if` coerces it to bool, raising
  `ValueError: The truth value of an array with more than one element is ambiguous`.
- Why it's a bug: misleading error for an unusual-but-legal input. The intent is
  to reject non-default overrides; an array override should be rejected with the
  intended "IdScaling ignores bias/scale" message, not a numpy truthiness error.
- Repro (verified): `IdScaling().offset_scaling(jnp.zeros(3), bias=jnp.zeros(3))`.
- Fix: recorded only (Low; out of fix scope). Suggested: compare with
  `np.ndim(bias) == 0 and bias != 0.` or `np.any(np.asarray(bias) != 0.)`.
- Tests: none
- Status: recorded-only

---

### P2-L2 — `set()` does not validate `bp_object_as_pytree`, unlike `environment()`  [Low]
- File: brainpy/math/environment.py:354-442 (vs `environment.__init__` :217-219)
- Category: edge/error / api-drift
- What: `environment.set()` validates `dt`, `mode`, `x64`, `float_`, `int_`,
  `bool_`, `complex_`, `numpy_func_return` up front (M-07 fix) but never checks
  that `bp_object_as_pytree` is a `bool`. `environment.__init__` does assert it.
  So `bm.set(bp_object_as_pytree='nope')` silently stores a string.
- Why it's a bug: minor API inconsistency; a bad value is stored and only
  surfaces later where the flag is consumed. Not silently-wrong numerics.
- Repro (verified): `bm.set(bp_object_as_pytree='not a bool')` stores the string.
- Fix: recorded only (Low). Suggested: add
  `if bp_object_as_pytree is not None: assert isinstance(bp_object_as_pytree, bool)`
  to the validation block.
- Tests: none
- Status: recorded-only

---

### P2-L3 — `keep_constraint` / `_keep_constraint` do not skip `SingleDeviceSharding` (inconsistent with M-09 fix)  [Low]
- File: brainpy/math/sharding.py:227-248
- Category: perf / style
- What: The M-09 fix made `ShardedArray.value` skip inserting
  `with_sharding_constraint` for `SingleDeviceSharding` (pure overhead on a
  single device). The standalone `keep_constraint`/`_keep_constraint` helpers
  still insert the constraint unconditionally. For symmetry they should apply
  the same guard.
- Why it's a bug: only a consistency/perf nit — verified that on a single CPU
  device XLA elides the constraint to an empty jaxpr (`jax.make_jaxpr` shows no
  equations), so there is no real runtime cost in jax 0.10.2. Recorded for
  consistency, not correctness.
- Repro: static / `jax.make_jaxpr(keep_constraint)(jnp.arange(3.))` → no eqns.
- Fix: recorded only (Low). Suggested: mirror the `SingleDeviceSharding` check.
- Tests: none
- Status: recorded-only

---

### P2-L4 — `Scaling.transform` raises bare `ZeroDivisionError` on a degenerate `scaled_V_range`  [Low]
- File: brainpy/math/scales.py:29-48
- Category: edge/error
- What: `scale = (V_max - V_min) / (scaled_V_max - scaled_V_min)` divides by zero
  when `scaled_V_min == scaled_V_max`, surfacing as a bare `ZeroDivisionError`
  with no context.
- Why it's a bug: invalid user input produces an unhelpful error. Low impact —
  the exception is already raised, just not descriptive.
- Repro (verified): `Scaling.transform([0., 10.], scaled_V_range=(1., 1.))`.
- Fix: recorded only (Low). Suggested: validate
  `scaled_V_max != scaled_V_min` with a clear message.
- Tests: none
- Status: recorded-only

---

## Re-verified as already-correct (prior 2026-06-18 fixes, no action)
- `enable_x64()` / `disable_x64()` keep brainstate `precision` and JAX
  `jax_enable_x64` in sync (C-10) — verified: enable→`(64, True)`, disable→`(32, False)`.
- `set()` validates before mutating (M-07).
- `Mode` is hashable and usable in sets / dict keys (H-10).
- `Array.device` is a property returning a `jax.Device`; `device_buffer`,
  `block_host_until_ready`, `block_until_ready`, `at` all work (H-11).
- `Array(scalar)` stores an array, `.shape` works (H-12).
- `_compatible_with_brainpy_array` returns `out` when `out=` is given (H-14).
- `remove_diag` traces cleanly under `jit`/`vmap` for `m <= n` (H-15).
- `ShardedArray.value` skips `with_sharding_constraint` on `SingleDeviceSharding` (M-09).
- `get_sharding` warns on a full axis-name mismatch (M-10).
- `remove_vmap` delegates to `brainstate.transform.unvmap`; global-reduction
  semantics documented and verified under `vmap`/`jit` (M-08).
- `IdScaling` rejects non-default scalar `bias`/`scale` (L-02).
- base `Array` vs `ShardedArray` value-setter policy documented (L-03).
</content>
