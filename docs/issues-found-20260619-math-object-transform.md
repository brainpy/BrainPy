# Audit 2026-06-19 — `brainpy/math/object_transform`

Reviewer: senior Python + JAX expert (P4 slice). Branch
`fix/audit-20260619-math-object-transform`. JAX 0.10.2, brainstate 0.5.1.

## Context

A prior audit (`dev/issues-found-20260618.md`) already fixed the major
Critical/High issues in this package (C-25 `VarDict.tree_unflatten`/`jax.util`,
C-26 `Variable` pytree metadata loss, H-01 `cls_jit` negative argnums, H-02
state-in-operands, H-03 zero-length pytree guard, H-04 `jit` `dyn_vars` kwargs,
H-05 `to()`/`cpu()`, H-06 `Variable.value` setter ordering, H-08
`register_implicit_vars` container flatten, H-09 `Variable.__hash__`, M-02
`cls_jit` `donate_argnums`, M-05 `ifelse` `check_cond=False`, L-04/L-05/L-06).
These were verified present and working in this worktree (all 222 in-scope tests
green at baseline). This document records a *fresh* review; remaining findings
are predominantly documented-contract / edge-case error-handling gaps.

---

### P4-M1 — `bm.cond` crashes on non-callable branches  [Medium]
- File: brainpy/math/object_transform/controls.py:96-158
- Category: edge/error
- What: The docstring types both `true_fun` and `false_fun` as
  ``callable, ArrayType, float, int, bool``, i.e. a constant branch is a
  supported input. But `cond` forwards the branches straight to
  `warp_to_no_state_input_output(true_fun)` (which just `@wraps` them) and then
  to `brainstate.transform.cond`, which calls them. A constant branch is never
  wrapped into a callable, so `bm.cond(True, 1.0, 2.0)` raises
  ``TypeError: 'float' object is not callable``. The sibling `ifelse` handles
  this correctly via its `make_callable` helper.
- Why it's a bug: A documented call form crashes. Historical BrainPy `cond`
  accepted constant branches.
- Repro: ``bm.cond(True, 1.0, 2.0)`` → ``TypeError: 'float' object is not callable``
- Fix: wrap non-callable `true_fun`/`false_fun` into zero-arg callables before
  forwarding (mirroring `ifelse.make_callable`), unwrapping any `Array`/`State`
  constant to its raw value so brainstate accepts it as an operand-free branch.
- Tests: `controls_test.py::TestCondBranchTypes` (3 cases)
- Status: fixed

### P4-M2 — `bm.ifelse` crashes on a scalar-bool `conditions`  [Medium]
- File: brainpy/math/object_transform/controls.py:161-267
- Category: edge/error
- What: The docstring types `conditions` as ``bool, sequence of bool``. The
  mutually-exclusive-condition conversion is guarded by
  ``isinstance(conditions, (list, tuple))``; a bare scalar bool falls straight
  through to `brainstate.transform.ifelse`, which immediately does
  ``len(conditions)`` and raises ``TypeError: object of type 'bool' has no
  len()``.
- Why it's a bug: A documented single-condition call form crashes.
- Repro: ``bm.ifelse(conditions=True, branches=[lambda: 1, lambda: 2])`` →
  ``TypeError: object of type 'bool' has no len()``
- Fix: normalize a scalar (non-list/tuple) `conditions` into a one-element list
  before the conversion block. The existing ``len(branches) > len(conditions)``
  branch then appends the implicit ``else`` condition, giving the correct
  two-way dispatch.
- Tests: `controls_test.py::TestIfElseScalarCondition` (2 cases)
- Status: fixed

### P4-M3 — `Collector.__sub__` raises raw `KeyError` on a missing value operand  [Medium]
- File: brainpy/math/object_transform/collectors.py:102-122
- Category: edge/error
- What: When subtracting a list/tuple that contains a *value* object (not a
  string key) which is not present in the collector, the code does
  ``id_to_keys[id(key)]`` without a membership check, raising a bare
  ``KeyError(<int id>)``. Every other "not found" path in `__sub__` raises a
  descriptive ``ValueError`` (and the co-located test
  `test_sub_with_list_missing_key_raises` asserts ``ValueError`` for the string
  case), so this is an inconsistent / unhelpful failure mode.
- Why it's a bug: Contract violation — the documented/observed behaviour for a
  missing removal target is `ValueError`, not a cryptic id-keyed `KeyError`.
- Repro:
  ```python
  c = Collector(); c['a'] = some_var
  c - [other_var_not_in_c]   # -> KeyError(140...id)
  ```
- Fix: use ``id_to_keys.get(id(key))`` and raise the same descriptive
  ``ValueError`` used elsewhere when the object is absent.
- Tests: `collectors_test.py::test_sub_with_list_missing_value_raises`
- Status: fixed

### P4-M4 — `VariableView.value` setter is non-robust and asymmetric with `Variable`  [Medium]
- File: brainpy/math/object_transform/variables.py:330-348
- Category: edge/error
- What: The setter accesses ``v.shape`` / ``v.dtype`` on the raw input *before*
  unwrapping, and only unwraps `Array` (not `brainstate.State`/`np.ndarray`).
  Consequences: ``view.value = [1., 2.]`` raises
  ``AttributeError: 'list' object has no attribute 'shape'`` and a numpy array
  is never canonicalized to the view's dtype. The parent `Variable.value`
  setter was already hardened (H-06) to unwrap `State`/`Array`/`np.ndarray`
  first; the view setter was left behind, so the two diverge.
- Why it's a bug: Assigning a plain list/number/State to a `VariableView`
  (a documented, public update path) crashes or silently mismatches dtype,
  unlike the equivalent assignment to a `Variable`.
- Repro: ``bm.VariableView(bm.Variable(bm.arange(5.)), slice(0, 2)).value = [1., 2.]``
  → ``AttributeError``
- Fix: unwrap `State`/`Array`/`np.ndarray` first (as the parent does), then use
  ``jnp.shape``/`_get_dtype` for validation. This makes `VariableView` accept
  the same inputs as `Variable` (numpy canonicalization, `State` unwrap) and,
  for a plain Python list, fail with the *same* descriptive ``MathError`` as
  the parent rather than an opaque ``AttributeError`` (a bare list remains
  rejected for both, consistent with the parent — see P4-L1).
- Tests: `object_transform_fixes_test.py::test_variable_view_setter_python_list_matches_variable`,
  `...::test_variable_view_setter_canonicalizes_numpy_dtype`,
  `...::test_variable_view_setter_unwraps_state`
- Status: fixed

### P4-L1 — `Variable.value = <python list>` yields a confusing "object" dtype error  [Low]
- File: brainpy/math/object_transform/variables.py:142-170
- Category: edge/error
- What: A plain Python list is not unwrapped/`jnp.asarray`-ed, so the dtype
  check computes ``canonicalize_dtype(list)`` → object dtype and raises
  ``MathError: ... while we got object`` instead of either accepting the list
  or giving a clear message. (Lists are not a documented input, hence Low.)
- Why it's a bug: Misleading diagnostic for a near-miss usage.
- Repro: ``bm.Variable(bm.arange(2.)).value = [1., 2.]``
- Fix: recorded only.
- Tests: none
- Status: recorded-only

### P4-L2 — `Variable.tree_unflatten` invokes `record_state_init` on every unflatten  [Low]
- File: brainpy/math/object_transform/variables.py:199-214
- Category: perf/correctness (latent)
- What: `tree_unflatten` calls ``brainstate.State.__init__`` to rebuild
  bookkeeping. That runs ``source_info_util.current()`` (non-trivial) and
  ``record_state_init(self)``, which appends the reconstructed state to every
  active ``TRACE_CONTEXT.new_state_catcher``. A `Variable` is reconstructed on
  *every* pytree round-trip (each jit/vmap/scan boundary, every `tree_map`).
  If such a round-trip happens inside a brainstate "new-state catcher" context
  (model-construction time), the rebuilt-but-not-actually-new state could be
  spuriously caught. Not reproducible through the normal brainstate transform
  paths (they close over states rather than passing Variables as pytree args),
  so left as Low.
- Why it's a bug: Theoretical state-leak / minor per-unflatten cost.
- Repro: static (no observable failure in normal usage; verified jit/tree_map
  round-trips do not leak).
- Fix: recorded only. (Reverting to full ``Variable.__init__`` would be worse —
  it re-runs batch-axis validation + naming. A clean fix needs a brainstate
  "rehydrate without recording" entry point, which is out of scope.)
- Tests: none
- Status: recorded-only

### P4-L3 — auto name counter can collide with a manually supplied name  [Low]
- File: brainpy/math/object_transform/naming.py:68-74
- Category: edge/error
- What: ``get_unique_name`` hands out ``f'{type}{counter}'`` and bumps the
  counter, ignoring names already taken manually. Creating ``Foo(name='Foo1')``
  before the auto counter reaches 1 makes the next auto-named ``Foo()`` raise
  ``UniqueNameError``. Long-standing historical BrainPy behaviour.
- Why it's a bug: Surprising collision; mitigated by `clear_name_cache()`.
- Repro: ``Foo(); Foo(name='Foo1'); Foo()`` → ``UniqueNameError``
- Fix: recorded only (historical contract; would change naming semantics).
- Tests: none
- Status: recorded-only

---

## Cross-check vs `dev/issues-found-20260618.md`

All object_transform / variables / transforms entries from the prior audit were
verified **already fixed** in this worktree and confirmed working:
C-25, C-26, H-01, H-02, H-03, H-04, H-05, H-06, H-08, H-09, M-02, M-03 (docstring
now says ``(final_carry, stacked_ys)``), M-04 (now documented), M-05, M-06 (now
documented intentional carry-passthrough), L-04, L-05, L-06. No still-present
verified bug from that list remained in scope.
