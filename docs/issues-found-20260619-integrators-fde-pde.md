# Issues found — FDE/PDE integrators (2026-06-19)

Reviewer: senior numerical-methods + JAX expert (fractional differential equations).
Scope: `brainpy/integrators/fde/{Caputo,GL,base,generic}.py`, `brainpy/integrators/pde/base.py`.
Environment: brainpy 2.7.8, jax 0.10.2, scipy (CPU).

Core numerics were validated head-to-head against independent reference implementations
(rectangle product-integration Caputo–Euler [Li & Zeng 2013], L1 scheme, and the
Grünwald–Letnikov short-memory recurrence), single- and multi-variable, with distinct
per-variable `alpha`, both inside and beyond the memory window. **All three solvers match
their analytic/reference recurrences to machine precision** — the previously-reported
numerical bugs (initial-condition mis-scaling, coefficient order, memory truncation) are
**not present** in this worktree (see "Already fixed" section).

---

### P7-H1 — `CaputoEuler.reset` replaces memory `Variable`s with plain `Array`s (state desync)  [High]
- File: brainpy/integrators/fde/Caputo.py:173 (and 172 is fine)
- Category: correctness / JAX-semantics
- What: `reset()` does `self.f_states[key] = bm.zeros(...)`. `self.f_states` is a plain
  Python `dict` (created at `__init__`, line 160), not a `bm.VarDict`, so this assignment
  *replaces* the registered `bm.Variable` with a bare `bm.Array`. The `bm.Variable`
  originally registered via `register_implicit_vars(self.f_states)` is now orphaned: it is
  still returned by `self.vars()` (stale), while `_integral_func` writes the memory buffer
  into the *new* `Array` object.
- Why it's a bug: any JAX-transformed re-run after `reset` (the documented way to re-run
  from new initial values, e.g. via `IntegratorRunner`) snapshots/restores the **stale**
  `Variable` from `self.vars()` while the integral function reads/writes the **new** `Array`,
  so the convolution memory is desynced and results are silently wrong. (`CaputoL1Schema`
  and `GLShortMemory` are immune — they store their memory in `bm.VarDict`, whose
  `__setitem__` does in-place `.value` assignment.)
- Repro:
  ```python
  import brainpy as bp, brainpy.math as bm, numpy as np
  bm.enable_x64()
  intg = bp.fde.CaputoEuler(lambda y, t: -y, alpha=0.8, num_memory=100, inits=[1.])
  intg.reset([1.])
  assert isinstance(intg.f_states['y'], bm.Variable)   # FAILS: it is a bm.Array
  runner = bp.IntegratorRunner(intg, monitors=['y'], dt=0.05, inits=[1.])
  runner.run(1.0)
  # last y == 0.911 (wrong); a fresh integrator (no reset) gives 0.380 (correct)
  ```
- Fix: store `f_states` in a `bm.VarDict` and reset via in-place `.value` assignment so the
  registered `Variable` identity is preserved.
- Tests: `Caputo_test.py::TestCaputoEulerReset::test_reset_preserves_variable`,
  `::test_reset_then_run_matches_fresh`
- Status: fixed

---

### P7-M1 — `fdeint(method=...)` default makes `set_default_fdeint` a no-op  [Medium]
- File: brainpy/integrators/fde/generic.py:35 (default), used at :63
- Category: api-drift / correctness
- What: `fdeint(..., method='l1', ...)` has the *literal* default `'l1'`. The body then does
  `method = _DEFAULT_FDE_METHOD if method is None else method`, but `method` is never `None`
  when the caller omits it, so the `_DEFAULT_FDE_METHOD` global (settable via
  `set_default_fdeint`) is ignored for default-method calls.
- Why it's a bug: `set_default_fdeint('euler')` followed by `fdeint(...)` still builds a
  `CaputoL1Schema`, contradicting the documented purpose of `set_default_fdeint` /
  `get_default_fdeint`. The public default-method mechanism is dead for the common path.
- Repro:
  ```python
  from brainpy.integrators.fde.generic import fdeint, set_default_fdeint
  set_default_fdeint('euler')
  type(fdeint(alpha=0.8, num_memory=20, inits=[1.], f=lambda y, t: -y)).__name__
  # 'CaputoL1Schema'  (expected 'CaputoEuler')
  ```
- Fix: change the keyword default to `method=None` so the `_DEFAULT_FDE_METHOD` fallback
  actually runs.
- Tests: `generic`-level test in `Caputo_test.py::TestFdeintDefaultMethod::test_set_default_fdeint_respected`
- Status: fixed

---

## Already fixed in this worktree (verified, recorded only)

These were reported against an earlier revision (`dev/issues-found-20260618.md`, FDE block)
and are **already corrected** in the code under review. Re-verified here; no further action.

- **C-08 / `CaputoEuler` initial-condition scaling** — `Caputo.py:211` now reads
  `self.inits[key] + integral * (dt**alpha/alpha)` with `integral = coef @ f_states`, i.e.
  `y0` is added *outside* the `dt^alpha/alpha` scaling. Verified `D^a y=0, y0=1 → y≡1` and a
  full `f=y, y0=2` reference run match to 1e-15. Status: recorded-only (no change needed).
- **H-30 / `GLShortMemory.reset` KeyError** — `GL.py:187` uses `key + '_delay'`. `reset` runs
  cleanly. Status: recorded-only.
- **H-31 / `CaputoL1Schema.hists()` ValueError** — `Caputo.py:384` uses `.items()`. `hists()`
  returns a dict cleanly. Status: recorded-only.
- **H-32 / `set_default_fdeint` wrong global** — `generic.py:87-88` assigns
  `_DEFAULT_FDE_METHOD`. `get_default_fdeint()` reflects the set value. Status: recorded-only.
  (But see P7-M1: the value is then ignored by `fdeint`'s literal default.)

---

## Low (recorded only — not fixed per task policy)

- **P7-L1** — `Caputo.py:192` type-checks `isinstance(devs, (bm.ndarray, jax.Array))` while
  `GL.py`/`CaputoL1Schema` use `bm.Array`. `bm.ndarray is bm.Array`, so this is cosmetic
  inconsistency only. Category: style.
- **P7-L2** — `generic.py:36` annotates `dt: str = None` in `fdeint`; should be
  `dt: float = None`. Category: style/typing.
- **P7-L3** — Docstrings of `CaputoL1Schema`/`GLShortMemory` say "fractional order in (0, 1)"
  in the `UnsupportedError` message, but both classes (correctly) accept `alpha == 1`.
  The class-level `Parameters` docstrings say `(0., 1.]`. Message/docstring drift only.
  Category: style/docs.
- **P7-L4** — Pervasive `Parameters::` / `Returns::` / `References::` / `Examples::`
  literal-block markers (won't render as NumPy-doc sections, violates CLAUDE.md). Present in
  all FDE files. Category: style/docs.
- **P7-L5** — `Caputo.py:50` docstring typo "may be arbitrary real numbers" written as
  "ay be"; `generic.py:107` "name: ste". Category: style/docs.
- **P7-L6** — `pde/base.py` `PDEIntegrator(Integrator): pass` is an unused stub with no PDE
  solvers, no docstring, not in any `__all__`. No functional bug; documents intent only.
  Category: style/dead-code.

---

## Out-of-scope / cross-cutting (left unfixed)

None. All identified Critical/High/Medium issues are inside scope and were fixed.
