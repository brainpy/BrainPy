# Audit findings — optim + losses slice (2026-06-19)

Scope: `brainpy/optim/optimizer.py`, `brainpy/optim/scheduler.py`,
`brainpy/losses/base.py`, `brainpy/losses/comparison.py`,
`brainpy/losses/regularization.py`, `brainpy/losses/utils.py`.

Several Critical/High issues flagged by the 2026-06-18 audit (C-01 Adam/AdamW
bias correction, C-02 nll sign, C-03 CE class-weight, C-04 MultiStepLR,
H-52 Adam lr-Variable mutation, H-53 CrossEntropyLoss ignore/smoothing,
M-29 SM3 attr ordering, CTC `.value` crash) were already fixed in this
worktree before this pass; they were re-verified as fixed and are not
re-listed except where a residual remained.

---

### P1-C1 — `Adan.update` crashes on every call (cond operand mis-binding)  [Critical]
- File: brainpy/optim/optimizer.py:818
- Category: correctness
- What: `cond(step == 0, lambda pg, g: g, lambda pg, g: pg, (prev_g_var.value, g))`
  passes a single 2-tuple as the operand. `jax.lax.cond` calls
  `branch(*operands)`, so the tuple is splatted: `pg=(prev, g)` and `g`
  is left unbound → `TypeError: <lambda>() missing 1 required positional
  argument: 'g'`. The whole `Adan` optimizer is unusable (both `no_prox`
  branches are unreachable).
- Why it's a bug: `jax.lax.cond(pred, true_fun, false_fun, *operands)` unpacks
  `operands` positionally into each branch function. A 2-tuple operand must be
  matched by a 2-arg branch, OR the two values must be passed as two operands.
- Repro: `O.Adan(lr=1e-2, train_vars={'w': v}).update({'w': g})` → TypeError.
- Fix: replaced the broken first-step guard with a self-managed per-update
  step `bm.Variable` (mirrors the `Adam`/`AdamW` pattern, independent of the
  LR scheduler's stale `last_epoch`). On the first update `pre_g := g` (so the
  gradient difference is 0) via `jnp.where(step == 1, g, prev_g)`; the moment
  bias-correction terms now use the real per-update step `t`.
- Tests: `test_adan_runs_and_updates`, `test_adan_no_prox_runs`,
  `test_adan_first_step_diff_is_zero`, `test_adan_step_counter_advances`
  (optimizer_test.py). Updated the obsolete `optimizer_coverage_test.py::TestAdan::test_update_is_currently_broken`.
- Status: fixed

### P1-C2 — `Adan` step counter frozen at 0 (bias correction never advances)  [Critical]
- File: brainpy/optim/optimizer.py:808-811
- Category: correctness
- What: `step = self.lr.last_epoch.value + 1`. With the default `Constant`
  scheduler `last_epoch` starts at `-1` and is *never* advanced by `update()`
  (only `step_epoch()` advances it, which optimizers never call). So `step`
  is `0` forever, `correct_{m,v,n} = 1/(1-(1-beta)**1)` is frozen, and the
  first-step guard `step == 0` is *always* true → the gradient-difference term
  `v` is permanently pinned to 0 and Nesterov momentum is disabled. This is the
  Adan analogue of C-01 (Adam/AdamW), which was fixed with a self-managed
  counter; Adan was left on the stale `last_epoch` source.
- Why it's a bug: same root cause as C-01 / M-01 — `last_epoch` is not the
  per-update step source.
- Repro: static + `test_adan_step_counter_advances`.
- Fix: introduced `self.step` `bm.Variable`, incremented once per `update()`;
  bias-correction and the first-step guard now key off it. (Fixed together
  with P1-C1.)
- Tests: as P1-C1.
- Status: fixed

### P1-H1 — `SM3` raises `KeyError` for scalar (0-dim) trainable variables  [High]
- File: brainpy/optim/optimizer.py:1083-1089, 1099-1104
- Category: edge/error
- What: `register_train_vars` builds accumulators with `for i in range(ndim)`.
  For a 0-dim (scalar) variable `ndim == 0`, so no `{k}_m{i}` accumulator is
  created, yet `update` unconditionally reads `self.implicit_vars[f'{k}_m0']`
  → `KeyError: 'w_m0'`. Scalar parameters are common (e.g. a learnable bias /
  temperature), so SM3 crashes on first use for them.
- Why it's a bug: the cover construction degenerates for rank-0 tensors. SM3
  on a scalar should behave like Adagrad on a single accumulator.
- Repro: `O.SM3(lr=0.1, train_vars={'w': scalar_var}).update({'w': g})` → KeyError.
- Fix: treat rank-0 variables as rank-1 — always register at least one
  accumulator (`max(ndim, 1)`), and in `update` clamp `ndim` to `>= 1` so the
  single-accumulator path runs (equivalent to Adagrad for scalars).
- Tests: `test_sm3_scalar_var_runs` (optimizer_test.py). Updated the obsolete
  `optimizer_coverage_test.py::TestSM3::test_scalar_var_is_broken`.
- Status: fixed

### P1-H2 — `multi_margin_loss` crashes on `bm.Array` inputs  [High]
- File: brainpy/losses/comparison.py:1048-1051
- Category: edge/error
- What: the function indexes `predicts[jnp.arange(batch_size), targets]` and
  calls `.at[...]` directly on the (possibly `bm.Array`) `predicts`/`targets`
  without `bm.as_jax(...)`. Under JAX ≥0.9 passing a `bm.Array` into
  `jnp`/advanced-indexing raises
  `ValueError: Triggering __jax_array__() during abstractification is no
  longer supported`. Every other loss in this file accepts `bm.Array`
  (the public brainpy array type), so this is an inconsistent crash trap;
  the public API contract is broken for the documented array type.
- Why it's a bug: ecosystem drift — implicit `__jax_array__` coercion was
  removed; brainpy losses must convert via `bm.as_jax`.
- Repro: `C.multi_margin_loss(bm.asarray(p), bm.asarray(t))` → ValueError.
- Fix: convert `predicts`/`targets` via `bm.as_jax(...)` at the top of the
  function (idiom used throughout the module).
- Tests: `test_multi_margin_accepts_bm_array` (comparison_test.py).
- Status: fixed

### P1-M1 — `l1_loss` functional default is `'sum'` (mismatches class + docstring)  [Medium]
- File: brainpy/losses/comparison.py:573
- Category: api-drift
- What: `def l1_loss(logits, targets, reduction='sum')`. The `L1Loss` class,
  the function's own docstring ("Default: ``'mean'``"), PyTorch's `L1Loss`,
  and `braintools.metric.l1_loss` (default `'sum'` in code but `'mean'` in its
  docstring) all disagree. A user calling `l1_loss(x, y)` silently gets the
  summed (not mean) error.
- Why it's a bug: surprising default that contradicts the documented contract
  and the OO wrapper. Flagged as L-10 in the 2026-06-18 audit; on closer review
  it is a behavioral/contract mismatch, not pure style, so treated as Medium.
- Repro: `C.l1_loss(x, y)` returns the sum, while `C.L1Loss()(x, y)` returns
  the mean.
- Fix: changed the functional default to `reduction='mean'` to match the class,
  the docstring and PyTorch.
- Tests: `test_l1_loss_default_is_mean` (comparison_test.py). The pre-existing
  `comparison_test.py::TestReductionDefaults::test_l1_loss_default_is_sum`
  asserted the buggy default and is updated.
- Status: fixed

### P1-L1 — `l1_loss` `'none'` reduction returns per-row L1 *norm*, not elementwise abs  [Low]
- File: brainpy/losses/comparison.py:621 (delegates to `braintools.metric.l1_loss`)
- Category: style/docs
- What: brainpy's `l1_loss` docstring promises the unreduced loss is the
  elementwise `l_n = |x_n - y_n|` (PyTorch semantics, same shape as input).
  The delegated `braintools.metric.l1_loss` reshapes to `(N, -1)` and returns
  the per-row L1 norm (sum over trailing axes), so for a `(2, 2)` input the
  `'none'` output is shape `(2,)` per-row sums, not `(2, 2)`. The two
  pre-existing coverage tests `TestRegressionLosses::test_l1_loss_reductions`
  / `test_l1_class` encode an *incorrect* expectation (per-row mean) and fail
  on baseline regardless of any change here.
- Why it's a bug: docstring/behavior mismatch introduced by the braintools
  delegation. Not fixed because correcting it would require either re-routing
  `l1_loss` away from `braintools.metric` (cross-cutting, changes public
  numerics for all callers) or rewriting upstream — out of scope.
- Repro: `np.asarray(C.l1_loss([[1,2],[3,4]], zeros, reduction='none'))` →
  `[3., 7.]` (shape `(2,)`), docstring implies shape `(2, 2)`.
- Fix: recorded only. (The two stale coverage tests are left as-is except where
  P1-M1 forces an update to the default-reduction test; the two `l1_loss`
  reduction-value assertions are pre-existing baseline failures noted in the
  report.)
- Status: recorded-only

### P1-L2 — `Adan._update_moments` is dead code with mismatched return order  [Low]
- File: brainpy/optim/optimizer.py:798-803
- Category: style/dead code
- What: the helper computes `m`, `gd`, `v`, `n` and returns `(m, n, v)`, but it
  is never used by `update` (which inlines the same math). The lone caller is a
  coverage test. The `(m, n, v)` ordering vs the `m, n, v = ...` unpacking in
  callers is internally consistent but the helper duplicates logic that can
  drift from `update`.
- Why it's a bug: dead/duplicated code; maintenance hazard.
- Fix: recorded only (not removed — keeping the diff focused on correctness; a
  test references it).
- Status: recorded-only

### P1-L3 — `Optimizer`/`SGD`/etc. docstrings use `Parameters::` (renders as literal block)  [Low]
- File: brainpy/optim/optimizer.py:46, 114, 169, … (pervasive); scheduler.py:91, …
- Category: style/docs
- What: NumPy-doc sections are written as `Parameters::` / `References::`
  (double colon → reStructuredText literal block) instead of the underlined
  `Parameters\n----------` form required by the project docstring style.
  Sphinx will not render these as parameter tables.
- Why it's a bug: violates the repo's documented NumPy-doc convention.
- Fix: recorded only (pervasive cosmetic change, out of correctness scope).
- Status: recorded-only

### P1-L4 — `SM3` not exported in `optimizer.__all__`  [Low]
- File: brainpy/optim/optimizer.py:28-40
- Category: style/api
- What: `SM3` is a fully implemented public optimizer class but is absent from
  `__all__` (every other optimizer is listed), so `from ...optimizer import *`
  does not expose it and Sphinx `automodule` may skip it.
- Why it's a bug: minor public-API inconsistency.
- Fix: recorded only (touching `__all__` export surface is borderline; left to
  the toplevel-glue owner since `optim/__init__.py` is out of scope).
- Status: recorded-only

### P1-L5 — `ExponentialLR.__call__` uses `self.last_epoch + 1` (Variable, not `.value`)  [Low]
- File: brainpy/optim/scheduler.py:343
- Category: style/correctness-latent
- What: `i = (self.last_epoch + 1) if i is None else i` reads the `bm.Variable`
  object directly rather than `self.last_epoch.value` (as `StepLR`,
  `CosineAnnealingLR`, etc. do). It happens to work because `bm.Variable`
  supports `+`, but it is inconsistent and the `__repr__` similarly prints the
  Variable object. Latent fragility, not an active wrong-result bug (the
  arithmetic broadcasts correctly).
- Why it's a bug: inconsistent Variable handling; could surprise under tracing
  if the Variable identity leaks.
- Fix: recorded only (no observed wrong result; default optimizers never call
  `ExponentialLR()` with `i=None` in a traced loop in-scope).
- Status: recorded-only
