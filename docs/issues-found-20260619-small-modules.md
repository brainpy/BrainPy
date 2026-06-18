# P16 small-modules — Issues Found (2026-06-19)

Scope: brainpy/connect, brainpy/initialize, brainpy/encoding, brainpy/inputs,
brainpy/algorithms, brainpy/tools. ID prefix `P16-`.

---

### P16-C1 — `spike_current` / `ramp_current` deprecation wrappers dispatch to the wrong function  [Critical]
- File: brainpy/inputs/currents.py:144-153, 183-192
- Category: correctness / api-drift
- What: The deprecated public aliases `spike_current(*args, **kwargs)` and
  `ramp_current(*args, **kwargs)` call `constant_input(*args, **kwargs)` instead
  of `spike_input` / `ramp_input`.
- Why it's a bug: `constant_input` has a completely different signature
  (`I_and_duration`), so any documented call such as
  `spike_current(sp_times=..., sp_lens=..., sp_sizes=..., duration=...)` raises
  `TypeError: constant_input() got an unexpected keyword argument 'sp_times'`.
  These are public, exported, still-documented APIs. They are 100% broken.
- Repro:
  ```python
  import brainpy as bp
  bp.inputs.spike_current(sp_times=[10, 20], sp_lens=1., sp_sizes=0.5, duration=40.)
  # TypeError: constant_input() got an unexpected keyword argument 'sp_times'
  ```
- Fix: route `spike_current` -> `spike_input`, `ramp_current` -> `ramp_input`.
- Tests: currents_test.py::test_spike_current_alias, test_ramp_current_alias
- Status: fixed

### P16-C2 — `LogisticRegression.call` crashes on every call (flatten then `.shape[1]`)  [Critical]
- File: brainpy/algorithms/offline.py:386-389
- Category: correctness
- What: `targets = targets.flatten()` makes `targets` 1-D, then the next line
  calls `self.init_weights(inputs.shape[1], targets.shape[1])`, indexing
  `shape[1]` of a 1-D array.
- Why it's a bug: `IndexError: tuple index out of range` on any invocation —
  the algorithm can never run.
- Repro:
  ```python
  from brainpy.algorithms.offline import LogisticRegression
  import brainpy.math as bm
  X = bm.random.rand(20, 3); y = bm.asarray((bm.random.rand(20,1) > .5).astype(float))
  LogisticRegression(max_iter=10)(y, X)   # IndexError
  ```
- Fix: initialise weights as `init_weights(inputs.shape[1], 1)` and flatten to a
  1-D parameter vector to match the 1-D `targets` used by the body function.
- Tests: offline_test.py::test_logistic_regression_runs
- Status: fixed

### P16-H1 — `ElasticNetRegression` train/predict feature mismatch (`add_bias` ignored in training)  [High]
- File: brainpy/algorithms/offline.py:542
- Category: correctness
- What: `call()` builds features with `polynomial_features(inputs, degree=...)`
  (so `add_bias` defaults to `True`), while `predict()` builds features with
  `polynomial_features(X, degree=..., add_bias=self.add_bias)`. With the default
  `add_bias=False`, training adds a bias column but prediction does not.
- Why it's a bug: weights are fit with `n+1` features but prediction supplies
  `n` features -> `dot_general` shape error (or, if shapes happened to align,
  silently wrong predictions).
- Repro:
  ```python
  from brainpy.algorithms.offline import ElasticNetRegression
  import brainpy.math as bm
  en = ElasticNetRegression(max_iter=20, add_bias=False)
  W = en(bm.random.rand(15,1), bm.random.rand(15,2))
  en.predict(W, bm.random.rand(15,2))   # TypeError: contracting dims (6,) vs (7,)
  ```
- Fix: pass `add_bias=self.add_bias` in `call()` so training and prediction use
  identical feature construction.
- Tests: offline_test.py::test_elasticnet_train_predict_consistent
- Status: fixed

### P16-H2 — `CSRConn.build_csr` consistency check is dead (`self.pre_num != self.pre_num`)  [High]
- File: brainpy/connect/custom_conn.py:103
- Category: edge/error
- What: The guard `if self.pre_num != self.pre_num:` compares a value with
  itself and is always `False`, so the "(pre_size, post_size) inconsistent with
  the sparse matrix" error can never fire.
- Why it's a bug: A `CSRConn` whose declared `pre_size` disagrees with the
  `indptr` length silently produces a malformed CSR (e.g. claims `pre_num=5`
  but returns an `indptr` of length 4), instead of raising. The clear intent is
  to compare the user-supplied `pre_num` against `self.inptr.size - 1`.
- Repro:
  ```python
  import numpy as np, brainpy as bp
  c = bp.conn.CSRConn(np.array([0,1,2,0,1], np.int32), np.array([0,2,3,5], np.int32))  # 3 pre
  c.require(5, 3, 'csr')   # no error -> malformed CSR (indptr len 4 for pre_num 5)
  ```
- Fix: compare `self.pre_num != self.inptr.size - 1` and raise `ConnectorError`.
- Tests: custom_conn_test.py::test_csrconn_inconsistent_pre_num_raises,
  test_csrconn_consistent_ok
- Status: fixed

### P16-M1 — `coo2csr` scatters int counts into a `uint32` buffer (FutureWarning -> future error)  [Medium]
- File: brainpy/connect/base.py:692, 700-702
- Category: numerics / api-drift
- What: `final_pre_count = onp.zeros(num_pre, dtype=jnp.uint32)` (and the jax
  branch `bm.zeros(num_pre, dtype=jnp.uint32)`) is filled with `pre_count`
  values whose dtype is the platform int (int32/int64). On the jax branch this
  triggers `FutureWarning: scatter inputs have incompatible types: cannot safely
  cast value from dtype=int32 to dtype=uint32 ...` and is slated to become a hard
  error in future JAX.
- Why it's a bug: `coo2csr` is on the default code path for building CSR/PRE2POST
  structures from COO connectivity; a future JAX release will turn the cast into
  an exception, breaking connectivity construction.
- Repro (static / warning):
  ```python
  import jax.numpy as jnp
  from brainpy.connect.base import coo2csr
  coo2csr((jnp.array([0,0,1,2,2,2]), jnp.array([1,2,0,0,1,2])), 3)  # FutureWarning
  ```
- Fix: build the count buffer with the connection index dtype (`get_idx_type()`)
  so the scatter dtype matches.
- Tests: base_coverage_test.py / random_conn — covered by no-warning assertion in
  custom_conn_test.py::test_coo2csr_no_dtype_warning
- Status: fixed

### P16-M2 — `polynomial_features` allocates one extra (always-zero) feature column  [Medium]
- File: brainpy/algorithms/utils.py:107-117
- Category: correctness / edge
- What: width is `1 + n_features + len(combinations)` after `n_features += 1`
  (when `add_bias`), but only `1` bias + original features + `len(combinations)`
  columns are written, leaving the final column permanently `0`. With
  `add_bias=False` the same `1 +` adds a spurious leading allocation slot too.
- Why it's a bug: every transformed design matrix carries a dead all-zero
  feature column, inflating the weight vector by one element (and any code that
  reasons about feature counts, e.g. RidgeRegression's per-feature penalty
  vector, sees the wrong dimensionality). Regression still "works" only because
  pinv/least-squares assigns ~0 weight to the zero column.
- Repro:
  ```python
  import brainpy.math as bm
  from brainpy.algorithms.utils import polynomial_features
  polynomial_features(bm.arange(6.).reshape(2,3), degree=2, add_bias=True).shape  # (2, 11), last col all 0
  ```
- Fix: drop the spurious `1 +`; allocate exactly `n_features + len(combinations)`
  columns (the bias slot is already accounted for by `n_features += 1`).
- Tests: utils_test.py::test_polynomial_features_no_dead_column
- Status: fixed

---

## Recorded only (Low — not fixed)

### P16-L1 — `LatencyEncoder` docstring example output shape is wrong  [Low]
- File: brainpy/encoding/stateful_encoding.py:111-120
- Category: style/docs
- What: The docstring shows `encoder.multi_steps(a, n_time=5)` producing a
  `(5, 3)` array, but `n_time` is a duration divided by `bm.get_dt()` (0.1), so
  the real output is `(50, 3)`. Misleading example.
- Status: recorded-only

### P16-L2 — `coo2csr` jax branch returns a host (numpy) `indptr` via `onp.insert`  [Low]
- File: brainpy/connect/base.py:704
- Category: perf/consistency
- What: `indptr = onp.insert(indptr, 0, 0)` converts a jax array to numpy, so the
  jax branch returns `indices` as a jax array but `indptr` as numpy — an
  inconsistent (device/host) return and an implicit device->host transfer. Path
  is already non-jittable (uses `jnp.unique` w/ return_counts), so no current
  crash; recorded only.
- Status: recorded-only

### P16-L3 — `_check_none` in initialize/generic.py is an empty no-op  [Low]
- File: brainpy/initialize/generic.py:36-37
- Category: dead code
- What: `_check_none(x, allow_none=False)` has a `pass` body and is never called.
- Status: recorded-only

### P16-L4 — `numba_jit(_random_subset)` wraps a Python-`set` builder  [Low]
- File: brainpy/connect/random_conn.py:706-713, 823-830, 969-976
- Category: perf/correctness-risk
- What: When numba is installed, `_random_subset` (which builds a Python `set`)
  is njit-compiled. Sets in nopython mode are limited; relies on numba fallback
  / object mode. Works on the tested machine; recorded as fragility only.
- Status: recorded-only

### P16-L5 — `RidgeRegression.__repr__` mislabels `alpha` as `beta`  [Low]
- File: brainpy/algorithms/offline.py:287
- Category: style
- What: `__repr__` prints `beta={self.regularizer.alpha}` though `beta` is the
  deprecated alias.
- Status: recorded-only

---

## Cross-check vs dev/issues-found-20260618.md (in-scope entries)

The prior audit's Critical/High entries that fall in this slice were re-verified
against the current worktree code and found **already fixed** (no action needed):

- C-23 (online.py RLS wrong for batch>1): now uses proper block RLS
  `K = P Hᵀ (I_B + H P Hᵀ)⁻¹`; verified `dw` shape correct for B=4.
- C-24 (PoissonEncoder.single_step crash): `single_step` now draws a single
  Bernoulli sample directly; verified it runs.
- H-46 (offline.py GD `.value` AttributeError on jax tracer): gradient-descent
  path now uses `jax.lax.while_loop` with no `.value`; verified Ridge/Linear GD run.
- H-47 (ridge penalizes bias column): now skips the intercept penalty for
  `add_bias` (PolynomialRidge); verified.
- M-30 (FixedProb floors `int(post_num*prob)` to 0; contradictory include_self
  guard): now uses `int(round(...))` and the guard was removed; verified.

Still-present prior entry (Low only):

- L-11 (LatencyEncoder docstring example output shape ignores dt) — same as
  P16-L1 above; recorded only.
