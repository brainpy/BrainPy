# P5 Audit — math/sparse, math/event, math/jitconn, delayvars, pre_syn_post

Date: 2026-06-19
Branch: `fix/audit-20260619-math-sparse-event`
Environment: jax 0.10.2, brainevent 0.1.0 (CPU), brainpy (worktree).

## Scope reviewed

- `brainpy/math/sparse/coo_mv.py`
- `brainpy/math/sparse/csr_mm.py`
- `brainpy/math/sparse/csr_mv.py`
- `brainpy/math/sparse/jax_prim.py`
- `brainpy/math/sparse/utils.py`
- `brainpy/math/event/csr_matmat.py`
- `brainpy/math/event/csr_matvec.py`
- `brainpy/math/jitconn/event_matvec.py`
- `brainpy/math/jitconn/matvec.py`
- `brainpy/math/delayvars.py`
- `brainpy/math/pre_syn_post.py`

## Summary

The sparse/event/jitconn operators in this branch are thin, **correct** wrappers
over `brainevent` 0.1. Every matvec/matmat path (including `transpose=True`,
event/binary masking, jitconn RNG-on-the-fly, autodiff, `jit`, `vmap`) was
verified numerically against dense references and matches. The major prior
audit bugs were **already fixed in this branch** before this review:

- `csrmm(transpose=True)` now does `csr.T @ matrix` (prev. C-07) — verified `Aᵀ@B`.
- `TimeDelay._true_fn` ring-buffer read now uses `% num_delay_step` (prev. C-09) — verified.
- `coomv` converts COO→CSR via `brainevent.coo2csr` (prev. H-17) — verified.
- `coo_to_csr` uses `argsort(stable=True)` + `.at[].set` + int dtype (prev. H-18) — verified.
- `csr_to_dense` wraps `brainevent.CSR(...).todense()` (prev. H-19) — verified.
- `TimeDelay.reset` mirrors `__init__` (dtype, callable before_t0) (prev. M-14) — verified.
- jitconn `seed=None` non-reproducibility documented in warnings (prev. M-13).

The fresh review found **1 Medium** genuine correctness trap (fixed) plus
several Low items (recorded only).

---

### P5-M1 — `coo_to_csr` silently produces a corrupt CSR when a `pre_id >= num_row`  [Medium]
- File: brainpy/math/sparse/utils.py:46-51
- Category: correctness / edge / error
- What: `final_pre_count.at[unique_pre_ids].set(pre_count)` drops any
  `pre_id >= num_row` because JAX silently ignores out-of-bounds scatter
  indices (default `mode='drop'`). The dropped entry's column index still
  remains in `indices`, but its contribution is missing from `indptr`, so the
  returned `indptr[-1]` no longer equals `len(indices) == nse`. The result is a
  structurally invalid CSR (row pointers and data length disagree), which then
  silently produces wrong connectivity / wrong matvec results downstream rather
  than raising.
- Why it's a bug: a too-small `num_row` (or a stray out-of-range pre id) yields
  silently-wrong output instead of an error. CSR consumers assume
  `indptr[-1] == nse`.
- Repro:
  ```python
  from brainpy.math.sparse.utils import coo_to_csr
  import numpy as np
  pre = np.array([0, 1, 3]); post = np.array([1, 2, 0])  # 3 >= num_row=3
  idx, iptr = coo_to_csr(pre, post, num_row=3)
  # idx == [1,2,0] (len 3) but iptr == [0,1,2,2] -> iptr[-1]=2 != 3  (corrupt)
  ```
- Fix: validate eagerly that `pre_ids` and `post_ids` are within
  `[0, num_row)` / `[0, ...)` and raise a clear `ValueError` for out-of-range
  pre ids. (`coo_to_csr` already cannot be `jit`-traced because of `jnp.unique`,
  so an eager bounds check does not regress any transform behaviour.)
- Tests: `test_coo_to_csr_out_of_range_pre_id_raises`,
  `test_coo_to_csr_valid_roundtrip`, `test_coo_to_csr_unsorted`,
  `test_coo_to_csr_empty_rows` in `brainpy/math/sparse/utils_test.py`.
- Status: fixed

---

### P5-L1 — `coo_to_csr` cannot be used inside `jit`/`vmap` (`jnp.unique`)  [Low]
- File: brainpy/math/sparse/utils.py:46
- Category: edge / perf
- What: `jnp.unique(pre_ids, return_counts=True)` requires a static output
  `size`; under `jit`/`vmap` it raises `ConcretizationTypeError`. The helper is
  a setup-time (eager) preprocessing utility, so this is acceptable, but it is
  undocumented.
- Why it's a bug: API surprise; a caller expecting JAX-transformable behaviour
  gets a trace-time crash.
- Repro: `jax.jit(lambda p,q: coo_to_csr(p,q,num_row=3))(pre, post)` raises.
- Fix: recorded only (document as eager-only, or reimplement via
  `segment_sum`/`bincount`).
- Tests: none
- Status: recorded-only

### P5-L2 — `LengthDelay.reset` dead read `self.data.value` + raw `_value` write  [Low]
- File: brainpy/math/delayvars.py:442-444
- Category: style / dead code
- What: `self.data.value` on its own line is a no-op (reads and discards). The
  next line writes `self.data._value = ...`, bypassing the `Variable` setter and
  any validation/sharding logic. Works today but is fragile.
- Fix: recorded only (use `self.data.value = jnp.zeros(...)`; drop the orphan read).
- Tests: none
- Status: recorded-only

### P5-L3 — `LengthDelay.reset` `delay_len is None` guard is effectively dead  [Low]
- File: brainpy/math/delayvars.py:427-430
- Category: style
- What: `__init__` initialises `self.num_delay_step = 0` (an int, never `None`),
  so the `if self.num_delay_step is None: raise` branch can never fire; a fresh
  object reaching `reset(delay_len=None)` would compute `0 - 1 == -1`. In
  practice `__init__` always passes an explicit `delay_len`, so this is latent.
- Fix: recorded only.
- Tests: none
- Status: recorded-only

### P5-L4 — `get_*_weight_matrix` do not unwrap a brainpy `Array` `weight`  [Low]
- File: brainpy/math/jitconn/matvec.py:337 (homo)
- Category: style / consistency
- What: `get_homo_weight_matrix` passes `weight` straight to
  `brainevent.JITCScalarR` without the `isinstance(weight, Array)` unwrap that
  `mv_prob_homo` performs. brainevent currently tolerates the `Array`, so no
  crash, but it is inconsistent with the matvec entry points.
- Fix: recorded only.
- Tests: none
- Status: recorded-only

### P5-L5 — `TimeDelay` non-multiple `delay_len/dt` interpolates the delay-0 query  [Low]
- File: brainpy/math/delayvars.py:285-307
- Category: numerics (by-design)
- What: When `delay_len` is not an integer multiple of `dt`, querying delay 0
  (`time == current_time`) returns a linear interpolation between the two most
  recent buffer slots rather than the most-recent value, because `delay_len` is
  used as the fixed reference offset into the ring buffer. This is the
  established TimeDelay semantics (the most-recent sample sits at a fixed buffer
  slot keyed off `delay_len`), not a regression, but it can surprise users who
  pick a `delay_len` that is not a multiple of `dt`.
- Fix: recorded only.
- Tests: none
- Status: recorded-only

### P5-L6 — `TimeDelay`/`LengthDelay` out-of-window reads are silently wrong when checking is off  [Low]
- File: brainpy/math/delayvars.py:264-283, 489-490
- Category: edge / error
- What: The bounds checks (`_check_time1/2`, `_check_delay`) only run under
  `brainpy.check.is_checking()` (off by default). Out-of-window queries then
  index the ring buffer with a wrapped/clamped index and silently return a wrong
  value. This matches the documented contract (query within the window) but is a
  foot-gun.
- Fix: recorded only.
- Tests: none
- Status: recorded-only

---

## Cross-check vs `dev/issues-found-20260618.md`

Entries touching this slice and their status in this branch:

| Prior ID | Title | Status in branch |
|----------|-------|------------------|
| C-07 | `csrmm(transpose=True)` wrong product | already fixed (`csr.T @ matrix`), re-verified |
| C-09 | `TimeDelay` read omits modulo | already fixed (`% num_delay_step`), re-verified |
| H-17 | `coomv` builds removed `brainevent.COO` | already fixed (COO→CSR), re-verified |
| H-18 | `coo_to_csr` broken (`argsort(kind=)`, in-place, float indptr) | already fixed, re-verified; tightened (P5-M1) |
| H-19 | `csr_to_dense` stale signature | already fixed (`CSR(...).todense()`), re-verified |
| M-13 | jitconn `seed=None` non-reproducible | documented in docstrings (warning blocks) |
| M-14 | `TimeDelay.reset` drops dtype / ignores callable before_t0 | already fixed, re-verified |

No still-present verified bug from the prior audit remains in this slice.
