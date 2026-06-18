# Analysis package audit — 2026-06-19 (P13)

Scope: `brainpy/analysis/*` (low-dim phase-plane/bifurcation, high-dim slow points,
stability classification, utils). Branch `fix/audit-20260619-analysis`.

Severity scale: Critical (silently wrong/crash in default usage), High (wrong in
realistic cases / broken public API), Medium (edge/fragility/perf), Low (style/docs).

---

### P13-C1 — 2D stable/unstable STAR vs DEGENERATE node classification is inverted and partly dead code  [Critical]
- File: brainpy/analysis/stability.py:147-163
- Category: correctness
- What: For a repeated-eigenvalue 2D fixed point (`e = p*p - 4*q == 0`), the code does
  `w = np.linalg.eigvals(J); if w[0] == w[1]: return *_DEGENERATE_2D else: return *_STAR_2D`.
  When `e == 0` the two eigenvalues are mathematically equal, so `np.linalg.eigvals`
  returns the *same* value for both a proper (star) node like `-I` and a defective
  (improper/degenerate) Jordan block like `[[-1,1],[0,-1]]`. Therefore `w[0] == w[1]`
  is effectively always true: the `*_STAR_2D` branch is unreachable dead code, and the
  reachable branch mislabels a true star node (`±I`) as "degenerate".
- Why it's a bug: A **star node** (proper node) has a full 2-D eigenspace — every
  direction is an eigenvector — and occurs iff `J` is a scalar multiple of the identity
  (`b == 0, c == 0, a == d`). A **degenerate/improper node** is defective (single
  eigenvector). The classifier swaps the two labels and never returns STAR.
- Repro: `stability_analysis([[-1.,0.],[0.,-1.]])` returns `'stable degenerate'`,
  but `-I` is the canonical *stable star*. Verified empirically.
- Fix: Distinguish a proper/star node by testing whether the off-diagonals vanish and
  diagonals are equal (scalar multiple of identity) → STAR; otherwise the repeated
  eigenvalue is defective → DEGENERATE. Applied for both stable (`p<0`) and unstable
  (`p>0`) branches.
- Tests: test_2d_stable_star_proper_node, test_2d_stable_degenerate_defective,
  test_2d_unstable_star_proper_node, test_2d_unstable_degenerate_defective
  (stability_test.py). Pre-existing coverage tests at
  stability_coverage_test.py:90-117 asserted the buggy `res in (DEGENERATE, STAR)`
  and a `2I`→DEGENERATE expectation; updated to assert the corrected labels (noted in
  test file).
- Status: fixed

### P13-H1 — `set_markersize` writes to a typo local, leaving module global stale  [Medium]
- File: brainpy/analysis/plotstyle.py:75-81
- Category: correctness
- What: `set_markersize` declares `global _markersize` then assigns to `__markersize`
  (double underscore typo). The intended module-level `_markersize` is never updated.
- Why it's a bug: The module global `_markersize` (used as the default for every
  `plot_schema` entry and any future-added entries) stays at its initial value of 10.
  `set_markersize` only mutates the already-built per-key dicts; any schema entry added
  after a call to `set_markersize` would silently use the stale default. The local
  `__markersize` is dead.
- Repro: `set_markersize(25); plotstyle._markersize` → still `10`. Verified empirically.
- Fix: assign to `_markersize` (the declared global).
- Tests: test_set_markersize_updates_global (plotstyle is exercised indirectly; added a
  direct regression in stability_coverage area is out of scope, so a focused test added
  to lowdim_phase_plane_coverage is avoided — instead a minimal test placed in a new
  plotstyle path). See test_set_markersize_updates_global.
- Status: fixed

### P13-H2 — `PhasePlane2D.plot_fixed_point(select_candidates='nullclines')` ignores fx-nullcline  [High]
- File: brainpy/analysis/lowdim/lowdim_phase_plane.py:328-329
- Category: correctness
- What: The `'nullclines'` branch gathers candidates with
  `key.startswith(C.fy_nullcline_points) or key.startswith(C.fy_nullcline_points)` —
  `fy` is repeated; the second clause was meant to be `C.fx_nullcline_points`.
- Why it's a bug: When the user requests fixed points using *both* nullclines, only the
  fy-nullcline candidate points are actually used. Fixed points that lie on the
  fx-nullcline candidate set (but were not in the fy set) are silently dropped, so the
  fixed-point search can miss real fixed points.
- Repro: static (matplotlib-driven; logic-level bug).
- Fix: change the second clause to `key.startswith(C.fx_nullcline_points)`.
- Tests: test_nullclines_select_uses_both (lowdim_phase_plane_coverage_test.py) —
  verifies the candidate union includes fx-nullcline points.
- Status: fixed

### P13-M1 — `find_fps_with_gd_method` mishandles single optimization loop / non-divisible num_opt  [Medium]
- File: brainpy/analysis/highdim/slow_points.py:379
- Category: edge/error
- What: `num_opt_loops = int(num_opt / num_batch)`. If `num_opt < num_batch` (e.g.
  `num_opt=50, num_batch=100`), `num_opt_loops == 0`, the optimization loop body never
  runs, `opt_losses` stays empty, and `jnp.concatenate(opt_losses)` raises
  "Need at least one array to concatenate".
- Why it's a bug: A perfectly reasonable call with a small `num_opt` crashes rather than
  running at least one batch.
- Repro: static (would require a full GD optimization run; logic-level).
- Fix: `num_opt_loops = max(1, int(np.ceil(num_opt / num_batch)))` so at least one batch
  is always executed.
- Tests: covered indirectly; a dedicated tiny GD run is added
  (test_gd_small_num_opt in slow_points_coverage_test.py).
- Status: fixed

### P13-M2 — `plot_vector_field(plot_method='streamplot')` crashes when user supplies `linewidth`  [Medium]
- File: brainpy/analysis/lowdim/lowdim_phase_plane.py:229-235
- Category: edge/error
- What: The streamplot branch does `linewidth = plot_style.get('linewidth', None)`
  (read, not remove) then calls `pyplot.streamplot(..., linewidth=linewidth, **plot_style)`.
  If the user passes `plot_style=dict(linewidth=...)`, `linewidth` is forwarded twice
  → `TypeError: got multiple values for keyword argument 'linewidth'`.
- Why it's a bug: A documented, supported `plot_style` key crashes the call.
- Repro: `PhasePlane2D(...).plot_vector_field(plot_method='streamplot', plot_style=dict(linewidth=1.0))`.
- Fix: use `plot_style.pop('linewidth', None)` so the key is consumed and passed once.
- Tests: test_pp2d_streamplot_custom_linewidth (lowdim_phase_plane_coverage_test.py) —
  the pre-existing `test_pp2d_streamplot_custom_linewidth_is_buggy` asserted the buggy
  TypeError; rewritten to assert the call now succeeds (noted: it asserted the fixed bug).
- Status: fixed

### P13-L1 — `stability_analysis` 3D fallthrough is partially dead / `assert` for validation  [Low]
- File: brainpy/analysis/stability.py:186-221
- Category: correctness/style
- What: When `is_real.sum() != 1` (e.g. all-complex pair count != expected), the function
  falls through to `eigenvalues = np.real(...)` only if the `is_real.sum()==1` branch is
  not taken; but the inner `if is_real.sum() == 1:` block always `return`s, so the
  trailing fallback (lines 217-221) only runs when `is_real.sum()` is 0 or >1 yet none
  of the complex sub-branches return. Also `assert np.conj(v1) == v2` uses `assert` for
  runtime validation (stripped under `-O`).
- Why it's a bug: Mostly latent; 3D classification is documented as best-effort. Not a
  default-path crash.
- Fix: recorded only (3D analyzer is rarely used; out of risk budget for this pass).
- Tests: none
- Status: recorded-only

### P13-L2 — `remove_return_shape` assumes `.shape` exists; scalars crash  [Low]
- File: brainpy/analysis/utils/function.py:38-44
- Category: edge/error
- What: `remove_return_shape` does `if r.shape == (1,)`. If the wrapped derivative returns
  a Python float (e.g. user-defined `f` returning a scalar), `r.shape` raises
  `AttributeError`. In practice derivative functions return arrays, so this is latent.
- Fix: recorded only.
- Tests: none
- Status: recorded-only

### P13-L3 — `get_args` time-variable detection breaks for keyword-only signatures with `t` absent  [Low]
- File: brainpy/analysis/utils/function.py:63-69
- Category: edge/error
- What: requires a parameter literally named `t`; this is by design for ODE integrators
  but the error message ("Do not find time variable 't'.") is raised generically.
- Fix: recorded only (matches integrator contract).
- Tests: none
- Status: recorded-only

### P13-L4 — `keep_unique` returns input `candidates` unconverted when `tolerance<=0`  [Low]
- File: brainpy/analysis/utils/others.py:127-130
- Category: style/consistency
- What: early returns hand back the raw `candidates` (possibly `bm.Array`/dict of) whereas
  the normal path returns numpy arrays. Callers (`SlowPointFinder.keep_unique`) then call
  `jnp.asarray` so it works, but the return dtype is inconsistent across branches.
- Fix: recorded only.
- Tests: none
- Status: recorded-only

---

## Cross-check vs dev/issues-found-20260618.md (analysis entries)

- **H-49** (`lowdim_analyzer.py:377,953`, `optimization.py:398` — arg-unwrap tests
  `isinstance(candidates, bm.Array)` instead of `isinstance(a, …)`): **already fixed** in
  the current tree (all three sites use `isinstance(a, bm.Array)`). No action.
- **H-50** (non-convertible 2D `_get_fixed_points` `jnp.concatenate([])` crash): **already
  fixed** (empty-guard at `lowdim_analyzer.py:1042-1047` returns correctly-shaped empties).
  No action.
- **M-33** (`stability.py` 2D star vs degenerate classification inverted): this is the same
  defect as **P13-C1** above — **fixed** (proper-node detected by scalar-multiple-of-identity
  test; STAR branch was previously unreachable).
- **M-34** (`stability.py:111-141` borderline center/saddle-node/line types gated on exact
  float `== 0` of autodiff Jacobians, almost never detected): **recorded-only**. Adding
  tolerance bands would change the classification of many near-borderline points and risks
  destabilising the existing 30 stability tests; out of risk budget for this pass.
- **M-35** (`slow_points.py` GD finder stops on *mean* loss while `tolerance` reads as
  per-point): **recorded-only**. This is an intentional aggregate stop criterion; per-point
  filtering is provided separately via `filter_loss`. Behaviour, not a crash/wrong-result.
- **L-15** (`get_sign2` passes a generator as `reshape` shape; helper currently unused):
  **recorded-only**. Latent and the JAX version in use consumes the generator without error.
</content>
