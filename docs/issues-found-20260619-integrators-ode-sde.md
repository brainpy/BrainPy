# Integrators (ODE/SDE) Audit — Issues Found (2026-06-19)

**Reviewer role:** Senior numerical-methods + JAX expert (ODE/SDE solvers)
**Scope:** `brainpy/integrators/{base,constants,joint_eq,runner,utils}.py`,
`brainpy/integrators/ode/{adaptive_rk,base,common,explicit_rk,exponential,generic}.py`,
`brainpy/integrators/sde/{base,generic,normal,srk_scalar}.py` (+ co-located `*_test.py`).
**Environment (verified):** Python 3.13 · jax 0.10.2 · brainpy 2.7.8 (CPU). `import brainpy` works; high-severity
findings are runtime-reproduced via empirical convergence-order tests.

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 1 |
| High     | 1 |
| Medium   | 0 |
| Low      | 5 |

The two correctness bugs are both **silent**: the integrators run and produce finite numbers, but the
numerical *order of accuracy* collapses. They were found by measuring strong/weak convergence rates against
SDEs/ODEs with known exact solutions, not by inspection alone.

Prior-audit cross-check (`dev/issues-found-20260618.md`): C-12 (adaptive-RK `tol` default) and C-13
(SDE `NameError` on bad `intg_type`/Heun guard) are **already fixed** in this worktree — verified by repro
(see P6-L5).

---

### P6-C1 — RKF45 last node `c6 = 1/3` instead of `1/2`: method degrades to order 1  [Critical]
- File: brainpy/integrators/ode/adaptive_rk.py:322
- Category: numerics
- What: `RKF45.C = [0, 0.25, 0.375, '12/13', 1, '1/3']`. The 6th stage node must be `c6 = 1/2`
  (its `A`-row `(-8/27, 2, -3544/2565, 1859/4104, -11/40)` sums to `1/2`, and the rendered docstring
  tableau line shows `1/2`). The code evaluates stage `k6` at `t + dt*(1/3)` instead of `t + dt*(1/2)`.
- Why it's a bug: The Runge–Kutta consistency condition `sum_j a_{ij} = c_i` is violated for row 6.
  Whenever the derivative `f` depends on time `t`, the order conditions break and the 5th-order
  `B1` solution collapses to **first order**. The companion error estimate is also wrong, so adaptive
  step control is misled. For autonomous `f` (no `t`) the node is unused, which is why existing
  smoke tests (autonomous / `f=-x`, Lorenz) never caught it. This is the default behavior of
  `odeint(..., method='rkf45')`.
- Repro (measured, `dy/dt = cos(t)`, `y(0)=0`, exact `sin(t)`):
  ```
  c6=1/3 (current): errors per halving 4.2e-3 -> 2.1e-3 -> 1.1e-3 -> 5.3e-4   order ~ 1.0
  c6=1/2 (correct): errors per halving 9.3e-7 -> 2.9e-8 -> 9.0e-10 -> 2.8e-11 order ~ 5.0
  ```
- Fix: `C = [0, 0.25, 0.375, '12/13', 1, 0.5]`.
- Tests: `ode/ode_method_adaptive_rk_test.py::TestRKF45NodeFix` (order-5 convergence on time-dependent ODE).
- Status: fixed

### P6-H1 — `KlPl` SDE final-stage diffusion weights wrong: method does not converge  [High]
- File: brainpy/integrators/sde/srk_scalar.py:373-374
- Category: numerics
- What: The KlPl (Kloeden–Platen, strong order 1.0, scalar Wiener) final stage uses
  `g1 = -I1 + I11/dt_sqrt + I10/dt`, `g2 = I11/dt_sqrt`. The correct order-1.0 SRK diffusion update is
  `g1 = I1 - I11/dt_sqrt`, `g2 = I11/dt_sqrt` (so the total diffusion weight `g1+g2 = I1`, reproducing
  `g·ΔW` for additive noise). The current code has the sign of `I1` flipped, the sign of `I11/dt_sqrt`
  flipped, and a spurious `+ I10/dt` term. The in-code comment at line 369 (`g1 = (I1 - I11/dt_sqrt + I10/dt)`)
  also disagrees with the code and is itself only partly right (the `I10/dt` should not be there).
- Why it's a bug: The lowest-order consistency requirement for the diffusion is `sum_i (weight_i) = I1`
  when `g` is constant. Current weights sum to `-I1 + 2·I11/dt_sqrt + I10/dt ≠ I1`, so the leading
  `g·ΔW` term is wrong and the scheme does **not** converge to the true solution.
- Repro (measured, geometric Brownian motion `dX = aX dt + bX dW`, exact `X0 exp((a-b²/2)T + bW)`,
  same driving path, mean abs error over 300 paths):
  ```
  n= 50  KlPl_current = 2.7e-1   KlPl_fixed = 2.1e-3
  n=100  KlPl_current = 2.6e-1   KlPl_fixed = 1.1e-3
  n=200  KlPl_current = 3.1e-1   KlPl_fixed = 5.4e-4   (fixed halves each doubling -> order ~1.0)
  n=400  KlPl_current = 2.9e-1   KlPl_fixed = 2.7e-4
  ```
  Current error is flat (~0.3) regardless of step size — no convergence.
- Fix: `g1 = {var}_I1 - {var}_I11/dt_sqrt`, `g2 = {var}_I11 / dt_sqrt`. Comment updated to match.
- Tests: `sde/srk_scalar_test.py::TestKlPlConvergence` (strong-order-1.0 convergence on linear SDE).
- Status: fixed

---

## Low (recorded only — not fixed per instructions)

### P6-L1 — Dead helper functions in `sde/normal.py`  [Low]
- File: brainpy/integrators/sde/normal.py:37-60
- Category: style
- What: `df_and_dg`, `dfdt`, `noise_terms` are module-level helpers that are never called (the SDE
  integrator classes build their steps inline with `bm.random.randn`). `noise_terms` even references a
  `random.normal(...).value` API and a code-generation style no longer used.
- Why it's a bug: Dead code; misleading (`noise_terms` looks like the noise generator but is unused).
- Fix: recorded only.
- Status: recorded-only

### P6-L2 — `SRK2W1` stage-4 drift uses `f_H0s1` where tableau/comment imply `f_H0s3`  [Low]
- File: brainpy/integrators/sde/srk_scalar.py:293 (comment at :289 says `0.25 * f_H0s3`)
- Category: numerics
- What: `H1s4 = x + 0.25*dt*f_H0s1 + dt_sqrt*(2*g_H1s1 - g_H1s2 + 0.5*g_H1s3)`. The drift uses
  `f_H0s1`, but the comment (`H1s4 = x + dt * 0.25 * f_H0s3 + ...`) and the `A^(0)` tableau row
  (nonzero entry in the third drift column) call for `f_H0s3`.
- Why it's *Low*: Empirically both variants retain strong order ≈1.5 on a linear SDE (errors
  4.1e-5/3.5e-5 at n=50, both ~order 1.5); the difference is within the method's own error constant
  because the `H1s4` stage only feeds the `g4 = I111/dt` weight, an `O(h^1.5)` term. No order loss in
  the tested regime, so recorded only (Low, not fixed).
- Fix: recorded only (would change `f_H0s1` -> `f_H0s3` at line 293 to match the published tableau).
- Status: recorded-only

### P6-L3 — Adaptive step controller exponent fixed at 0.2 regardless of method order  [Low]
- File: brainpy/integrators/ode/adaptive_rk.py:226
- Category: numerics
- What: `factor = 0.9 * (tol / (error + 1e-12)) ** 0.2`. The exponent `0.2 = 1/5` is appropriate for an
  order-4/5 pair (RKF45, DOPRI, CashKarp) but is used unchanged for `RKF12`/`HeunEuler` (order 1/2),
  where `1/(p+1) = 1/2` would be the standard PI-free controller exponent.
- Why it's *Low*: Step-size control efficiency only; results stay correct (controller still
  shrinks/grows toward tolerance), just sub-optimal step selection for the low-order pairs.
- Fix: recorded only.
- Status: recorded-only

### P6-L4 — Docstring tableau typo in `BogackiShampine` (`4/90` should be `4/9`)  [Low]
- File: brainpy/integrators/ode/adaptive_rk.py:453
- Category: style
- What: Rendered docstring tableau shows `2/9 & 1/3 & 4/90` for the B1 row; the code `B1 = ['2/9','1/3','4/9',0]`
  is correct. Pure doc typo.
- Why it's *Low*: Documentation only; code is right.
- Fix: recorded only.
- Status: recorded-only

### P6-L5 — Prior-audit C-12 / C-13 already fixed (verification note)  [Low / informational]
- Files: brainpy/integrators/ode/adaptive_rk.py:70,187 ; brainpy/integrators/sde/base.py:20 ; sde/normal.py:20,225
- Category: correctness (informational)
- What: The 2026-06-18 audit reported `TypeError` on `odeint(method='rkf45', adaptive=True)` with no `tol`
  (C-12) and `NameError: errors` on bad `intg_type` / the `Heun` Ito guard (C-13). In this worktree
  `adaptive_rk.py` already imports `from brainpy import _errors as errors` and sets
  `code_scope['tol'] = self.tol`, and `sde/base.py` / `sde/normal.py` already import `errors`.
  Verified by repro: `odeint(f, method='rkf45', adaptive=True, dt=0.1)(1.,0.)` returns a value + new dt
  (no crash); the still-broken numeric is the unrelated `c6` node (P6-C1).
- Status: recorded-only (already-fixed elsewhere; no action)
