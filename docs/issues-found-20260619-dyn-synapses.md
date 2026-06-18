# P9 — dyn/synapses + dyn/projections audit (2026-06-19)

Branch: `fix/audit-20260619-dyn-synapses`
Scope: `brainpy/dyn/synapses/{abstract_models,bio_models,delay_couplings}.py`,
`brainpy/dyn/projections/{align_post,align_pre,base,conn,delta,inputs,plasticity,utils,vanilla}.py`
(+ co-located `*_test.py`).

Note on prior audit (`dev/issues-found-20260618.md`): several previously-reported
synapse/projection bugs are **already fixed in this tree** and were re-verified as
not-present: C-06/H-39 (STP facilitation `u` uses decayed locals — see the
documented comment + `u = u + pre_spike*U*(1-u); x = x - pre_spike*u*x`), C-17
(PoissonInput uses `scale=sqrt(b*p)` std, not variance), C-18
(`HalfProjAlignPost.update` reuses `current` instead of calling `comm` twice),
C-19/H-41 (`STDP_Song2000` passes `W_min/W_max=None` through unchanged and uses
`bm.as_jax` on traces), H-40 (`projections/base.py` now re-exports the real
`Projection`/`SynConn`). These are noted for the record only.

---

### P9-H1 — DualExpon raises ZeroDivisionError / yields NaN when `tau_rise == tau_decay`  [High]
- File: brainpy/dyn/synapses/abstract_models.py:159-164,265-266
- Category: numerics / edge / api-drift
- What: `_format_dual_exp_A` computes the peak normalizer
  `A = tau_decay/(tau_decay - tau_rise) * (tau_rise/tau_decay)**(tau_rise/(tau_rise-tau_decay))`.
  When `tau_rise == tau_decay` the leading factor divides by zero. For Python-float
  taus this raises `ZeroDivisionError` at construction; for array taus it silently
  produces `inf`, and `DualExpon`'s `a = (tau_decay-tau_rise)/(tau_rise*tau_decay)*A`
  then becomes `0*inf = nan`.
- Why it's a bug: equal rise/decay time constants is a legitimate, common request
  (the dual-exponential degenerates to the normalized alpha function). It is also the
  parameterization used by the `dynold` `AlphaCUBA/AlphaCOBA` compat classes (C-20),
  which route through `DualExpon`. A crash / silent NaN on a realistic default is wrong.
- Repro: `bp.dyn.DualExpon(2, tau_rise=10., tau_decay=10.)` -> `ZeroDivisionError`;
  `bp.dyn.DualExpon(2, tau_rise=bm.asarray([10.,5.]), tau_decay=bm.asarray([10.,50.]))`
  -> `a = [nan, 0.258...]`.
- Fix: compute `DualExpon.a` via the closed form `a = (1/tau_rise) *
  (tau_rise/tau_decay)**(tau_rise/(tau_rise-tau_decay))` (algebraically equal to the
  old expression but free of the `(tau_decay-tau_rise)` cancellation), and special-case
  the equal-tau limit element-wise to its L'Hôpital value `a = e/tau` (verified: a
  single-spike response then peaks at exactly 1.0). The `A is None` auto-normalizer path
  only; an explicitly supplied `A` is honoured unchanged.
- Tests: `abstract_models_test.py::TestDualExpon::test_equal_tau_no_crash`,
  `::test_equal_tau_matches_alpha`
- Status: fixed

### P9-H2 — DualExponV2 silently outputs all-zeros (or NaN) when `tau_rise == tau_decay`  [High]
- File: brainpy/dyn/synapses/abstract_models.py:159-164,413
- Category: numerics / edge
- What: `DualExponV2` stores `self.a = A` and returns `a * (g_decay - g_rise)`. With
  equal taus the two gates obey identical ODEs with identical inputs, so
  `g_decay - g_rise ≡ 0` for all time; the synapse is structurally singular. With the
  default `A=None` normalizer this additionally hits the same div-by-zero/`inf` as P9-H1,
  giving `inf*0 = nan`.
- Why it's a bug: the model produces a silently-dead (identically zero) or NaN synapse
  for an innocuous parameter choice, with no diagnostic. Unlike `DualExpon`, no finite
  `A` can recover a non-zero waveform, so the only correct behaviour is a clear error.
- Repro: `bp.dyn.DualExponV2(2, tau_rise=10., tau_decay=10.)` -> `ZeroDivisionError`;
  with an explicit `A=1.` the output is identically 0 over a full simulation.
- Fix: in `_format_dual_exp_A`, when `A is None` and `tau_rise == tau_decay`, raise a
  clear `ValueError` telling the user the dual-exponential normalizer is undefined for
  equal time constants and to use `brainpy.dyn.Alpha` (single-tau alpha synapse) instead.
  (Only the V2/auto-normalizer path reaches this branch; `DualExpon` computes `a`
  directly per P9-H1 and never calls the helper for the equal-tau case.)
- Tests: `abstract_models_test.py::TestDualExpon::test_v2_equal_tau_raises`
- Status: fixed

### P9-H3 — STP.reset_state crashes for per-neuron (array) `U`  [High]
- File: brainpy/dyn/synapses/abstract_models.py:855-858
- Category: correctness / edge
- What: `reset_state` does `self.u = self.init_variable(bm.ones, ...)` then
  `self.u.fill_(self.U)`. `fill_` requires a scalar fill value (`shape == ()`), so when
  `U` is a per-neuron array the call raises
  `MathError: The shape of the fill value must be ()`.
- Why it's a bug: heterogeneous release probability `U` across synapses is a standard,
  realistic short-term-plasticity configuration; the model crashes on construction.
- Repro: `bp.dyn.STP(3, U=bm.asarray([0.1, 0.2, 0.3]))` -> `MathError`.
- Fix: initialise `u` by broadcasting `U` into the (possibly batched) state:
  `self.u = self.init_variable(bm.ones, batch_or_mode); self.u.value = self.u.value * self.U`.
  Works for scalar `U`, array `U`, and batched modes.
- Tests: `abstract_models_test.py::TestSTP::test_array_U_reset`,
  `::test_array_U_run`
- Status: fixed

### P9-M1 — STP/STD discrete jumps assume binary `pre_spike`; graded inputs scaled wrongly  [Medium]
- File: brainpy/dyn/synapses/abstract_models.py:800-801,883-884
- Category: numerics
- What: the "simplified" updates `x = x - pre_spike*U*self.x` (STD) and
  `u = u + pre_spike*U*(1-u); x = x - pre_spike*u*x` (STP) reproduce the original
  `bm.where(pre_spike, ...)` exactly only for `pre_spike in {0,1}`. For graded
  (non-binary) presynaptic signals the depression/facilitation magnitude is scaled by
  the graded value, which is not the documented Tsodyks–Markram jump.
- Why it's a bug: callers feeding graded "spikes" get a different (linearly scaled)
  release, with no warning. (Matches prior-audit M-22.)
- Repro: static — compare `bm.where(pre>0, x-U*x, x)` vs `x-pre*U*x` for `pre=0.5`.
- Fix: recorded only. The simplified form is a defensible generalization, is faithful to
  the historical `dynold` STD/STP convention for the binary case, and changing it risks
  altering long-standing numeric behaviour. Documenting/asserting binary input is a
  cross-cutting API decision left to maintainers.
- Tests: none
- Status: recorded-only

### P9-M2 — STD applies depression jump to the pre-decay state, inconsistent with the STP fix  [Medium]
- File: brainpy/dyn/synapses/abstract_models.py:795-801
- Category: numerics
- What: `STD.update` integrates `x -> x_decayed` but then applies the release jump using
  the **pre-decay** Variable: `self.x.value = x_decayed - pre_spike*U*self.x`. The
  companion `STP` model was deliberately changed (prior-audit H-39, documented comment)
  to apply jumps to the **decayed** local `x`. STD was left on the pre-decay form, so the
  two short-term-plasticity models now discretize the spike jump inconsistently (off by
  one decay step in the jump term).
- Why it's a bug: an asymmetry/correctness smell; the decayed-local form is the cleaner
  discretization.
- Repro: static.
- Fix: recorded only. The pre-decay form matches the original commented code AND the
  `dynold` reference (`short_term_plasticity.py` STD), and the per-step difference is
  `O(dt/tau)` confined to the jump term. Changing it alters historical STD numerics for
  every user; flagged for a maintainer decision rather than fixed unilaterally.
- Tests: none
- Status: recorded-only

### P9-L1 — DelayCoupling docstrings reference a non-existent gain `g`; malformed `Parameters::`  [Low]
- File: brainpy/dyn/synapses/delay_couplings.py:131-163,234-256
- Category: style
- What: `DiffusiveCoupling`/`AdditiveCoupling` docstrings describe
  `coupling = g * (...)` but no `g` gain parameter exists (the gain is folded into
  `conn_mat`). They also use the literal-block `Parameters::` marker instead of a
  NumPy-doc underlined `Parameters` section (CLAUDE.md mandates NumPy-doc). Matches
  prior-audit L-09/L-14.
- Why it's a bug: misleading docs / Sphinx rendering.
- Fix: recorded only (Low).
- Tests: none
- Status: recorded-only

### P9-L2 — GABAa docstring lists wrong default alpha/beta  [Low]
- File: brainpy/dyn/synapses/bio_models.py:286-287
- Category: style
- What: the `Args` block says `alpha: Default 0.062` and `beta: Default 3.57`, but the
  constructor defaults are `alpha=0.53`, `beta=0.18`.
- Why it's a bug: documentation does not match code.
- Fix: recorded only (Low).
- Tests: none
- Status: recorded-only

### P9-L3 — SynConn.update raises with non-f-string message; conn_mat shape check on raw input  [Low]
- File: brainpy/dyn/projections/conn.py:119; delay_couplings.py:81
- Category: style / edge
- What: minor robustness/style items (e.g. `AdditiveCoupling.update`'s final
  `else: raise ValueError` has no message; `conn_mat.shape` is read before confirming
  the object exposes `.shape`).
- Fix: recorded only (Low).
- Tests: none
- Status: recorded-only
