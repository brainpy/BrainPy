# BrainPy `dynold` audit — 2026-06-19 (P11)

Scope: legacy/backward-compat dynamics package `brainpy/dynold` (old-API neurons,
synapses, synaptic outputs, plasticity, learning rules).

Severity scale: Critical (silently wrong/crash in common usage) · High (wrong in
realistic cases / broken public API) · Medium (edge cases, fragility,
error-handling) · Low (style/docs/typing/dead code, recorded only).

---

### P11-C1 — experimental `STP.reset_state` swaps `variable_` arguments → crash on construction  [Critical]
- File: brainpy/dynold/experimental/syn_plasticity.py:169-170
- Category: api-drift / correctness
- What: `reset_state` calls `variable_(jnp.ones, batch_size, self.num)` and
  `variable_(OneInit(self.U), batch_size, self.num)`. The `variable_` signature is
  `variable_(init, sizes, batch_or_mode, ...)`, i.e. `sizes` comes **before**
  `batch_or_mode`. The sibling `STD.reset_state` (line 91) uses the correct order
  `variable_(jnp.ones, self.num, batch_size)`.
- Why it's a bug: `__init__` calls `self.reset_state(self.mode)`. The `Mode` object
  lands in the `sizes` slot, and `to_size(NonBatchingMode)` raises
  `ValueError: Cannot make a size for NonBatchingMode`. `brainpy.dynold.experimental
  .syn_plasticity.STP` is therefore **completely unconstructable** with default mode.
- Repro:
  ```python
  from brainpy.dynold.experimental.syn_plasticity import STP
  STP(pre_size=5)  # ValueError: Cannot make a size for NonBatchingMode
  ```
- Fix: swap to `variable_(jnp.ones, self.num, batch_size)` and
  `variable_(OneInit(self.U), self.num, batch_size)`, mirroring `STD.reset_state`.
- Tests: experimental/syn_plasticity_test.py::TestSTP::test_stp_construction_ok,
  test_stp_update_state_changes (pre-existing `test_stp_construction_is_broken`
  retargeted — it asserted the bug being fixed).
- Status: fixed

---

### P11-H1 — experimental sparse synapse paths call `csrmv(..., method='cusparse')` → TypeError  [High]
- File: brainpy/dynold/experimental/abstract_synapses.py:144-150 (Exponential, sparse+stp),
  291-299 (DualExponential, sparse always)
- Category: api-drift / edge
- What: Both paths call `bm.sparse.csrmv(data, indices, indptr, s, shape=..., transpose=True, method='cusparse')`.
  The current `bm.sparse.csrmv` signature is
  `csrmv(data, indices, indptr, vector, *, shape, transpose=False)` — there is no
  `method` keyword. Passing it raises `TypeError: csrmv() got an unexpected keyword
  argument 'method'`.
- Why it's a bug: `Exponential(comp_method='sparse', stp=...)` and **any**
  `DualExponential(comp_method='sparse')` crash on first `update`. The modern
  `dnn.linear.CSRLinear` calls `csrmv` without `method`.
- Repro:
  ```python
  conn = bp.conn.FixedProb(0.5)(pre_size=5, post_size=4)
  syn = asyn.DualExponential(conn, comp_method='sparse')
  share.save(t=0.0, dt=bm.get_dt()); syn.update(bm.ones(5))  # TypeError
  ```
- Fix: drop the `method='cusparse'` kwarg in both call sites.
- Tests: experimental/abstract_synapses_test.py — retargeted the two pinned
  `*_defect` tests (`TestExponential.test_sparse_with_stp_defect`,
  `TestDualExponential.test_sparse_defect`) to assert correct output shape.
- Status: fixed

---

### P11-H2 — experimental `Exponential` sparse+stp path drops STP filtering  [High]
- File: brainpy/dynold/experimental/abstract_synapses.py:143-153
- Category: correctness
- What: When `stp is not None`, the method computes
  `syn_value = self.stp(pre_spike) * pre_spike` (line 123), but in the sparse branch
  it then evaluates `post_vs = f(pre_spike)` (line 153), feeding the **raw**
  `pre_spike` into `csrmv` instead of the STP-filtered `syn_value`. All other layouts
  (All2All / One2One / dense) correctly use `syn_value`.
- Why it's a bug: short-term plasticity has no effect on the conductance for the
  sparse layout — the synapse behaves as if no STP were attached, silently producing
  wrong currents.
- Repro: static (path was previously masked by the `method=` TypeError in P11-H1).
- Fix: pass `syn_value` to `f` in the sparse-with-stp branch (`post_vs = f(syn_value)`).
- Tests: covered by retargeted `TestExponential.test_sparse_with_stp_defect` (now
  asserts a finite, shape-correct output through the stp+sparse path).
- Status: fixed

---

### P11-M1 — STD / STP discrete jumps read pre-decay state (off-by-one decay)  [Medium]
- File: brainpy/dynold/synplast/short_term_plasticity.py:98 (STD), 194-195 (STP)
- Category: numerics / correctness
- What: The Tsodyks–Markram jumps must use the value *at spike arrival*, i.e. the
  state integrated forward to time `t` (the local `x`/`u`), not the previous step's
  stored Variable. STD does `self.x.value = where(spike, x - U*self.x, x)` — the
  depression term uses `self.x` (pre-decay) instead of the decayed local `x`. STP
  does `u = where(spike, u + U*(1 - self.u), u)` and `x = where(spike, x - u*self.x, x)`
  — both facilitation and depression read pre-decay `self.u`/`self.x`.
- Why it's a bug: each spike applies the jump to a slightly stale value; the error per
  step is `U·(recovery over dt)` and accumulates across spike trains. Correct discrete
  Tsodyks–Markram is `u^+ = u^- + U(1-u^-)`, `x^+ = x^- - u^+ x^-` with `u^-,x^-`
  the *decayed* (current-time) values. Mirrors dyn-synapses M-22/H-39.
- Repro: static (sub-dt drift; visible over long facilitating trains).
- Fix: use the decayed locals — STD: `x - self.U * x`; STP: `u + self.U*(1 - u)`
  then `x - u * x`.
- Tests: synplast/short_term_plasticity_test.py::test_std_jump_uses_decayed_state,
  test_stp_jump_uses_decayed_state.
- Status: fixed

---

### P11-L1 — missing `raise` before `ValueError` in dead error branches  [Low]
- File: brainpy/dynold/synapses/base.py:233 (`ValueError(f'Unknown sparse data type...')`),
  brainpy/dynold/experimental/base.py:98 (same)
- Category: edge/error
- What: `ValueError(...)` is constructed but not raised, so an unknown
  `sparse_data`/`data_if_sparse` silently falls through with `conn_mask` unbound (would
  later `UnboundLocalError`) rather than giving the intended diagnostic.
- Why it's a bug: error-handling gap; unreachable in default usage because the value is
  validated earlier in the same function, hence Low.
- Fix: recorded only (prepend `raise`).
- Status: recorded-only

---

### P11-L2 — dynold `LIF` wrapper inherits modern `LifRef` defaults (V_rest=0, V_reset=-5, V_th=20)  [Low]
- File: brainpy/dynold/neurons/reduced_models.py:155-240 (LIF forwards `*args/**kwargs`
  to `lif.LifRef`)
- Category: api-drift / docs
- What: Historical brainpy-2.x `LIF` defaulted to `V_rest=-65, V_reset=-65, V_th=-50`
  (cf. ExpIF/AdExIF tables in the same module). The modern `LifRef` it now forwards to
  uses `V_rest=0, V_reset=-5, V_th=20`. The dynold `LIF` docstring lists the parameters
  without a default table, so there is no in-file contradiction, but behaviour silently
  changed vs the pre-3.0 API. (ExpIF/AdExIF `LifRef` defaults now match their docstring
  tables, so no drift there.)
- Why it's a bug: silent default drift for legacy callers; recorded as Low because the
  wrapper intentionally delegates to the modern model and changing the default would
  itself be a cross-cutting behaviour change best owned by the dyn-neurons package.
- Fix: recorded only (would need either historical defaults in the dynold wrapper or a
  docstring/changelog note; cross-cuts dyn-neurons defaults).
- Status: recorded-only

---

### P11-L3 — `_STPModel.update` recomputes spike-gated current; doc/clarity  [Low]
- File: brainpy/dynold/synapses/learning_rules.py:39-47
- Category: style
- What: C-21 (2026-06-18 audit) — STP learning rule injecting current with zero
  presynaptic spikes — is **already fixed** here: `_STPModel.update` gates the injected
  amplitude as `self[1](pre_spike * ux)`. Verified: a no-input run keeps `syn.I == 0`.
  Recorded for traceability only; no further change.
- Status: recorded-only

---

## Cross-check vs dev/issues-found-20260618.md (dynold entries)
- C-17 (PoissonInput Gaussian uses variance as std) — already fixed in
  `experimental/others.py` (`scale = bm.sqrt(b * p)`), verified present.
- C-20 (AlphaCUBA/COBA ZeroDivisionError) — already fixed in `synapses/compat.py`
  (routes through single-tau `Alpha`), verified present.
- C-21 (dynold STP learning rule injects current at rest) — already fixed in
  `synapses/learning_rules.py` (`pre_spike * ux` gating), verified: no-input run →
  `syn.I == 0`. See P11-L3.
- M-24 (ALIFBellec2020 `a_initializer=OneInit(-50.)`) — see note below.
- L-16 (missing `raise` before `ValueError`) — see P11-L1.

### P11-M2 — `ALIFBellec2020` / `LIF_SFA_Bellec2020` default `a_initializer=OneInit(-50.)`  [Medium]
- File: brainpy/dynold/neurons/reduced_models.py:1312 (ALIFBellec2020),
  :1475 (LIF_SFA_Bellec2020)
- Category: correctness
- What: The adaptation variable `a` (governed by `tau_a da = -a`, incremented by 1 per
  spike, contributing `beta*a` to the effective threshold `V_th + beta*a`) started at
  `-50` by default. Physically the SFA adaptation should start at rest (~0); a `-50`
  start lowers the effective threshold by `beta*(-50) = -80 mV` for thousands of ms
  (tau_a=2000), making the neuron transiently hyper-excitable / spuriously firing from a
  cold start. `ALIFBellec2020` is publicly exported (`bp.neurons.ALIFBellec2020`).
- Why it's a bug: surprising, unphysical default that biases the first simulation.
  Verified M-24 in the prior (2026-06-18) audit.
- Repro: static (effective threshold offset persists for ~tau_a ms after reset).
- Fix: default `a_initializer=ZeroInit()` for both Bellec SFA models.
- Tests: neurons/reduced_neurons_test.py::TestBellecAdaptation::
  test_default_adaptation_starts_at_rest (both models).
- Status: fixed
