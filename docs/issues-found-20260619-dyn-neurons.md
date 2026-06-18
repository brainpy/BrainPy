# Audit findings — `brainpy/dyn/{neurons,channels,ions}` (2026-06-19)

Scope: `brainpy/dyn/neurons/{base,hh,lif}.py`, `brainpy/dyn/channels/*.py`,
`brainpy/dyn/ions/{base,calcium,potassium,sodium}.py` (+ co-located `*_test.py`).

Environment: brainpy 2.7.8, brainstate 0.5.1, brainunit 0.5.1, jax 0.10.2 (CPU).

## Cross-check status (vs `dev/issues-found-20260618.md`)

The neuron/channel entries from the prior audit are **already fixed in this worktree**:

- **C-14** (HH/Markov channel gating NaN at voltage singularity): FIXED. A branch-safe
  `_exprel` helper is present and used in `sodium.py`, `sodium_compatible.py`,
  `potassium.py`, `potassium_compatible.py`, and `calcium.py` (`ICaHT_Re1993`). Verified
  finite value *and* finite gradient at the singular voltages
  (`IK_HH1952v2(1).f_p_alpha([-55.0]) == 0.1`, not NaN). Regression tests already exist
  in `channels/dyn_channels_fixes_test.py`.
- **H-33** (`ions/base.py` `add_elem(k=v)` literal-keyword bug): FIXED
  (`ions/base.py:55` now `self.add_elem(**{k: v})`). Regression test exists.
- **H-34** (`ExpIFRef*` silently dropping `noise=`): FIXED (`lif.py:1113-1116` guards on
  `self.noise` and uses `sdeint`).
- **H-35** (`IzhikevichRef`/`GifRef` resetting state with grad-carrying `spike`): FIXED
  (`lif.py:3821-3831`, `4504-4522` now use `spike_no_grad` for every state reset;
  `AdExIFRef`/`AdQuaIFRef` likewise).
- **M-17** (`PotassiumFixed` `E` default): already `-95.` and asserted by a regression test.

Only one verified, still-present neuron/channel bug was found in scope (P8-H1, below),
plus several Low items recorded for documentation.

---

### P8-H1 — `CondNeuGroup` scales synaptic current by `1e-3/A`, double-/mis-scaling it  [High]
- File: `brainpy/dyn/neurons/hh.py:148-194` (`CondNeuGroupLTC.update` + `CondNeuGroup.update`/`derivative`)
- Category: correctness / numerics
- What: The conductance-based neuron converts the **external injected current** `x`
  (an absolute current, e.g. from `bp.inputs`, in nA) into a current *density* with the
  factor `x = x * (1e-3 / self.A)` inside `CondNeuGroupLTC.update`. Channel currents
  (`ch.current(V)`, already a density in µA/cm²) are correctly left unscaled in
  `derivative`. In the LTC class, **synaptic** currents (`sum_current_inputs`) are summed
  inside `derivative` and therefore also left unscaled (correct — they are densities).
  But `CondNeuGroup.update` (the non-LTC default class) folds `sum_current_inputs` into
  `x` *before* `super().update(x)` applies the `1e-3/A` factor, so the synaptic current
  is multiplied by `1e-3/A`.
- Why it's a bug: For any `A != 1e-3` (the default is `A=1e-3`, which masks the bug),
  synaptic input to a `CondNeuGroup` is silently rescaled relative to channel currents and
  relative to the otherwise-identical `CondNeuGroupLTC`. With `A=1.0` the synaptic drive is
  attenuated by 1000×.
- Repro (runtime, reproduced):
  ```python
  A = 1.0
  for cls in [bp.dyn.CondNeuGroupLTC, bp.dyn.CondNeuGroup]:
      neu = cls(1, A=A, IL=bp.dyn.IL(1, g_max=0.0, E=-70.))
      neu.reset_state()
      neu.add_inp_fun('syn', lambda V, init=0.: init + 10.0)   # constant synaptic density
      bp.share.save(t=0., dt=0.1, i=0)
      V0 = float(neu.V.value[0]); neu.update(0.); V1 = float(neu.V.value[0])
      print(cls.__name__, V1 - V0)     # expected dt*syn/C = 1.0
  # CondNeuGroupLTC -> 1.0     (correct)
  # CondNeuGroup    -> 9.99e-4 (wrong: scaled by 1e-3/A)
  ```
- Fix: In `CondNeuGroup`, evaluate the synaptic current at the pre-step `self.V` (preserving
  the non-LTC "evaluate synapses at the fixed membrane potential" semantics) but inject it
  into the derivative as an unscaled **density**, exactly like channel currents — instead of
  folding it into the pre-scaled external `x`. Implemented by stashing the precomputed
  synaptic density on the instance and adding it inside `CondNeuGroup.derivative`.
- Tests: `channels`/`neurons` — `hh_test.py::test_condneugroup_synaptic_current_scaling`
  (new) asserts `CondNeuGroup` and `CondNeuGroupLTC` give identical synaptic drive for
  `A != 1e-3`.
- Status: fixed

---

### P8-L1 — `ICaN_IS2008.derivative` variable names swapped (`phi_p` holds steady-state, `p_inf` holds tau)  [Low]
- File: `brainpy/dyn/channels/calcium.py:332-335`
- Category: style
- What: In `derivative`, the local `phi_p` actually holds the steady-state activation
  `p_inf(V)` and `p_inf` holds the time constant `tau_p(V)`; the returned value
  `self.phi * (phi_p - p) / p_inf` is numerically correct but the names are inverted and
  confusing.
- Why it's a bug: Readability/maintenance hazard only; math is correct.
- Repro: static.
- Fix: recorded only.
- Tests: none.
- Status: recorded-only

---

### P8-L2 — Pervasive non-rendering NumPy-doc section markers (`Parameters::`, `References::`, `See Also::`)  [Low]
- File: `brainpy/dyn/channels/*.py`, `brainpy/dyn/ions/*.py`, `brainpy/dyn/neurons/hh.py`
  (e.g. `sodium.py:88`, `potassium.py:95`, `calcium.py:92`, many more)
- Category: style/docs
- What: Docstrings use `Parameters::`/`References::`/`See Also::` (literal double-colon
  RST literal-block markers) instead of the NumPy-doc underline form. These will not render
  as proper sections in Sphinx/numpydoc and violate the project docstring style.
- Why it's a bug: Documentation only; no runtime effect.
- Repro: static.
- Fix: recorded only (out of risk budget; mechanical but very large surface).
- Tests: none.
- Status: recorded-only

---

### P8-L3 — `IAHP_De1994.reset_state` has a dead/duplicate assignment to `self.p`  [Low]
- File: `brainpy/dyn/channels/potassium_calcium_compatible.py:133`
- Category: style / dead code
- What: `self.p[:] = C2 / C3` is executed and then immediately overwritten by
  `self.p.value = bm.broadcast_to(C2 / C3, size)` at the end of the method. The first
  in-place assignment is redundant (and would also fail for a batched `C2/C3` whose shape
  differs from the unbatched `self.p`, though in practice `reset_state` is called before
  batching expansion).
- Why it's a bug: Dead code / minor fragility; behavior currently correct.
- Repro: static.
- Fix: recorded only.
- Tests: none.
- Status: recorded-only

---

### P8-L4 — `CalciumDyna.reset_state` passes `batch_size` as the `mode` positional of `variable(...)`  [Low]
- File: `brainpy/dyn/ions/calcium.py:133`
- Category: edge/api
- What: `variable(self._C_initializer, batch_size, self.varshape)` passes `batch_size`
  (an `int`/`None`/`Mode`) into the second positional parameter of `brainpy.initialize.variable`,
  whose signature is `variable(data, batch_or_mode, sizes, ...)`. This happens to work for the
  tested `int`/`None`/`Mode` values that `batch_or_mode` accepts, but is fragile and easy to
  misread.
- Why it's a bug: Works for current call sites; latent fragility only.
- Repro: static.
- Fix: recorded only.
- Tests: none.
- Status: recorded-only
