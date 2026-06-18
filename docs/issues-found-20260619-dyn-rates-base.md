# Audit — dyn/rates + dyn/outs + dyn/others + dyn base/utils (P10)

Date: 2026-06-19
Branch: `fix/audit-20260619-dyn-rates-base`
Scope: `brainpy/dyn/rates/{nvar,populations,reservoir,rnncells}.py`,
`brainpy/dyn/outs/{base,outputs}.py`, `brainpy/dyn/others/{common,input,noise}.py`,
`brainpy/dyn/{base,utils,_docs}.py` (+ co-located `*_test.py`).

## Summary

A prior audit pass (2026-06-18, see `dev/issues-found-20260618.md`) already fixed
the bulk of the verified Critical/High bugs in this slice and left regression tests
in `brainpy/dyn/rates/dyn_rates_dynold_fixes_test.py`. Those fixes are present and
green in the current tree:

- C-15 `ThresholdLinearModel` `randn(*shape)` — fixed (`populations.py:1051,1060`).
- C-16 `StuartLandauOscillator.dy` `+w*x` rotational coupling — fixed (`populations.py:721`).
- H-36 `LSTMCell` `h`/`c` setters slice the last axis — fixed (`rnncells.py:401,412`).
- H-37 `Reservoir` recurrent noise `uniform(-1, 1)` — fixed (`reservoir.py:228`).
- H-38 `Reservoir` bias added in `update()` — fixed (`reservoir.py:223-224`).

This pass performs a fresh review and fixes the still-present
correctness/robustness issues (M-20, M-21), re-examines the previously-recorded
M-18/M-19 (both turn out NOT to be bugs in the current brainstate 0.5 stack), and
records remaining Low items.

---

### P10-H1 — `RNNCell`/`GRUCell`/`LSTMCell.reset_state()` crashes in default usage  [High]
- File: `brainpy/dyn/rates/rnncells.py:127, 239, 375`
- Category: edge/error, api-drift
- What: `reset_state` builds the state via
  `parameter(self._state_initializer, (batch_or_mode, self.num_out), allow_none=False)`.
  With the default `batch_or_mode=None` (the value supplied by `bp.reset_state(node)`
  and by a bare `node.reset_state()` in non-batching mode) the shape tuple becomes
  `(None, num_out)`, and `tools.size2num(None)` raises
  `ValueError: Do not support type <class 'NoneType'>: None`.
- Why it's a bug: `bp.reset_state(net)` / `node.reset_state()` is the standard reset
  API. Any network containing an unbatched `RNNCell`/`GRUCell`/`LSTMCell` crashes on
  reset. `__init__` already builds the state correctly with
  `variable(jnp.zeros, self.mode, self.num_out)`, which handles `None`; only
  `reset_state` regressed to `parameter((None, ...))`.
- Repro:
  ```python
  import brainpy as bp
  cell = bp.dyn.RNNCell(num_in=3, num_out=4)   # NonBatchingMode
  bp.reset_state(cell)                          # ValueError: ... None
  ```
- Fix: use `variable(self._state_initializer, batch_or_mode, self.num_out)` (matching
  `__init__`), which yields `(num_out,)` for `None`, `(B, num_out)` for an int `B`,
  and the mode-aware shape for a `Mode`. Applied to all three cells (LSTM uses
  `num_out * 2`).
- Tests: `test_rnn_cells_reset_state_none_unbatched`,
  `test_rnn_cells_reset_state_via_bp_reset_state`,
  `test_rnn_cells_reset_state_int_batch` in `rnncells_test.py`.
- Status: fixed

### P10-M1 — `ThresholdLinearModel` noise scales as `dt`, not `sqrt(dt)`  [Medium]
- File: `brainpy/dyn/rates/populations.py:1046-1062`
- Category: numerics
- What: the Euler update folds the Gaussian noise into the drift
  (`de += randn(*shape) * noise_e`, then `de = de / tau_e`, then
  `e = max(e + de * dt, 0.)`). The noise increment therefore scales linearly with
  `dt`. A correct Euler–Maruyama step for `tau de = (-e + beta·[I]_+) dt + noise·dW`
  needs the stochastic term to scale as `sqrt(dt)` (`dW ~ sqrt(dt)·N(0,1)`).
- Why it's a bug: the effective noise intensity is `dt`-dependent — halving `dt`
  changes the realized noise standard deviation by 2x instead of `sqrt(2)`, so the
  stationary statistics of the simulated rate change with the integration step.
  Measured: noise std ratio for `dt=0.1` vs `dt=0.01` is exactly `10` (the `dt`
  scaling); the correct Euler–Maruyama ratio is `sqrt(10) ≈ 3.16`.
- Repro: static + measured (see commit message / regression test).
- Fix: move the noise out of the `dt`-scaled drift and add it as a separate
  `sqrt(dt)` Euler–Maruyama increment:
  `e = max(e + (-e + beta_e·[I]_+)/tau_e · dt + noise_e/tau_e · sqrt(dt)·randn, 0)`.
- Tests: `test_threshold_linear_model_noise_scales_as_sqrt_dt` in `rates_test.py`.
- Status: fixed

### P10-L1 — `FeedbackFHN.reset_state` rebinds `self.input`/`input_y` instead of `.value=`  [Low] (recorded only — was M-18)
- File: `brainpy/dyn/rates/populations.py:370-371`
- Category: style
- What: `reset_state` does `self.input = variable(...)` / `self.input_y = variable(...)`
  whereas the sibling rate models (`FHN`, `QIF`, `StuartLandau`, `WilsonCowan`) use
  `self.input.value = ...`.
- Why not a bug here: under brainstate 0.5, assigning a fresh `Variable` to an
  attribute that already holds a `State` performs an in-place value/shape update
  (object identity is preserved, value resets, batched reshape works), so captured
  references and monitors are not broken. Verified empirically.
- Fix: recorded only (consistency nit; out of Critical/High/Medium scope).
- Status: recorded-only

### P10-L2 — Prior audit M-19 (`FeedbackFHN` delay "double-count") is not a bug  [Low] (recorded only)
- File: `brainpy/dyn/rates/populations.py:374`
- Category: correctness (false positive in prior audit)
- What: 2026-06-18 M-19 claimed that because `state_delays={'x': self.x_delay}` is
  registered with the integrator, querying `self.x_delay(t - self.delay)` in `dx`
  double-counts the delay and should be `self.x_delay(t)`.
- Why it's not a bug: `state_delays` only causes the integrator to call
  `delay.update(new_x)` after each step (buffer maintenance). `TimeDelay.__call__`
  takes an **absolute time** (see its docstring: `delay(-0.5)` → value at t=-0.5), so
  `x_delay(t - delay)` is the correct way to read `x(t - delay)`. Querying
  `x_delay(t)` would return the *current* value (no delay) and destroy the feedback.
  Verified empirically: the query returns the historical value ~`delay` ms in the
  past, not the current value.
- Fix: recorded only — leave `x_delay(t - self.delay)` as-is. Changing it (per the
  earlier audit) would introduce a regression.
- Status: recorded-only

### P10-L3 — `OutputGroup.reset_state` signature uses `batch_size` not `batch_or_mode`  [Low] (recorded only)
- File: `brainpy/dyn/others/input.py:102`
- Category: style
- What: `OutputGroup.reset_state(self, batch_size=None, ...)` while the rest of the
  module (`InputGroup`, `SpikeTimeGroup`, `PoissonGroup`) uses `batch_or_mode`. The
  body is a no-op `pass`, so callers passing positionally still work; no functional
  impact.
- Fix: recorded only.
- Status: recorded-only

### P10-L4 — NumPy-doc nonconformance across rates/outs docstrings  [Low] (recorded only)
- File: `brainpy/dyn/rates/populations.py` (and `nvar.py`, `reservoir.py`,
  `outs/outputs.py`), e.g. `Parameters::`, `References::`, `See Also::` literal-block
  markers and bare `Reference` headings.
- Category: style
- What: CLAUDE.md mandates underlined NumPy-doc sections; these files use the legacy
  `Section::` literal-block form (matching the rest of the repo, also flagged as L-14
  in the 2026-06-18 audit).
- Fix: recorded only (repo-wide cosmetic; out of scope).
- Status: recorded-only

---

## Verified-correct (checked, no change)

- `NVAR` feature construction: stride/`select_ids` picks exactly `delay` time points;
  monomial `comb_ids` and constant/linear concatenation correct (matches 2026-06-18
  Appendix B).
- `Reservoir` spectral-radius rescaling (`Wrec *= spectral_radius / current_sr`) is
  applied after connectivity masking and before sparse reduction — correct ordering.
- `QIF` / `FHN` / `WilsonCowan` ODE right-hand sides match their docstrings.
- `MgBlock` magnesium curve `1/(1 + [Mg]/β·exp(α(V_off - V)))` matches the documented
  `g_inf`; `COBA`/`CUBA` outputs correct.
- `OUProcess` uses `sdeint` with constant diffusion `g = sigma` → correct `sqrt(dt)`
  scaling; `reset_state` initializes `x` at `mean`.
- `PoissonGroup` spike probability `freqs · dt / 1000` (Hz·ms) correct.
- `LSTMCell`/`GRUCell` gate equations match docstrings (forget-gate `+1` bias; GRU
  reset/update split) — GRU confirmed correct by 2026-06-18 Appendix B.
- `get_spk_type` mode → dtype mapping correct.
