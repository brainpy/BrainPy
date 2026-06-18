# P12 — `brainpy/dnn` expert review (2026-06-19)

Branch: `fix/audit-20260619-dnn`. Scope: `brainpy/dnn/{activations,base,conv,dropout,function,interoperation_flax,linear,normalization,pooling}.py` + co-located tests.

Environment: jax 0.10.2, brainstate 0.5.1, braintools 0.1.10, brainunit 0.5.1 (CPU).

Severity legend: Critical (silently wrong/crash in default usage), High (wrong in realistic cases / broken public API), Medium (edge/fragility/error-handling), Low (style/docs — record only).

---

### P12-M1 — `LayerNorm` shape-mismatch error does `", ".join(<ints>)` → masks real error with `TypeError`  [Medium]
- File: brainpy/dnn/normalization.py:536
- Category: edge/error
- What: When the trailing input dims do not match `normalized_shape`, the guard raises
  `ValueError(f'... (..., {", ".join(self.normalized_shape)}), but we got {x.shape}')`.
  `self.normalized_shape` is a tuple of `int`, so `", ".join(...)` raises
  `TypeError: sequence item 0: expected str instance, int found` instead of the intended `ValueError`.
- Why it's a bug: The user-facing diagnostic is replaced by an opaque `TypeError`, hiding the actual
  shape problem. Any wrong-shape input to an affine/non-affine `LayerNorm` hits this.
- Repro:
  ```python
  bp.dnn.LayerNorm(10)(bm.random.randn(2, 5, 8))  # -> TypeError, not ValueError
  ```
- Fix: `", ".join(map(str, self.normalized_shape))` (and format the expected shape readably).
- Tests: `normalization_test.py::Test_Normalization::test_LayerNorm_shape_mismatch_raises_valueerror`
- Status: fixed
- (== prior audit M-26)

---

### P12-M2 — Pooling rejects the leftmost negative `channel_axis` (`channel_axis == -x_dim`)  [Medium]
- File: brainpy/dnn/pooling.py:118 (`Pool._infer_shape`), 390 (`_MaxPoolNd._infer_shape`), 787 (`AdaptivePool.update`)
- Category: edge/error
- What: The bound check is `if channel_axis and not 0 <= abs(channel_axis) < x_dim: raise`.
  Using `abs(channel_axis)` makes `channel_axis == -x_dim` (the valid leftmost axis, e.g. `-3` for a
  3-D `(C, H, W)` input) fail the check and raise `ValueError`.
- Why it's a bug: A legitimate, common channels-first layout (`channel_axis=-3` on `(C,H,W)`, or
  `-4` on `(N,C,H,W)`) is wrongly rejected. The correct numpy-style bound is `-x_dim <= axis < x_dim`.
- Repro:
  ```python
  bp.dnn.MaxPool2d(2, channel_axis=-3)(bm.random.randn(6, 4, 4))  # ValueError: Invalid channel axis -3
  ```
- Fix: Replace the `abs()` bound test with `not -x_dim <= channel_axis < x_dim` in all three sites.
- Tests: `pooling_layers_test.py::TestPoolingChannelAxis::test_maxpool2d_leftmost_negative_channel_axis`,
  `...::test_adaptiveavgpool2d_leftmost_negative_channel_axis`
- Status: fixed
- (== prior audit M-27)

---

### P12-M3 — `BatchNorm` stores the *biased* batch variance into `running_var` (PyTorch uses unbiased)  [Medium]
- File: brainpy/dnn/normalization.py:172-174
- Category: numerics
- What: In fit mode the running estimate is updated with the biased population variance
  `var = mean_of_square - mean**2` (divisor `N`). PyTorch / the conventional BatchNorm running statistic
  uses the *unbiased* sample variance (divisor `N-1`, i.e. Bessel's correction) for the running buffer,
  while keeping the biased variance only for normalizing the current batch.
- Why it's a bug: The running variance used at eval time is systematically too small by a factor of
  `(N-1)/N`. For small batch/window counts `N` this is a meaningful (a few percent) bias in the
  eval-time normalization — i.e. inference results drift from the trained reference.
- Repro:
  ```python
  bn = bp.dnn.BatchNorm1d(3, affine=False); bp.share.save(fit=True)
  bn(bm.random.randn(4, 5, 3))           # N = 20
  # running_var == 0.99*1 + 0.01*biased_var, not unbiased_var
  ```
- Fix: Scale the variance fed into the `running_var` EMA by `N/(N-1)` (with `N` = number of reduced
  elements), guarding `N == 1`. The batch normalization itself keeps the biased `var`.
- Tests: `normalization_test.py::Test_Normalization::test_BatchNorm_running_var_is_unbiased`
- Status: fixed
- (== prior audit M-25)

---

### P12-L1 — `Flatten` default `start_dim=0` contradicts its docstring example and PyTorch (`start_dim=1`)  [Low]
- File: brainpy/dnn/function.py:91 (default), 74-87 (docstring example)
- Category: api-drift/style
- What: `Flatten.__init__` defaults `start_dim=0`. The class docstring example claims
  `Flatten()` on `(32, 1, 5, 5)` yields `(32, 25)` (PyTorch's `start_dim=1` semantics), but in
  `NonBatchingMode` the actual default flattens the batch dim too → `(800,)`.
- Why it's a bug: The documented contract and the implemented default disagree. PyTorch's `nn.Flatten`
  default is `start_dim=1`.
- Repro:
  ```python
  bp.dnn.Flatten()(bm.random.randn(32, 1, 5, 5)).shape   # (800,), docstring says (32, 25)
  ```
- Fix: recorded only. NOTE: changing the default to `1` would change the documented `NonBatchingMode`
  contract and break `function_test.py::test_flatten_non_batching_mode` (asserts `(600,)` from default
  start_dim under `NonBatchingMode`). The discrepancy is documentation/default-value drift, not a
  silent numeric error, and a default change is a cross-cutting API change. Left for maintainers to
  decide (fix docs vs. change default + migrate the test).
- Tests: none
- Status: recorded-only

---

## Cross-check vs `dev/issues-found-20260618.md` (dnn entries)

- **C-05** (`GroupNorm`/`InstanceNorm` reduce over the group axis): **already fixed** in this tree
  (`normalization.py:640` reduces over spatial + within-group channel axis, keeps the group axis).
  Verified: `GroupNorm(3,6) != GroupNorm(1,6) != GroupNorm(6,6)`, per-group means ≈ 0. No action.
- **H-51** (`BatchNorm`/affine `LayerNorm`/`GroupNorm` crash out-of-the-box under default mode):
  **already fixed** (`BatchNorm` defaults to `training_mode`; affine params wrapped as `Variable` vs
  `TrainVar` per mode instead of a hard assert). No action.
- **M-25**: still present → fixed here as **P12-M3**.
- **M-26**: still present → fixed here as **P12-M1**.
- **M-27**: still present → fixed here as **P12-M2**.
- **M-28** (`Flatten` default): still present → recorded as **P12-L1** (see note; default change out of safe scope).

## Checked and found correct (no action)
- `Dropout`: `prob` = keep-probability; `bernoulli(prob)` keeps with prob `prob`; survivors scaled by
  `1/prob == 1/(1-rate)`; eval (`fit=False`) is a no-op. Correct.
- `Conv*` / `ConvTranspose*`: kernel shapes, `feature_group_count=groups`, dimension numbers,
  bias broadcast (channels-last), non-batching unsqueeze/squeeze, `SAME`/`VALID`/int/tuple padding
  normalization. Correct (smoke-checked shapes & a transpose upsample).
- `AvgPool`/`_AvgPoolNd` non-VALID averaging via a second `reduce_window` count. Correct.
- `GroupNorm` affine broadcast via `lax.broadcast_to_rank` to channels-last. Correct.
- Activation wrappers delegate to `bm.*`; formulas match docstrings. `Softmax2d` uses axis `-3`
  (channels-first `(N,C,H,W)`/`(C,H,W)`), matching its documented contract.
- `interoperation_flax`: Flax round-trip param flatten/unflatten, `ToFlaxRNNCell` carry handling.
  Correct for flax present/absent.
