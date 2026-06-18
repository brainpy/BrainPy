# Audit — brainpy.math compatibility / activations / linalg / fft (2026-06-19)

Scope: `brainpy/math/{activations,compat_numpy,compat_pytorch,compat_tensorflow,fft,interoperability,linalg}.py`

Environment: jax 0.10.2, jaxlib 0.10.1, numpy 2.4.6 (CPU). Implicit
`__jax_array__` still honoured by `jax.numpy` in this version, so brainpy
`Array` leaves are still implicitly convertible.

---

### P3-H1 — `gelu` silently wrong for integer (non-floating) inputs  [High]
- File: brainpy/math/activations.py:146-178
- Category: correctness / dtype
- What: `gelu` reads `x.dtype` and forces the result to that dtype without first
  promoting the input to an inexact type. For an integer input both branches
  are wrong:
  - `approximate=True`: `sqrt_2_over_pi = np.sqrt(2/pi).astype(x.dtype)` truncates
    the constant `0.7978…` to `0` for an int dtype, and `0.044715` likewise
    vanishes, so the whole `tanh(...)` argument collapses to `0`, giving
    `cdf = 0.5` and `gelu(x) = x/2`.
  - `approximate=False`: `jnp.array(..., dtype=x.dtype)` truncates the float
    result back to int.
- Why it's a bug: `gelu(jnp.array([1,2,3], int32), approximate=True)` returns
  `[0.5, 1.0, 1.5]` instead of the correct `[0.8412, 1.9546, 2.9964]`
  (matches `jax.nn.gelu`). `approximate=False` on the same input returns
  `[0,0,0,1]`. JAX's own `gelu` promotes the argument to inexact first.
- Repro:
  ```python
  import brainpy.math.activations as act, jax.numpy as jnp
  act.gelu(jnp.array([1,2,3], jnp.int32), approximate=True)   # [0.5,1.,1.5] WRONG
  ```
- Fix: promote `x` to an inexact dtype (`jnp.promote_types(x.dtype, jnp.float32)`
  / `jnp.asarray(x) * 1.0` style) before computing, mirroring `jax.nn.gelu`.
- Tests: test_gelu_integer_input_matches_float (both branches)
- Status: fixed

### P3-H2 — `unflatten` ignores negative `dim`  [High]
- File: brainpy/math/compat_pytorch.py:104-126
- Category: correctness
- What: `unflatten(x, dim, sizes)` builds `shape[:dim] + sizes + shape[dim+1:]`
  without normalising a negative `dim`. PyTorch's `torch.unflatten` accepts
  negative `dim`. With `dim=-1` on a `(6,)` array the slices become
  `shape[:-1]=()` and `shape[0:]=(6,)`, yielding the target shape `(2,3,6)` and a
  reshape failure (size 6 → 36). The leading `assert x.ndim > dim` is also wrong
  for negative `dim` (always true) and is evaluated on the raw input before
  `_as_jax_array_`.
- Why it's a bug: `unflatten(arange(6), -1, (2,3))` raises `TypeError: cannot
  reshape array of shape (6,) into shape (2,3,6)` instead of returning `(2,3)`.
- Repro:
  ```python
  import brainpy.math.compat_pytorch as cpt, brainpy.math as bm, jax.numpy as jnp
  cpt.unflatten(bm.asarray(jnp.arange(6.)), -1, (2,3))   # raises, should be (2,3)
  ```
- Fix: normalise `dim` (`dim += x.ndim` when negative) and validate against
  `x.ndim` after canonicalisation; convert to jax array before indexing `.shape`.
- Tests: test_unflatten_negative_dim, test_unflatten_dim_out_of_range
- Status: fixed

### P3-M1 — `jnp.ones_like(data)` passed a brainpy `Array` in TF segment helpers  [Medium]
- File: brainpy/math/compat_tensorflow.py:143,211,228 (segment_mean,
  unsorted_segment_sqrt_n, unsorted_segment_mean)
- Category: api-drift / fragility
- What: the denominator is computed with `jnp.ones_like(data)` where `data` is
  the *un-converted* argument (possibly a brainpy `Array`), while every other
  argument is funnelled through `_as_jax_array_`. This only works because
  `jax.numpy` still honours the implicit `__jax_array__` protocol; the audit
  brief flags that protocol as scheduled for removal (JAX ≥ 0.9). The
  inconsistency is a latent break.
- Why it's a bug: once implicit `__jax_array__` is dropped, `jnp.ones_like(<bp
  Array>)` raises and `segment_mean` / `unsorted_segment_mean` /
  `unsorted_segment_sqrt_n` crash on brainpy inputs (the common case).
- Repro: static (currently works via the deprecated protocol).
- Fix: convert once (`data = _as_jax_array_(data)`) and use `jnp.ones_like` on
  the jax array.
- Tests: test_segment_mean_array_input, test_unsorted_segment_mean_array_input,
  test_unsorted_segment_sqrt_n_array_input
- Status: fixed

### P3-M2 — `gelu(approximate=False)` uses deprecated `jax.lax.erf`  [Low]
- File: brainpy/math/activations.py:177
- Category: api-drift
- What: uses `jax.lax.erf`, which is being phased out in favour of
  `jax.scipy.special.erf` / `lax.erfc` (jax.nn.gelu switched to `lax.erfc`). Not
  yet a warning on 0.10.2, but a drift risk.
- Why it's a bug: future removal would break the exact-GELU branch.
- Repro: static.
- Fix: recorded only (folded into the P3-H1 rewrite, which uses
  `jax.scipy.special.erf`).
- Tests: covered indirectly by test_gelu_integer_input_matches_float.
- Status: fixed (as part of P3-H1)

### P3-L1 — `one_hot` uses deprecated `jax.core.concrete_or_error`  [Low]
- File: brainpy/math/activations.py:444
- Category: api-drift
- What: `jax.core.concrete_or_error` emits a `DeprecationWarning` ("Use
  jax.extend.core.concrete_or_error") on every `one_hot` call.
- Why it's a bug: noisy deprecation; future removal risk.
- Repro: `import warnings; act.one_hot(jnp.array([0,1]), 2)` warns.
- Fix: recorded only (Low).
- Tests: none
- Status: recorded-only

### P3-L2 — `one_hot` hard-codes `jnp.float64` default dtype  [Low]
- File: brainpy/math/activations.py:446
- Category: numerics / api-drift
- What: `dtype = canonicalize_dtype(jnp.float64 if dtype is None else dtype)`.
  With `jax_enable_x64` off (the default) this is canonicalised to float32, but
  the literal `float64` path is more roundabout than `jnp.float_`/`None`.
  Harmless on current default config (the canonicalize call absorbs it) so
  recorded only.
- Why it's a bug: minor; no incorrect output on the default config.
- Repro: static.
- Fix: recorded only (Low).
- Tests: none
- Status: recorded-only

### P3-L3 — `asfarray` forces `jnp.float64` → spurious truncation warning  [Low]
- File: brainpy/math/compat_numpy.py:217-220
- Category: numerics
- What: the H-13 fix coerces integer input to `jnp.float64`; with x64 off this
  emits "Explicitly requested dtype float64 … will be truncated to float32" on
  every call. Output is correct (float32), only noisy.
- Why it's a bug: noisy warning; cosmetic.
- Repro: `import warnings; cn.asfarray([1,2,3])` warns.
- Fix: recorded only (Low) — changing the dtype literal is a behaviour tweak
  outside the verified-bug remit and the existing H-13 test pins float output.
- Tests: none
- Status: recorded-only

### P3-L4 — `interoperability.from_numpy` returns a numpy ndarray  [Low]
- File: brainpy/math/interoperability.py:119-120
- Category: api / naming
- What: `from_numpy(arr)` delegates to `as_ndarray`, returning a
  `numpy.ndarray`. The PyTorch-style name implies a conversion *into* the
  framework array type (jax/brainpy). Long-standing BrainPy behaviour; callers
  may depend on it.
- Why it's a bug: misleading name; not a correctness defect.
- Repro: `type(io.from_numpy(np.arange(3)))  # numpy.ndarray`.
- Fix: recorded only (Low) — behaviour change with downstream risk.
- Tests: none
- Status: recorded-only

### P3-L5 — `as_device_array` docstring references removed `DeviceArray`  [Low]
- File: brainpy/math/interoperability.py:39-64
- Category: style / docs
- What: docstring and name reference `jax.numpy.DeviceArray`, a type removed from
  modern JAX (now `jax.Array`). Function itself is correct.
- Why it's a bug: stale docs.
- Repro: static.
- Fix: recorded only (Low).
- Tests: none
- Status: recorded-only
