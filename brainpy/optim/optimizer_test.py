# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Regression tests for the 2026-06-19 optim/losses audit.

These pin the bug fixes for:

- P1-C1/P1-C2: ``Adan.update`` crashed on every call (``jax.lax.cond`` operand
  mis-binding) and its step counter was frozen at 0 (bias correction / Nesterov
  term disabled).
- P1-H1: ``SM3`` raised ``KeyError`` for scalar (0-dim) trainable variables.
"""

import numpy as np

import brainpy.math as bm
from brainpy.optim import optimizer as O


def _vec_var(values):
    return bm.Variable(bm.asarray(np.asarray(values, dtype=np.float32)))


def _scalar_var(value=2.0):
    return bm.Variable(bm.asarray(np.asarray(value, dtype=np.float32)))


def _train(opt, var_dict, grad_fn, steps=5):
    for _ in range(steps):
        grads = {k: grad_fn(bm.as_jax(v.value)) for k, v in var_dict.items()}
        opt.update(grads)
    return {k: np.asarray(bm.as_jax(v.value)) for k, v in var_dict.items()}


# ---------------------------------------------------------------------------
# Adan  (P1-C1, P1-C2)
# ---------------------------------------------------------------------------
class TestAdanFixed:
    def test_adan_runs_and_updates(self):
        # Previously raised TypeError on the very first update.
        v = _vec_var([2.0, -3.0])
        opt = O.Adan(lr=1e-2, train_vars={'w': v})
        out = _train(opt, {'w': v}, lambda val: val, steps=5)
        assert np.all(np.isfinite(out['w']))
        # gradient = value -> the parameter should move toward zero.
        assert np.all(np.abs(out['w']) < np.array([2.0, 3.0]))

    def test_adan_no_prox_runs(self):
        v = _vec_var([2.0, -3.0])
        opt = O.Adan(lr=1e-2, train_vars={'w': v}, no_prox=True)
        out = _train(opt, {'w': v}, lambda val: val, steps=5)
        assert np.all(np.isfinite(out['w']))

    def test_adan_step_counter_advances(self):
        v = _vec_var([1.0])
        opt = O.Adan(lr=1e-2, train_vars={'w': v})
        for _ in range(4):
            opt.update({'w': bm.as_jax(v.value)})
        assert int(bm.as_jax(opt.step.value)) == 4

    def test_adan_first_step_diff_is_zero(self):
        # On the very first update the gradient difference (g - g_prev) must be
        # treated as 0, so the exp_avg_diff (``_v``) accumulator stays 0 after
        # one step regardless of the gradient magnitude.
        v = _vec_var([5.0])
        opt = O.Adan(lr=1e-2, train_vars={'w': v})
        opt.update({'w': bm.as_jax(v.value)})
        v_diff = np.asarray(bm.as_jax(opt.implicit_vars['w_v'].value))
        assert np.allclose(v_diff, 0.0)

    def test_adan_nesterov_term_active_after_two_steps(self):
        # With the step counter frozen the diff term was permanently 0; here a
        # changing gradient must produce a non-zero exp_avg_diff after step 2.
        v = _vec_var([1.0])
        opt = O.Adan(lr=1e-2, train_vars={'w': v})
        opt.update({'w': np.asarray([1.0], dtype=np.float32)})
        opt.update({'w': np.asarray([3.0], dtype=np.float32)})  # gradient changed
        v_diff = np.asarray(bm.as_jax(opt.implicit_vars['w_v'].value))
        assert not np.allclose(v_diff, 0.0)


# ---------------------------------------------------------------------------
# SM3  (P1-H1)
# ---------------------------------------------------------------------------
class TestSM3Fixed:
    def test_sm3_scalar_var_runs(self):
        # Previously raised KeyError('w_m0') for a 0-dim variable.
        v = _scalar_var(2.0)
        opt = O.SM3(lr=0.1, train_vars={'w': v})
        out = _train(opt, {'w': v}, lambda val: np.ones_like(val), steps=4)
        assert np.all(np.isfinite(out['w']))
        # gradient is +1 each step -> scalar parameter must decrease.
        assert float(out['w']) < 2.0

    def test_sm3_scalar_matches_adagrad_like_step(self):
        # For a scalar with constant gradient g=1, SM3 reduces to an Adagrad-like
        # update: cache accumulates g^2, step = lr * g / sqrt(cache + eps).
        v = _scalar_var(0.0)
        opt = O.SM3(lr=0.1, train_vars={'w': v}, eps=1e-30)
        opt.update({'w': np.asarray(1.0, dtype=np.float32)})
        # after one step cache = 1, update = 0.1 * 1 / sqrt(1) = 0.1
        assert float(bm.as_jax(v.value)) == np.float32(-0.1)

    def test_sm3_scalar_still_works_with_momentum(self):
        v = _scalar_var(2.0)
        opt = O.SM3(lr=0.1, train_vars={'w': v}, momentum=0.5, beta=0.5)
        out = _train(opt, {'w': v}, lambda val: np.ones_like(val), steps=3)
        assert np.all(np.isfinite(out['w']))
