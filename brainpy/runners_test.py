# -*- coding: utf-8 -*-
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
"""Regression tests for ``brainpy.runners.DSRunner``."""

import unittest

import jax.numpy as jnp
import numpy as np

import brainpy as bp
import brainpy.math as bm


class _GainAccumulator(bp.DynamicalSystem):
    """Adds a shared ``gain`` (default 1.0) to an internal counter each step."""

    def __init__(self):
        super().__init__()
        self.i = bm.Variable(jnp.zeros(1))

    def update(self, x):
        gain = bp.share.load('gain', 1.0)
        self.i += gain
        return self.i.value

    def reset_state(self, *args, **kwargs):
        self.i.value = jnp.zeros(1)


class TestDSRunnerSharedArgs(unittest.TestCase):
    """Regression for M4 (audit 2026-07-08): ``DSRunner(memory_efficient=True)``
    dropped ``shared_args`` in its predict loop, so user-supplied shared arguments
    were silently ignored (the standard path forwarded them)."""

    def setUp(self):
        bm.set_dt(1.0)

    def _run(self, memory_efficient):
        model = _GainAccumulator()
        runner = bp.DSRunner(model, monitors=['i'],
                             memory_efficient=memory_efficient, progress_bar=False)
        runner.predict(inputs=np.zeros((3, 1)), shared_args={'gain': 2.0}, reset_state=True)
        return np.asarray(runner.mon['i']).ravel()

    def test_shared_args_honored_in_memory_efficient(self):
        # gain=2 -> cumulative [2, 4, 6]; with the bug memory_efficient defaulted the
        # gain to 1 and produced [1, 2, 3].
        got = self._run(memory_efficient=True)
        np.testing.assert_allclose(got, [2., 4., 6.])

    def test_memory_efficient_matches_standard(self):
        std = self._run(memory_efficient=False)
        mem = self._run(memory_efficient=True)
        np.testing.assert_allclose(mem, std)


if __name__ == '__main__':
    unittest.main()
