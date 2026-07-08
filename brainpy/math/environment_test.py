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
import subprocess
import sys
import unittest

import jax

import brainpy.math as bm


class TestEnvironment(unittest.TestCase):
    def test_numpy_func_return(self):
        # Reset random state to ensure clean state between tests
        bm.random.seed()

        with bm.environment(numpy_func_return='jax_array'):
            a = bm.random.randn(3, 3)
            self.assertTrue(isinstance(a, jax.Array))
        with bm.environment(numpy_func_return='bp_array'):
            a = bm.zeros([3, 3])
            self.assertTrue(isinstance(a, bm.Array))


# Regression: ``clear_buffer_memory()`` used to delete JAX's runtime effect-token
# buffers along with everything else, so a *subsequent* ordered-effect dispatch
# crashed with "INVALID_ARGUMENT: BlockHostUntilReady() called on deleted or
# donated buffer" (hit by ``jax_vectorize_map(..., clear_buffer=True)`` around a
# ``DSRunner``/``for_loop``). The fix resets JAX's runtime-token registry after
# deleting live buffers. Run in a subprocess: the real ``clear_buffer_memory``
# deletes *every* live device buffer, which would poison the shared pytest
# session if run in-process.
_CLEAR_BUFFER_ORDERED_EFFECT_SNIPPET = r"""
import jax
import jax.numpy as jnp
from jax.experimental import io_callback
import brainpy.math as bm

def f(x):
    # ordered=True registers a runtime effect token that lives as an on-device buffer
    y = io_callback(lambda v: v, jax.ShapeDtypeStruct(x.shape, x.dtype), x, ordered=True)
    return y + 1.0

jf = jax.jit(f)
jf(jnp.ones(3)).block_until_ready()          # first ordered-effect dispatch
bm.clear_buffer_memory()                      # deletes live buffers (incl. the token)
jf(jnp.ones(3)).block_until_ready()          # must NOT raise "deleted or donated buffer"
print('OK')
"""


class TestClearBufferMemory(unittest.TestCase):
    def test_ordered_effect_survives_clear_buffer_memory(self):
        proc = subprocess.run(
            [sys.executable, '-c', _CLEAR_BUFFER_ORDERED_EFFECT_SNIPPET],
            capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(
            proc.returncode, 0,
            msg=f'ordered-effect dispatch after clear_buffer_memory failed:\n'
                f'STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}',
        )
        self.assertIn('OK', proc.stdout)
        self.assertNotIn('deleted or donated buffer', proc.stderr)
