# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


import pytest

pytest.skip('Skip this test because it is not implemented yet.',
            allow_module_level=True)

import jax
import jax.numpy as jnp
import flax.linen as nn

import brainpy as bp
import brainpy.math as bm

bm.set_platform('cpu')
bm.set_mode(bm.training_mode)

cell = bp.dnn.ToFlaxRNNCell(bp.dyn.RNNCell(num_in=1, num_out=1, ))


class myRNN(nn.Module):
    @nn.compact
    def __call__(self, x):  # x:(batch, time, features)
        x = nn.RNN(cell)(x)  # Use nn.RNN to unfold the recurrent cell
        return x


def test_init():
    model = myRNN()
    model.init(jax.random.PRNGKey(0), jnp.ones([1, 10, 1]))  # batch,time,feature
