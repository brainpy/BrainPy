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
from typing import Union, Optional

import brainpy.math as bm
from brainpy.check import is_float, is_integer
from brainpy.context import share
from brainpy.dynsys import DynamicalSystem


class PoissonInput(DynamicalSystem):
    """Poisson Input.

    Adds independent Poisson input to a target variable. For large
    numbers of inputs, this is much more efficient than creating a
    `PoissonGroup`. The synaptic events are generated randomly during the
    simulation and are not preloaded and stored in memory. All the inputs must
    target the same variable, have the same frequency and same synaptic weight.
    All neurons in the target variable receive independent realizations of
    Poisson spike trains.

    Parameters::

    num_input: int
      The number of inputs.
    freq: float
      The frequency of each of the inputs. Must be a scalar.
    weight: float
      The synaptic weight. Must be a scalar.
    """

    def __init__(
        self,
        target_shape,
        num_input: int,
        freq: Union[int, float],
        weight: Union[int, float],
        seed: Optional[int] = None,
        mode: bm.Mode = None,
        name: str = None
    ):
        super(PoissonInput, self).__init__(name=name, mode=mode)

        # check data
        is_integer(num_input, 'num_input', min_bound=1)
        is_float(freq, 'freq', min_bound=0., allow_int=True)
        is_float(weight, 'weight', allow_int=True)
        assert self.mode.is_parent_of(bm.NonBatchingMode, bm.BatchingMode)

        # parameters
        self.target_shape = target_shape
        self.num_input = num_input
        self.freq = freq
        self.weight = weight
        self.seed = seed

    def update(self):
        p = self.freq * share.dt / 1e3
        a = self.num_input * p
        b = self.num_input * (1 - p)
        if isinstance(share.dt, (int, float)):  # dt is not in tracing
            if (a > 5) and (b > 5):
                inp = bm.random.normal(a, b * p, self.target_shape)
            else:
                inp = bm.random.binomial(self.num_input, p, self.target_shape)

        else:  # dt is in tracing
            inp = bm.cond((a > 5) * (b > 5),
                          lambda _: bm.random.normal(a, b * p, self.target_shape),
                          lambda _: bm.random.binomial(self.num_input, p, self.target_shape),
                          None)
        return inp * self.weight

    def __repr__(self):
        names = self.__class__.__name__
        return f'{names}(shape={self.target_shape}, num_input={self.num_input}, freq={self.freq}, weight={self.weight})'

    def reset_state(self, batch_size=None):
        pass

    def reset(self, batch_size=None):
        self.reset_state(batch_size)
