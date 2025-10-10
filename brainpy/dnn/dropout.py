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
from typing import Optional

from brainpy import math as bm, check
from brainpy.context import share
from brainpy.dnn.base import Layer

__all__ = [
    'Dropout'
]


class Dropout(Layer):
    """A layer that stochastically ignores a subset of inputs each training step.

    In training, to compensate for the fraction of input values dropped (`rate`),
    all surviving values are multiplied by `1 / (1 - rate)`.

    This layer is active only during training (``mode=brainpy.math.training_mode``). In other
    circumstances it is a no-op.

    .. [1] Srivastava, Nitish, et al. "Dropout: a simple way to prevent
           neural networks from overfitting." The journal of machine learning
           research 15.1 (2014): 1929-1958.

    Args:
      prob: Probability to keep element of the tensor.
      mode: Mode. The computation mode of the object.
      name: str. The name of the dynamic system.

    """

    def __init__(
        self,
        prob: float,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None
    ):
        super(Dropout, self).__init__(mode=mode, name=name)
        self.prob = check.is_float(prob, min_bound=0., max_bound=1.)

    def update(self, x, fit: Optional[bool] = None):
        if fit is None:
            fit = share['fit']
        if fit:
            keep_mask = bm.random.bernoulli(self.prob, x.shape)
            return bm.where(keep_mask, x / self.prob, 0.)
        else:
            return x
