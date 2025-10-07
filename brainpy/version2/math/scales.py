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
from typing import Sequence, Union

__all__ = [
    'Scaling',
    'IdScaling',
]


class Scaling(object):
    def __init__(self, scale, bias):
        self.scale = scale
        self.bias = bias

    @classmethod
    def transform(
        cls,
        V_range: Sequence[Union[float, int]],
        scaled_V_range: Sequence[Union[float, int]] = (0., 1.)
    ) -> 'Scaling':
        """Transform the membrane potential range to a ``Scaling`` instance.

        Args:
          V_range:   [V_min, V_max]
          scaled_V_range:  [scaled_V_min, scaled_V_max]

        Returns:
          The instanced scaling object.
        """
        V_min, V_max = V_range
        scaled_V_min, scaled_V_max = scaled_V_range
        scale = (V_max - V_min) / (scaled_V_max - scaled_V_min)
        bias = scaled_V_min * scale - V_min
        return cls(scale=scale, bias=bias)

    def offset_scaling(self, x, bias=None, scale=None):
        if bias is None:
            bias = self.bias
        if scale is None:
            scale = self.scale
        return (x + bias) / scale

    def std_scaling(self, x, scale=None):
        if scale is None:
            scale = self.scale
        return x / scale

    def inv_scaling(self, x, scale=None):
        if scale is None:
            scale = self.scale
        return x * scale

    def clone(self, bias=None, scale=None):
        if bias is None:
            bias = self.bias
        if scale is None:
            scale = self.scale
        return Scaling(bias=bias, scale=scale)


class IdScaling(Scaling):
    def __init__(self):
        super().__init__(scale=1., bias=0.)

    def offset_scaling(self, x, bias=None, scale=None):
        return x

    def std_scaling(self, x, scale=None):
        return x

    def inv_scaling(self, x, scale=None):
        return x

    def clone(self, bias=None, scale=None):
        return IdScaling()
