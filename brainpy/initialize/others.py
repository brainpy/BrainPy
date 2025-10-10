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
from typing import Callable

import brainpy.math as bm
from .base import Initializer


class Clip(Initializer):
    def __init__(self, init: Callable, min=None, max=None):
        self.min = min
        self.max = max
        self.init = init

    def __call__(self, shape, dtype=None):
        x = self.init(shape, dtype)
        if self.min is not None:
            x = bm.maximum(self.min, x)
        if self.max is not None:
            x = bm.minimum(self.max, x)
        return x

    def __repr__(self):
        return f'{self.__class__.__name__}({self.init}, min={self.min}, max={self.max})'
