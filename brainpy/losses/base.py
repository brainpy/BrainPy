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

from brainpy.dnn.base import Layer

__all__ = [
    'Loss',
    'WeightedLoss',
]


class Loss(Layer):
    reduction: str

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction


class WeightedLoss(Loss):
    weight: Optional

    def __init__(self, weight: Optional = None, reduction: str = 'mean') -> None:
        super().__init__(reduction)
        self.weight = weight
