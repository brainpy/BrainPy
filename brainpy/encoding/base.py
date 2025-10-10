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
from brainpy.math.object_transform.base import BrainPyObject

__all__ = [
    'Encoder'
]


class Encoder(BrainPyObject):
    """Base class for encoding rate values as spike trains."""

    def __repr__(self):
        return self.__class__.__name__

    def single_step(self, *args, **kwargs):
        raise NotImplementedError('Please implement the function for single step encoding.')

    def multi_steps(self, *args, **kwargs):
        raise NotImplementedError('Encode implement the function for multiple-step encoding.')
