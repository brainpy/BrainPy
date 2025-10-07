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
import abc

__all__ = [
    'Initializer',
    '_InterLayerInitializer',
    '_IntraLayerInitializer'
]


class Initializer(abc.ABC):
    """Base Initialization Class."""

    @abc.abstractmethod
    def __call__(self, shape, dtype=None):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class _InterLayerInitializer(Initializer):
    """The superclass of Initializers that initialize the weights between two layers."""
    pass


class _IntraLayerInitializer(Initializer):
    """The superclass of Initializers that initialize the weights within a layer."""
    pass
