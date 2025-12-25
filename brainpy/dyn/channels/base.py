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
from brainpy.dyn.base import IonChaDyn
from brainpy.dyn.neurons.hh import HHTypedNeuron
from brainpy.mixin import TreeNode

__all__ = [
    'IonChannel',
]


class IonChannel(IonChaDyn, TreeNode):
    """Base class for ion channels."""

    '''The type of the master object.'''
    master_type = HHTypedNeuron

    def update(self, *args, **kwargs):
        raise NotImplementedError('Must be implemented by the subclass.')

    def current(self, *args, **kwargs):
        raise NotImplementedError('Must be implemented by the subclass.')

    def reset_state(self, *args, **kwargs):
        raise NotImplementedError('Must be implemented by the subclass.')

    def clear_input(self):
        pass

    def __repr__(self):
        return f'{self.name}(size={self.size})'
