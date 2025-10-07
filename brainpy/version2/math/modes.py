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


import brainstate

__all__ = [
    'Mode',
    'NonBatchingMode',
    'BatchingMode',
    'TrainingMode',
    'nonbatching_mode',
    'batching_mode',
    'training_mode',
]


class Mode(brainstate.mixin.Mode):
    """Base class for computation Mode
    """

    def __repr__(self):
        return self.__class__.__name__

    def __eq__(self, other: 'Mode'):
        if not isinstance(other, Mode):
            return False
        return other.__class__ == self.__class__

    def is_one_of(self, *modes):
        for m_ in modes:
            if not isinstance(m_, type):
                raise TypeError(f'The supported type must be a tuple/list of type. But we got {m_}')
        return self.__class__ in modes

    def is_a(self, mode: type):
        """Check whether the mode is exactly the desired mode."""
        assert isinstance(mode, type), 'Must be a type.'
        return self.__class__ == mode

    def is_parent_of(self, *modes):
        """Check whether the mode is a parent of the given modes."""
        cls = self.__class__
        for m_ in modes:
            if not isinstance(m_, type):
                raise TypeError(f'The supported type must be a tuple/list of type. But we got {m_}')
        if all([not issubclass(m_, cls) for m_ in modes]):
            return False
        else:
            return True

    def is_child_of(self, *modes):
        """Check whether the mode is a children of one of the given modes."""
        for m_ in modes:
            if not isinstance(m_, type):
                raise TypeError(f'The supported type must be a tuple/list of type. But we got {m_}')
        return isinstance(self, modes)

    def is_batch_mode(self):
        return isinstance(self, BatchingMode)

    def is_train_mode(self):
        return isinstance(self, TrainingMode)

    def is_nonbatch_mode(self):
        return isinstance(self, NonBatchingMode)


class NonBatchingMode(Mode):
    """Normal non-batching mode.

    :py:class:`~.NonBatchingMode` is usually used in models of traditional
    computational neuroscience.
    """
    pass

    @property
    def batch_size(self):
        return tuple()


class BatchingMode(Mode):
    """Batching mode.

    :py:class:`~.NonBatchingMode` is usually used in models of model trainings.
    """

    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size

    def __repr__(self):
        return f'{self.__class__.__name__}(batch_size={self.batch_size})'


class TrainingMode(BatchingMode):
    """Training mode requires data batching."""

    def to_batch_mode(self):
        return BatchingMode(self.batch_size)


nonbatching_mode = NonBatchingMode()
'''Default instance of the non-batching computation mode.'''

batching_mode = BatchingMode()
'''Default instance of the batching computation mode.'''

training_mode = TrainingMode()
'''Default instance of the training computation mode.'''
