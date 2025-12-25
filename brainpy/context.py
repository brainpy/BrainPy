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
"""
Context for brainpy computation.

This context defines all shared data used in all modules in a computation.
"""

from typing import Any, Union

import brainstate

from brainpy.math.defaults import env
from brainpy.tools.dicts import DotDict

__all__ = [
    'share',
]


class _ShareContext:
    def __init__(self):
        super().__init__()

        # Shared data across all nodes at current time step.
        # -------------

        self._arguments = DotDict()
        self._category = dict()

    @property
    def dt(self):
        return brainstate.environ.get_dt(env=env)

    @dt.setter
    def dt(self, dt):
        self.set_dt(dt)

    def set_dt(self, dt: Union[int, float]):
        brainstate.environ.set(dt=dt, env=env)

    def load(self, key, value: Any = None, desc: str = None):
        """Load the shared data by the ``key``.

        Args:
          key (str): the key to indicate the data.
          value (Any): the default value when ``key`` is not defined in the shared.
          desc: (str): the description of the key.
        """
        return brainstate.environ.get(key, value, desc, env=env)

    def save(self, *args, **kwargs) -> None:
        """Save shared arguments in the global context."""
        assert len(args) % 2 == 0
        for i in range(0, len(args), 2):
            identifier = args[i]
            data = args[i + 1]
            brainstate.environ.set(**{identifier: data}, env=env)
        brainstate.environ.set(**kwargs, env=env)

    def __setitem__(self, key, value):
        """Enable setting the shared item by ``bp.share[key] = value``."""
        self.save(key, value)

    def __getitem__(self, item):
        """Enable loading the shared parameter by ``bp.share[key]``."""
        return self.load(item)

    def get_shargs(self) -> DotDict:
        """Get all shared arguments in the global context."""
        return DotDict(brainstate.environ.all(env=env))


share = _ShareContext()
