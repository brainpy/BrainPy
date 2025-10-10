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
from typing import Dict, Callable

from brainpy import dynsys
from brainpy.dyn.base import IonChaDyn
from brainpy.dynsys import DynamicalSystem, DynView
from brainpy.math.object_transform.base import StateLoadResult

__all__ = [
    'reset_level',
    'reset_state',
    'load_state',
    'save_state',
    'clear_input',
]

_max_level = 10


def reset_level(level: int = 0):
    """The decorator for indicating the resetting level.

    The function takes an optional integer argument level with a default value of 0.

    The lower the level, the earlier the function is called.

    >>> import brainpy as bp
    >>> bp.reset_level(0)
    >>> bp.reset_level(-1)
    >>> bp.reset_level(-2)

    """
    if level < 0:
        level = _max_level + level
    if level < 0 or level >= _max_level:
        raise ValueError(f'"reset_level" must be an integer in [0, 10). but we got {level}')

    def wrap(fun: Callable):
        fun.reset_level = level
        return fun

    return wrap


def reset_state(target: DynamicalSystem, *args, **kwargs):
    """Reset states of all children nodes in the given target.

    See https://brainpy.readthedocs.io/en/latest/tutorial_toolbox/state_resetting.html for details.

    Args:
      target: The target DynamicalSystem.
    """
    dynsys.the_top_layer_reset_state = False

    try:
        nodes = list(target.nodes().subset(DynamicalSystem).not_subset(DynView).not_subset(IonChaDyn).unique().values())
        nodes_with_level = []

        # reset node whose `reset_state` has no `reset_level`
        for node in nodes:
            if not hasattr(node.reset_state, 'reset_level'):
                node.reset_state(*args, **kwargs)
            else:
                nodes_with_level.append(node)

        # reset the node's states
        for l in range(_max_level):
            for node in nodes_with_level:
                if node.reset_state.reset_level == l:
                    node.reset_state(*args, **kwargs)

    finally:
        dynsys.the_top_layer_reset_state = True


def clear_input(target: DynamicalSystem, *args, **kwargs):
    """Clear all inputs in the given target.

    Args:
      target:The target DynamicalSystem.

    """
    for node in target.nodes().subset(DynamicalSystem).not_subset(DynView).unique().values():
        node.clear_input(*args, **kwargs)


def load_state(target: DynamicalSystem, state_dict: Dict, **kwargs):
    """Copy parameters and buffers from :attr:`state_dict` into
    this module and its descendants.

    Args:
      target: DynamicalSystem. The dynamical system to load its states.
      state_dict: dict. A dict containing parameters and persistent buffers.

    Returns:
    -------
      ``NamedTuple``  with ``missing_keys`` and ``unexpected_keys`` fields:

      * **missing_keys** is a list of str containing the missing keys
      * **unexpected_keys** is a list of str containing the unexpected keys
    """
    nodes = target.nodes().subset(DynamicalSystem).not_subset(DynView).unique()
    missing_keys = []
    unexpected_keys = []
    for name, node in nodes.items():
        r = node.load_state(state_dict[name], **kwargs)
        if r is not None:
            missing, unexpected = r
            missing_keys.extend([f'{name}.{key}' for key in missing])
            unexpected_keys.extend([f'{name}.{key}' for key in unexpected])
    return StateLoadResult(missing_keys, unexpected_keys)


def save_state(target: DynamicalSystem, **kwargs) -> Dict:
    """Save all states in the ``target`` as a dictionary for later disk serialization.

    Args:
      target: DynamicalSystem. The node to save its states.

    Returns:
      Dict. The state dict for serialization.
    """
    nodes = target.nodes().subset(DynamicalSystem).not_subset(DynView).unique()  # retrieve all nodes
    return {key: node.save_state(**kwargs) for key, node in nodes.items()}
