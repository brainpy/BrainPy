from typing import Dict

from brainpy._src.dyn.base import IonChaDyn
from brainpy._src.dynsys import DynamicalSystem, DynView
from brainpy._src.math.object_transform.base import StateLoadResult


__all__ = [
  'reset_state',
  'load_state',
  'save_state',
  'clear_input',
]


def reset_state(target: DynamicalSystem, *args, **kwargs):
  """Reset states of all children nodes in the given target.

  See https://brainpy.readthedocs.io/en/latest/tutorial_toolbox/state_resetting.html for details.

  Args:
    target: The target DynamicalSystem.
    *args:
    **kwargs:
  """
  for node in target.nodes().subset(DynamicalSystem).not_subset(DynView).not_subset(IonChaDyn).unique().values():
    node.reset_state(*args, **kwargs)


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

