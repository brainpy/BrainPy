from .dynsys import DynamicalSystem, DynView
from brainpy._src.dyn.base import IonChaDyn

__all__ = [
  'reset_state',
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
