from typing import Tuple

import jax.numpy as jnp

from brainpy._src import tools

__all__ = [
  'info'
]


def info(events: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Collect event information, including event indices, and event number.

  This function supports JAX transformations, including `jit()`,
  `vmap()` and `pmap()`.

  Parameters
  ----------
  events: jnp.ndarray
    The events.

  Returns
  -------
  res: tuple
    A tuple with two elements, denoting the event indices and the event number.
  """
  bl = tools.import_brainpylib()
  return bl.event_ops.event_info(events)
