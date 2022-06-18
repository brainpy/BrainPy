# -*- coding: utf-8 -*-

import jax.numpy as jnp
from jax import vmap

from brainpy.math.numpy_ops import as_device_array

__all__ = [
  'pre2syn'
]


_pre2syn = vmap(lambda pre_id, pre_vs: pre_vs[pre_id], in_axes=(0, None))


def pre2syn(pre_values, pre_ids):
  """The pre-to-syn computation.

  Change the pre-synaptic data to the data with the dimension of synapses.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

    syn_val = np.zeros(len(pre_ids))
    for syn_i, pre_i in enumerate(pre_ids):
      syn_val[i] = pre_values[pre_i]

  Parameters
  ----------
  pre_values: float, jax.numpy.ndarray, JaxArray, Variable
    The pre-synaptic value.
  pre_ids: jax.numpy.ndarray, JaxArray
    The pre-synaptic neuron index.

  Returns
  -------
  syn_val: jax.numpy.ndarray, JaxArray
    The synaptic value.
  """
  pre_values = as_device_array(pre_values)
  pre_ids = as_device_array(pre_ids)
  if jnp.ndim(pre_values) == 0:
    return jnp.ones(len(pre_ids), dtype=pre_values.dtype) * pre_values
  else:
    return _pre2syn(pre_ids, pre_values)
