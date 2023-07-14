# -*- coding: utf-8 -*-

from typing import Sequence, Dict, Union

import numpy as np
from jax import vmap, pmap
from jax.tree_util import tree_unflatten, tree_flatten

import brainpy.math as bm
from brainpy.types import ArrayType

__all__ = [
  'jax_vectorize_map',
  'jax_parallelize_map',
]


def jax_vectorize_map(
    func: callable,
    arguments: Union[Dict[str, ArrayType], Sequence[ArrayType]],
    num_parallel: int,
    clear_buffer: bool = False
):
  """Perform a vectorized map of a function by using ``jax.vmap``.

  This function can be used in CPU or GPU backends. But it is highly
  suitable to be used in GPU backends. This is because ``jax.vmap``
  can parallelize the mapped axis on GPU devices.

  Parameters
  ----------
  func: callable, function
    The function to be mapped.
  arguments: sequence, dict
    The function arguments, used to define tasks.
  num_parallel: int
    The number of batch size.
  clear_buffer: bool
    Clear the buffer memory after running each batch data.

  Returns
  -------
  results: Any
    The running results.
  """
  if not isinstance(arguments, (dict, tuple, list)):
    raise TypeError(f'"arguments" must be sequence or dict, but we got {type(arguments)}')
  elements, tree = tree_flatten(arguments, is_leaf=lambda a: isinstance(a, bm.Array))
  if clear_buffer:
    elements = [np.asarray(ele) for ele in elements]
  num_pars = [len(ele) for ele in elements]
  if len(np.unique(num_pars)) != 1:
    raise ValueError(f'All elements in parameters should have the same length. '
                     f'But we got {tree_unflatten(tree, num_pars)}')

  res_tree = None
  results = None
  vmap_func = vmap(func)
  for i in range(0, num_pars[0], num_parallel):
    run_f = vmap(func) if clear_buffer else vmap_func
    if isinstance(arguments, dict):
      r = run_f(**tree_unflatten(tree, [ele[i: i + num_parallel] for ele in elements]))
    elif isinstance(arguments, (tuple, list)):
      r = run_f(*tree_unflatten(tree, [ele[i: i + num_parallel] for ele in elements]))
    else:
      raise TypeError
    res_values, res_tree = tree_flatten(r, is_leaf=lambda a: isinstance(a, bm.Array))
    if results is None:
      results = tuple([np.asarray(val) if clear_buffer else val] for val in res_values)
    else:
      for j, val in enumerate(res_values):
        results[j].append(np.asarray(val) if clear_buffer else val)
    if clear_buffer:
      bm.clear_buffer_memory()
  if res_tree is None:
    return None
  results = ([np.concatenate(res, axis=0) for res in results]
             if clear_buffer else
             [bm.concatenate(res, axis=0) for res in results])
  return tree_unflatten(res_tree, results)


def jax_parallelize_map(
    func: callable,
    arguments: Union[Dict[str, ArrayType], Sequence[ArrayType]],
    num_parallel: int,
    clear_buffer: bool = False
):
  """Perform a parallelized map of a function by using ``jax.pmap``.

  This function can be used in multi- CPU or GPU backends.
  If you are using it in a single CPU, please set host device count
  by ``brainpy.math.set_host_device_count(n)`` before.

  Parameters
  ----------
  func: callable, function
    The function to be mapped.
  arguments: sequence, dict
    The function arguments, used to define tasks.
  num_parallel: int
    The number of batch size.
  clear_buffer: bool
    Clear the buffer memory after running each batch data.

  Returns
  -------
  results: Any
    The running results.
  """
  if not isinstance(arguments, (dict, tuple, list)):
    raise TypeError(f'"arguments" must be sequence or dict, but we got {type(arguments)}')
  elements, tree = tree_flatten(arguments, is_leaf=lambda a: isinstance(a, bm.Array))
  if clear_buffer:
    elements = [np.asarray(ele) for ele in elements]
  num_pars = [len(ele) for ele in elements]
  if len(np.unique(num_pars)) != 1:
    raise ValueError(f'All elements in parameters should have the same length. '
                     f'But we got {tree_unflatten(tree, num_pars)}')

  res_tree = None
  results = None
  vmap_func = pmap(func)
  for i in range(0, num_pars[0], num_parallel):
    run_f = pmap(func) if clear_buffer else vmap_func
    if isinstance(arguments, dict):
      r = run_f(**tree_unflatten(tree, [ele[i: i + num_parallel] for ele in elements]))
    else:
      r = run_f(*tree_unflatten(tree, [ele[i: i + num_parallel] for ele in elements]))
    res_values, res_tree = tree_flatten(r, is_leaf=lambda a: isinstance(a, bm.Array))
    if results is None:
      results = tuple([np.asarray(val) if clear_buffer else val] for val in res_values)
    else:
      for j, val in enumerate(res_values):
        results[j].append(np.asarray(val) if clear_buffer else val)
    if clear_buffer:
      bm.clear_buffer_memory()
  if res_tree is None:
    return None
  results = ([np.concatenate(res, axis=0) for res in results]
             if clear_buffer else
             [bm.concatenate(res, axis=0) for res in results])
  return tree_unflatten(res_tree, results)
