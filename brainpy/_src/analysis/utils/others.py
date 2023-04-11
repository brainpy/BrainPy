# -*- coding: utf-8 -*-

from typing import Union, Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

import brainpy.math as bm
from .function import f_without_jaxarray_return
from .measurement import euclidean_distance_jax

__all__ = [
  'Segment',
  'check_initials',
  'check_plot_durations',
  'get_sign',
  'get_sign2',
  'keep_unique',
  'rescale',
]


class Segment(object):
  def __init__(self, targets, num_segments):
    assert isinstance(targets, (tuple, list))
    # num segments
    if isinstance(num_segments, int):
      num_segments = tuple([num_segments] * len(targets))
    assert isinstance(num_segments, (tuple, list)) and len(num_segments) == len(targets)
    arg_lens = tuple(len(p) for p in targets)
    self. arg_pre_len = tuple(int(np.ceil(l / num_segments[i])) for i, l in enumerate(arg_lens))
    arg_id_segments = tuple(np.arange(0, l, self. arg_pre_len[i]) for i, l in enumerate(arg_lens))
    self. arg_id_segments = tuple(ids.flatten() for ids in np.meshgrid(*arg_id_segments))
    if len(arg_id_segments) == 0:
     self. arg_id_segments = ((0,),)
    self.targets = targets

  def __iter__(self):
    for ids in zip(*self. arg_id_segments):
      yield tuple(p[ids[i]: ids[i] + self. arg_pre_len[i]] for i, p in enumerate(self.targets))


def check_initials(initials, target_var_names):
  # check the initial values
  assert isinstance(initials, dict)
  for p in target_var_names:
    assert p in initials
  initials = {p: bm.as_jax(initials[p], dtype=bm.float_) for p in target_var_names}
  len_of_init = []
  for v in initials.values():
    assert isinstance(v, (tuple, list, np.ndarray, jnp.ndarray, bm.ndarray))
    len_of_init.append(len(v))
  len_of_init = np.unique(len_of_init)
  assert len(len_of_init) == 1
  return initials


def check_plot_durations(plot_durations, duration, initials):
  if plot_durations is None:
    plot_durations = [(0., duration) for _ in range(len(initials))]
  if isinstance(plot_durations[0], (int, float)):
    assert len(plot_durations) == 2
    plot_durations = [plot_durations for _ in range(len(initials))]
  else:
    assert len(plot_durations) == len(initials)
    for dur in plot_durations:
      assert len(dur) == 2
  return plot_durations


def get_sign(f, xs, ys):
  f = f_without_jaxarray_return(f)
  xs = xs.value if isinstance(xs, bm.Array) else xs
  ys = ys.value if isinstance(ys, bm.Array) else ys
  Y, X = jnp.meshgrid(ys, xs)
  return jnp.sign(f(X, Y))


def get_sign2(f, *xyz, args=()):
  in_axes = tuple(range(len(xyz))) + tuple([None] * len(args))
  f = jax.jit(jax.vmap(f_without_jaxarray_return(f), in_axes=in_axes))
  xyz = tuple((v.value if isinstance(v, bm.Array) else v) for v in xyz)
  XYZ = jnp.meshgrid(*xyz)
  XYZ = tuple(jnp.moveaxis(v, 1, 0).flatten() for v in XYZ)
  shape = (len(v) for v in xyz)
  return jnp.sign(f(*(XYZ + args))).reshape(shape)


def keep_unique(candidates: Union[np.ndarray, Dict[str, np.ndarray]],
                tolerance: float=2.5e-2):
  """Filter unique fixed points by choosing a representative within tolerance.

  Parameters
  ----------
  candidates: np.ndarray, dict
    The fixed points with the shape of (num_point, num_dim).
  tolerance: float
    tolerance.

  Returns
  -------
  fps_and_ids : tuple
    A 2-tuple of (kept fixed points, ids of kept fixed points).
  """
  if isinstance(candidates, dict):
    element = tuple(candidates.values())[0]
    num_fps = element.shape[0]
    dtype = element.dtype
  else:
    num_fps = candidates.shape[0]
    dtype = candidates.dtype
  keep_ids = np.arange(num_fps)
  if tolerance <= 0.0:
    return candidates, keep_ids
  if num_fps <= 1:
    return candidates, keep_ids
  candidates = tree_map(lambda a: np.asarray(a), candidates, is_leaf=lambda a: isinstance(a, bm.Array))

  # If point A and point B are within identical_tol of each other, and the
  # A is first in the list, we keep A.
  distances = np.asarray(euclidean_distance_jax(candidates, num_fps))
  example_idxs = np.arange(num_fps)
  all_drop_idxs = []
  for fidx in range(num_fps - 1):
    distances_f = distances[fidx, fidx + 1:]
    drop_idxs = example_idxs[fidx + 1:][distances_f <= tolerance]
    all_drop_idxs += list(drop_idxs)
  keep_ids = np.setdiff1d(example_idxs, np.unique(all_drop_idxs))
  if keep_ids.shape[0] > 0:
    unique_fps = tree_map(lambda a: a[keep_ids], candidates)
  else:
    unique_fps = np.array([], dtype=dtype)
  return unique_fps, keep_ids


def keep_unique_jax(candidates, tolerance=2.5e-2):
  """Filter unique fixed points by choosing a representative within tolerance.

  Parameters
  ----------
  candidates: Tesnor
    The fixed points with the shape of (num_point, num_dim).

  Returns
  -------
  fps_and_ids : tuple
    A 2-tuple of (kept fixed points, ids of kept fixed points).
  """
  keep_ids = np.arange(candidates.shape[0])
  if tolerance <= 0.0:
    return candidates, keep_ids
  if candidates.shape[0] <= 1:
    return candidates, keep_ids

  # If point A and point B are within identical_tol of each other, and the
  # A is first in the list, we keep A.
  nfps = candidates.shape[0]
  distances = euclidean_distance_jax(candidates)
  example_idxs = np.arange(nfps)
  all_drop_idxs = []
  for fidx in range(nfps - 1):
    distances_f = distances[fidx, fidx + 1:]
    drop_idxs = example_idxs[fidx + 1:][distances_f <= tolerance]
    all_drop_idxs += list(drop_idxs)
  keep_ids = np.setdiff1d(example_idxs, np.unique(np.asarray(all_drop_idxs)))
  if keep_ids.shape[0] > 0:
    unique_fps = candidates[keep_ids, :]
  else:
    unique_fps = np.array([], dtype=candidates.dtype)
  return unique_fps, keep_ids


def rescale(min_max, scale=0.01):
  """Rescale lim."""
  min_, max_ = min_max
  length = max_ - min_
  min_ -= scale * length
  max_ += scale * length
  return min_, max_
