# -*- coding: utf-8 -*-


import jax.lax
import jax.numpy as jnp
from jax import vmap

import brainpy._src.math as bm
from brainpy._src.optimizers.brentq import jax_brentq, ECONVERGED
from . import f_without_jaxarray_return

__all__ = [
  'get_brentq_candidates',
  'brentq_candidates',
  'brentq_roots',
  'brentq_roots2',
  'roots_of_1d_by_x',
  'roots_of_1d_by_xy',
]


def get_brentq_candidates(f, xs, ys):
  f = f_without_jaxarray_return(f)
  xs = bm.as_jax(xs)
  ys = bm.as_jax(ys)
  Y, X = jnp.meshgrid(ys, xs)
  vals = f(X, Y)
  signs = jnp.sign(vals)
  x_ids, y_ids = jnp.where(signs[:-1] * signs[1:] <= 0)
  starts = xs[x_ids]
  ends = xs[x_ids + 1]
  args = ys[y_ids]
  return starts, ends, args


def brentq_candidates(vmap_f, *values, args=()):
  # change the position of meshgrid values
  values = tuple((v.value if isinstance(v, bm.Array) else v) for v in values)
  xs = values[0]
  mesh_values = jnp.meshgrid(*values)
  if jnp.ndim(mesh_values[0]) > 1:
    mesh_values = tuple(jnp.moveaxis(m, 0, 1) for m in mesh_values)
  mesh_values = tuple(m.flatten() for m in mesh_values)
  # function outputs
  signs = jnp.sign(vmap_f(*(mesh_values + args)))
  # compute the selected values
  signs = signs.reshape((xs.shape[0], -1))
  par_len = signs.shape[1]
  signs1 = signs.at[-1].set(1)  # discard the final row
  signs2 = jnp.vstack((signs[1:], signs[:1])).at[-1].set(1)  # discard the first row
  ids = jnp.where((signs1 * signs2).flatten() <= 0)[0]
  x_starts = mesh_values[0][ids]
  x_ends = mesh_values[0][ids + par_len]
  other_vals = tuple(v[ids] for v in mesh_values[1:])
  return x_starts, x_ends, other_vals


def brentq_roots(f, starts, ends, *vmap_args, args=()):
  in_axes = (0, 0, tuple([0] * len(vmap_args)) + tuple([None] * len(args)))
  vmap_f_opt = jax.jit(vmap(jax_brentq(f_without_jaxarray_return(f)), in_axes=in_axes))
  all_args = vmap_args + args
  if len(all_args):
    res = vmap_f_opt(starts, ends, all_args)
  else:
    res = vmap_f_opt(starts, ends, )
  valid_idx = jnp.where(res['status'] == ECONVERGED)[0]
  roots = res['root'][valid_idx]
  vmap_args = tuple(a[valid_idx] for a in vmap_args)
  return roots, vmap_args


def brentq_roots2(vmap_f, starts, ends, *vmap_args, args=()):
  all_args = vmap_args + args
  res = vmap_f(starts, ends, all_args)
  valid_idx = jnp.where(res['status'] == ECONVERGED)[0]
  roots = res['root'][valid_idx]
  vmap_args = tuple(a[valid_idx] for a in vmap_args)
  return roots, vmap_args

def roots_of_1d_by_x(f, candidates, args=()):
  """Find the roots of the given function by numerical methods.
  """
  f = f_without_jaxarray_return(f)
  candidates = candidates.value if isinstance(candidates, bm.Array) else candidates
  args = tuple(a.value if isinstance(candidates, bm.Array) else a for a in args)
  vals = f(candidates, *args)
  signs = jnp.sign(vals)
  zero_sign_idx = jnp.where(signs == 0)[0]
  fps = candidates[zero_sign_idx]
  candidate_ids = jnp.where(signs[:-1] * signs[1:] < 0)[0]
  if len(candidate_ids) <= 0:
    return fps
  starts = candidates[candidate_ids]
  ends = candidates[candidate_ids + 1]
  f_opt = jax.jit(vmap(jax_brentq(f), in_axes=(0, 0, None)))
  res = f_opt(starts, ends, args)
  valid_idx = jnp.where(res['status'] == ECONVERGED)[0]
  fps2 = res['root'][valid_idx]
  return jnp.concatenate([fps, fps2])


def roots_of_1d_by_xy(f, starts, ends, args):
  f_opt = jax.jit(vmap(jax_brentq(f_without_jaxarray_return(f))))
  res = f_opt(starts, ends, (args,))
  valid_idx = jnp.where(res['status'] == ECONVERGED)[0]
  xs = res['root'][valid_idx]
  ys = args[valid_idx]
  return xs, ys


