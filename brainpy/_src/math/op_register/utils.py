# -*- coding: utf-8 -*-


from functools import partial

import jax.numpy as jnp
from jax import lax
from jax.interpreters import batching
from jax.tree_util import tree_flatten, tree_unflatten

__all__ = [
  'register_general_batching',
]


def _general_batching_rule(prim, args, axes, **kwargs):
  batch_axes, batch_args, non_batch_args = [], {}, {}
  for ax_i, ax in enumerate(axes):
    if ax is None:
      non_batch_args[f'ax{ax_i}'] = args[ax_i]
    else:
      batch_args[f'ax{ax_i}'] = args[ax_i] if ax == 0 else jnp.moveaxis(args[ax_i], ax, 0)
      batch_axes.append(ax_i)

  def f(_, x):
    pars = tuple([(x[f'ax{i}'] if i in batch_axes else non_batch_args[f'ax{i}'])
                  for i in range(len(axes))])
    return 0, prim.bind(*pars, **kwargs)

  _, outs = lax.scan(f, 0, batch_args)
  out_vals, out_tree = tree_flatten(outs)
  out_dim = tree_unflatten(out_tree, (0,) * len(out_vals))
  return outs, out_dim


def register_general_batching(prim):
  batching.primitive_batchers[prim] = partial(_general_batching_rule, prim)




