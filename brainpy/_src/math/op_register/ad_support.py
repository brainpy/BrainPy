import functools
from functools import partial

from jax import tree_util
from jax.core import Primitive
from jax.interpreters import ad

__all__ = [
  'defjvp',
]


def defjvp(primitive, *jvp_rules):
  """Define JVP rules for any JAX primitive.

  This function is similar to ``jax.interpreters.ad.defjvp``.
  However, the JAX one only supports primitive with ``multiple_results=False``.
  ``brainpy.math.defjvp`` enables to define the independent JVP rule for
  each input parameter no matter ``multiple_results=False/True``.

  For examples, please see ``test_ad_support.py``.

  Args:
    primitive: Primitive, XLACustomOp.
    *jvp_rules: The JVP translation rule for each primal.
  """
  assert isinstance(primitive, Primitive)
  if primitive.multiple_results:
    ad.primitive_jvps[primitive] = partial(_standard_jvp, jvp_rules, primitive)
  else:
    ad.primitive_jvps[primitive] = partial(ad.standard_jvp, jvp_rules, primitive)


def _standard_jvp(jvp_rules, primitive: Primitive, primals, tangents, **params):
  assert primitive.multiple_results
  val_out = tuple(primitive.bind(*primals, **params))
  tree = tree_util.tree_structure(val_out)
  tangents_out = []
  for rule, t in zip(jvp_rules, tangents):
    if rule is not None and type(t) is not ad.Zero:
      r = tuple(rule(t, *primals, **params))
      tangents_out.append(r)
      assert tree_util.tree_structure(r) == tree
  return val_out, functools.reduce(_add_tangents,
                                   tangents_out,
                                   tree_util.tree_map(lambda a: ad.Zero.from_value(a), val_out))


def _add_tangents(xs, ys):
  return tree_util.tree_map(ad.add_tangents, xs, ys, is_leaf=lambda a: isinstance(a, ad.Zero))

