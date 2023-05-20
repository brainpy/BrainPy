
from jax.tree_util import tree_flatten, tree_map, tree_unflatten

from brainpy import math as bm
from brainpy._src.dynsys import DynamicalSystemNS

try:
  import flax  # noqa
except:
  flax = None


__all__ = [
  'FromFlax',
  'ToFlax',
]


def _as_jax(a):
  if isinstance(a, bm.Array):
    return a.value
  else:
    return a


def _is_bp(a):
  return isinstance(a, bm.Array)


class FromFlax(DynamicalSystemNS):
  def __init__(self, flax_module, *module_args, **module_kwargs):
    super().__init__()
    self.flax_module = flax_module
    params = self.flax_module.init(bm.random.split_key(),
                                   *tree_map(_as_jax, module_args, is_leaf=_is_bp),
                                   **tree_map(_as_jax, module_kwargs, is_leaf=_is_bp))
    leaves, self._tree = tree_flatten(params)
    self.variables = bm.VarList(tree_map(bm.TrainVar, leaves))

  def update(self, *args, **kwargs):
    params = tree_unflatten(self._tree, [v.value for v in self.variables])
    return self.flax_module.apply(params,
                                  *tree_map(_as_jax, args, is_leaf=_is_bp),
                                  **tree_map(_as_jax, kwargs, is_leaf=_is_bp))

  def reset_state(self, *args, **kwargs):
    pass


if flax is not None:
  class ToFlax(flax.linen.Module):
    pass


else:
  class ToFlax(object):
    def __init__(self, *args, **kwargs):
      raise ModuleNotFoundError('"flax" is not installed, or importing "flax" has errors. Please check.')


