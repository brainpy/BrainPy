import warnings
import jax
from brainpy._src.math.ndarray import VariableStack
from brainpy._src.math.object_transform.naming import (cache_stack,
                                                       get_stack_cache)


def dynvar_deprecation(dyn_vars=None):
  if dyn_vars is not None:
    warnings.warn('\n'
                  'From brainpy>=2.4.0, users no longer need to provide ``dyn_vars`` into '
                  'transformation functions like "jit", "grad", "for_loop", etc. '
                  'Because these transformations are capable of automatically collecting them.',
                  UserWarning)


def node_deprecation(child_objs=None):
  if child_objs is not None:
    warnings.warn('\n'
                  'From brainpy>=2.4.0, users no longer need to provide ``child_objs`` into '
                  'transformation functions like "jit", "grad", "for_loop", etc. '
                  'Because these transformations are capable of automatically collecting them.',
                  UserWarning)


def abstract(x):
  if callable(x):
    return x
  else:
    return jax.api_util.shaped_abstractify(x)


def evaluate_dyn_vars(f, *args, **kwargs):
  # TODO: better way for cache mechanism
  stack = get_stack_cache(f)
  if stack is None:
    with jax.ensure_compile_time_eval():
      args, kwargs = jax.tree_util.tree_map(abstract, (args, kwargs))
      with VariableStack() as stack:
        _ = jax.eval_shape(f, *args, **kwargs)
      cache_stack(f, stack)  # cache
  return stack



