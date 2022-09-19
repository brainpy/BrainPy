# -*- coding: utf-8 -*-


from jax.lax import cond
from jax.experimental.host_callback import id_tap

__all__ = [
  'check_error_in_jit'
]


def check_error_in_jit(pred, err_fun, err_arg=None):
  """Check errors in a jit function.

  Parameters
  ----------
  pred: bool
    The boolean prediction.
  err_fun: callable
    The error function, which raise errors.
  err_arg: any
    The arguments which passed into `err_f`.
  """
  from brainpy.math.remove_vmap import remove_vmap

  def err_f(x):
    id_tap(lambda arg, transforms: err_fun(arg), x)
    return
  cond(remove_vmap(pred), err_f, lambda _: None, err_arg)

