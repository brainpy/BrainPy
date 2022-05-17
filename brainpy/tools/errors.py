# -*- coding: utf-8 -*-


from jax.lax import cond
from jax.experimental.host_callback import id_tap

__all__ = [
  'check_error_in_jit'
]


def _make_err_func(f):
  f2 = lambda arg, transforms: f(arg)

  def err_f(x):
    id_tap(f2, x)
    return
  return err_f


def check_error_in_jit(pred, err_f, err_arg=None):
  """Check errors in a jit function.

  Parameters
  ----------
  pred: bool
    The boolean prediction.
  err_f: callable
    The error function, which raise errors.
  err_arg: any
    The arguments which passed into `err_f`.
  """
  cond(pred, _make_err_func(err_f), lambda _: None, err_arg)


