# -*- coding: utf-8 -*-

import brainpy.math as bm

__all__ = [
  'f_without_jaxarray_return',
]


def f_without_jaxarray_return(f):
  def f2(*args, **kwargs):
    r = f(*args, **kwargs)
    return r.value if isinstance(r, bm.JaxArray) else r

  return f2


