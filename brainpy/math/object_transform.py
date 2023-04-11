# -*- coding: utf-8 -*-

from brainpy._src.math.object_transform.autograd import (
  grad as grad,
  vector_grad as vector_grad,
  jacobian as jacobian,
  jacrev as jacrev,
  jacfwd as jacfwd,
  hessian as hessian,
)

from brainpy._src.math.object_transform.controls import (
  make_loop as make_loop,
  make_while as make_while,
  make_cond as make_cond,
  cond as cond,
  ifelse as ifelse,
  for_loop as for_loop,
  while_loop as while_loop,
)


from brainpy._src.math.object_transform.jit import (
  jit as jit,
  cls_jit as cls_jit,
)


from brainpy._src.math.object_transform.function import (
  to_object as to_object,
  function as function,
)
