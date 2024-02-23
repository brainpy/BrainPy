# -*- coding: utf-8 -*-

from brainpy._src.math.object_transform.base import (
  BrainPyObject as BrainPyObject,
  FunAsObject as FunAsObject
)
from brainpy._src.math.object_transform.function import (Partial as Partial)
from brainpy._src.math.object_transform.base import (
  NodeList as NodeList,
  NodeDict as NodeDict,
  node_dict as node_dict,
  node_list as node_list,
)
from brainpy._src.math.object_transform.variables import (
  Variable as Variable,
  Parameter as Parameter,
  TrainVar as TrainVar,
  VariableView as VariableView,
  VarList as VarList,
  VarDict as VarDict,
  var_list as var_list,
  var_dict as var_dict,
)

from brainpy._src.math.object_transform.autograd import (
  grad as grad,
  vector_grad as vector_grad,
  functional_vector_grad as functional_vector_grad,
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
  scan as scan,
)


from brainpy._src.math.object_transform.jit import (
  jit as jit,
  cls_jit as cls_jit,
)


from brainpy._src.math.object_transform.function import (
  to_object as to_object,
  function as function,
)

from brainpy._src.math.object_transform.tools import (
  eval_shape as eval_shape,
)

