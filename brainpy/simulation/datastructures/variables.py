# -*- coding: utf-8 -*-


from brainpy.backend import ops
from .base import Data

__all__ = [
  'Var',
  'ZerosVar',
  'OnesVar',
  'RandVar',
  'NormalVar',
]


class Var(Data):
  def __init__(self, value):
    super(Var, self).__init__(value=value, train=False)


class ZerosVar(Var):
  def __init__(self, size, dtype=None):
    super(ZerosVar, self).__init__(ops.zeros(size, dtype=dtype))


class OnesVar(Var):
  def __init__(self, size, dtype=None):
    super(OnesVar, self).__init__(ops.zeros(size, dtype=dtype))


class RandVar(Var):
  def __init__(self, size, dtype=None):
    super(RandVar, self).__init__(ops.as_tensor(ops.rand(size=size), dtype=dtype))


class NormalVar(Var):
  def __init__(self, size, dtype=None):
    super(NormalVar, self).__init__(ops.as_tensor(ops.normal(size=size), dtype=dtype))
