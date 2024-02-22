from brainpy._src.dependency_check import import_taichi_else_None
if import_taichi_else_None() is not None:
  from ._matvec import *
  from ._event_matvec import *