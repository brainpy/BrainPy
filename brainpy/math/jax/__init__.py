# -*- coding: utf-8 -*-

# necessity to wrap the jax.numpy.ndarray:
# 1. for parameters and variables which want to
#    modify in the JIT mode, this wrapper is necessary.
#    Because, directly change the value in the "self.xx"
#    cannot work.
# 2. In JAX, ndarray is immutable. This wrapper make
#    the index update is the same way with the numpy
#


from . import linalg
from . import random
from .driver import JaxDSDriver as DriverForDS
from .driver import JaxDiffIntDriver as DriverForDiffInt
from .code import *
from .core import *
from .ndarray import *
from .math import *

