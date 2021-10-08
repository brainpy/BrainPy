# -*- coding: utf-8 -*-

# necessity to wrap the jax.numpy.ndarray:
# 1. for parameters and variables which want to
#    modify in the JIT mode, this wrapper is necessary.
#    Because, directly change the value in the "self.xx"
#    cannot work.
# 2. In JAX, ndarray is immutable. This wrapper make
#    the index update is the same way with the numpy
#


from . import activations
from . import fft
from . import linalg
from . import losses
from . import optimizers
from . import random

from .activations import *
from .code import *
from .controls import *
from .compilation import *
from .function import *
from .gradient import *
from .jaxarray import *
from .ops import *
