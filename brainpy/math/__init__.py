# -*- coding: utf-8 -*-


"""
The ``math`` module for whole BrainPy ecosystem.
This module provides basic mathematical operations, including:

- numpy-like array operations
- linear algebra functions
- random sampling functions
- discrete fourier transform functions
- just-in-time compilation for class objects
- automatic differentiation for class objects
- dedicated operators for brain dynamics
- activation functions
- device/dtype switching
- and others

Details in the following.
"""


# necessity to wrap the jax.numpy.ndarray:
# 1. for parameters and variables which want to
#    modify in the JIT mode, this wrapper is necessary.
#    Because, directly change the value in the "self.xx"
#    cannot work.
# 2. In JAX, ndarray is immutable. This wrapper make
#    the index update is the same way with the numpy
#


# data structure
from .jaxarray import *
from .delayvars import *

# functions
from .activations import *
from . import activations

# high-level numpy operations
from .numpy_ops import *
from .index_tricks import *
from . import fft
from . import linalg
from . import random

# operators
from .operators import *

# JAX transformations extended on Variable and class objects
from .autograd import *
from .controls import *
from .jit import *

# settings
from . import setting
from .setting import *
from .function import *
