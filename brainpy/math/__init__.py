# -*- coding: utf-8 -*-


"""
The ``math`` module for whole BrainPy ecosystem.
This module provides basic mathematical operations, including:

- numpy-like array operations
- linear algebra functions
- random sampling functions
- discrete fourier transform functions
- compilations of ``jit``, ``vmap``, ``pmap`` for class objects
- automatic differentiation of ``grad``, ``jacocian``, ``hessian``, etc. for class objects

It also provides fundamental tools to construct brain models, including:

- loss functions
- activation functions
- optimizers
- connectivity functions
- initialization functions

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

# high-level numpy operations
from .numpy_ops import *
from .operators import *
from . import fft
from . import linalg
from . import random

# JAX transformations extended on class objects
from .autograd import *
from .controls import *
from .jit import *
from .parallels import *

# settings
from . import setting
from .setting import *
from .function import *

# functions
from . import numpy
from .activations import *
from . import activations
from .compact import *

