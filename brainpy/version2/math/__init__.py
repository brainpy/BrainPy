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
- dedicated operators for brain dynamics modeling
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


from . import activations
from . import random, linalg, fft
from . import surrogate, event, sparse, jitconn
# functions
from .activations import *
from .compat_numpy import *
from .compat_tensorflow import *
# high-level numpy operations
from .datatypes import *
from .delayvars import *
from .environment import *
from .interoperability import *
# environment settings
from .modes import *
# data structure
from .ndarray import *
# Variable and Objects for object-oriented JAX transformations
from .object_transform import *
from .others import *
# operators
from .pre_syn_post import *
from .scales import *
