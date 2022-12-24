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


# data structure
from .ndarray import *
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
from . import surrogate

# JAX transformations for Variable and class objects
from .object_transform import *


# environment settings
from .modes import *
from .environment import *
from .others import *

mode = NonBatchingMode()
'''Default computation mode.'''

dt = 0.1
'''Default time step.'''

import jax.numpy as jnp

bool_ = jnp.bool_
'''Default bool data type.'''

int_ = jnp.int32
'''Default integer data type.'''

float_ = jnp.float32
'''Default float data type.'''

complex_ = jnp.complex64
'''Default complex data type.'''

del jnp
