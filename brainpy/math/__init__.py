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
- device switching
- default type switching
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
# from .parallels import *

# settings
from . import setting
from .setting import *
from .function import *

# functions
from .activations import *
from . import activations
from .compat import *


def get_dint():
  """Get default int type."""
  return int_


def get_dfloat():
  """Get default float type."""
  return float_


def get_dcomplex():
  """Get default complex type."""
  return complex_


def set_dint(int_type):
  """Set default int type."""
  global int_
  assert isinstance(int_type, type)
  int_ = int_type


def set_dfloat(float_type):
  """Set default float type."""
  global float_
  assert isinstance(float_type, type)
  float_ = float_type


def set_dcomplex(complex_type):
  """Set default complex type."""
  global complex_
  assert isinstance(complex_type, type)
  complex_ = complex_type

