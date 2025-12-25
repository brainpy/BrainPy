# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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

import brainstate
import braintools

random = brainstate.random
surrogate = braintools.surrogate
import jax.numpy as jnp
from jax import config

from . import activations
# high-level numpy operations
from . import fft
from . import linalg
# others
from . import sharding
from . import event, sparse, jitconn
# functions
from .activations import *
from .compat_numpy import *
from .compat_pytorch import *
from .compat_tensorflow import *
from .datatypes import *
from .delayvars import *
from .einops import *
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

del jnp, config
from brainpy.deprecations import deprecation_getattr

from .defaults import defaults

__getattr__ = deprecation_getattr(
    __name__,
    {},
    redirects=[
        'mode', 'membrane_scaling', 'dt', 'bool_', 'int_', 'float_', 'complex_', 'bp_object_as_pytree',
        'numpy_func_return'
    ],
    redirect_module=defaults
)
del deprecation_getattr, defaults, brainstate
