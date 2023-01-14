# -*- coding: utf-8 -*-


# data structure
from .ndarray import *
from .delayvars import *
from .arrayoperation import *
from .arraycompatible import *

# functions
from .activations import *
from . import activations

# operators
from .operators import *
from . import surrogate

# Variable and Objects for object-oriented JAX transformations
from .object_base import *
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
from jax import config

bool_ = jnp.bool_
'''Default bool data type.'''

int_ = jnp.int64 if config.read('jax_enable_x64') else jnp.int32
'''Default integer data type.'''

float_ = jnp.float64 if config.read('jax_enable_x64') else jnp.float32
'''Default float data type.'''

complex_ = jnp.complex128 if config.read('jax_enable_x64') else jnp.complex64
'''Default complex data type.'''

del jnp, config

# high-level numpy operations
from . import fft
from . import linalg
from . import random

from brainpy._src.math.surrogate.compt import (
  spike_with_sigmoid_grad as spike_with_sigmoid_grad,
  spike_with_linear_grad as spike_with_linear_grad,
  spike_with_gaussian_grad as spike_with_gaussian_grad,
  spike_with_mg_grad as spike_with_mg_grad,
  spike2_with_sigmoid_grad as spike2_with_sigmoid_grad,
  spike2_with_linear_grad as spike2_with_linear_grad,
)
