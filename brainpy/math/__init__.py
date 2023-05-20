# -*- coding: utf-8 -*-


# data structure
from .ndarray import *
from .delayvars import *
from .interoperability import *
from .datatypes import *
from .compat_numpy import *
from .compat_tensorflow import *
from .compat_pytorch import *

# functions
from .activations import *
from . import activations

# operators
from .pre_syn_post import *
from .op_register import *
from . import surrogate, event, sparse, jitconn

# Variable and Objects for object-oriented JAX transformations
from .object_base import *
from .object_transform import *

# environment settings
from .modes import *
from .environment import *
from .others import *

# high-level numpy operations
from . import fft
from . import linalg
from . import random

# others
import jax.numpy as jnp
from jax import config

mode = NonBatchingMode()
'''Default computation mode.'''

dt = 0.1
'''Default time step.'''

bool_ = jnp.bool_
'''Default bool data type.'''

int_ = jnp.int64 if config.read('jax_enable_x64') else jnp.int32
'''Default integer data type.'''

float_ = jnp.float64 if config.read('jax_enable_x64') else jnp.float32
'''Default float data type.'''

complex_ = jnp.complex128 if config.read('jax_enable_x64') else jnp.complex64
'''Default complex data type.'''

del jnp, config


from brainpy._src.math.surrogate._compt import (
  spike_with_sigmoid_grad as spike_with_sigmoid_grad,
  spike_with_linear_grad as spike_with_linear_grad,
  spike_with_gaussian_grad as spike_with_gaussian_grad,
  spike_with_mg_grad as spike_with_mg_grad,
)
