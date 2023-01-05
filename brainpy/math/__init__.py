# -*- coding: utf-8 -*-


# data structure
from .ndarray import *
from .delayvars import *
from .arrayoperation import *

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

import brainpy._src.math.fft as bm_fft
fft.__dict__['fft'] = bm_fft.fft
fft.__dict__['fft2'] = bm_fft.fft2
fft.__dict__['fftfreq'] = bm_fft.fftfreq
fft.__dict__['fftn'] = bm_fft.fftn
fft.__dict__['fftshift'] = bm_fft.fftshift
fft.__dict__['hfft'] = bm_fft.hfft
fft.__dict__['ifft'] = bm_fft.ifft
fft.__dict__['ifft2'] = bm_fft.ifft2
fft.__dict__['ifftn'] = bm_fft.ifftn
fft.__dict__['ifftshift'] = bm_fft.ifftshift
fft.__dict__['ihfft'] = bm_fft.ihfft
fft.__dict__['irfft'] = bm_fft.irfft
fft.__dict__['irfft2'] = bm_fft.irfft2
fft.__dict__['irfftn'] = bm_fft.irfftn
fft.__dict__['rfft'] = bm_fft.rfft
fft.__dict__['rfft2'] = bm_fft.rfft2
fft.__dict__['rfftfreq'] = bm_fft.rfftfreq
fft.__dict__['rfftn'] = bm_fft.rfftn
del bm_fft

import brainpy._src.math.linalg as bm_linalg
linalg.__dict__['cholesky'] = bm_linalg.cholesky
linalg.__dict__['cond'] = bm_linalg.cond
linalg.__dict__['det'] = bm_linalg.det
linalg.__dict__['eig'] = bm_linalg.eig
linalg.__dict__['eigh'] = bm_linalg.eigh
linalg.__dict__['eigvals'] = bm_linalg.eigvals
linalg.__dict__['eigvalsh'] = bm_linalg.eigvalsh
linalg.__dict__['inv'] = bm_linalg.inv
linalg.__dict__['svd'] = bm_linalg.svd
linalg.__dict__['lstsq'] = bm_linalg.lstsq
linalg.__dict__['matrix_power'] = bm_linalg.matrix_power
linalg.__dict__['matrix_rank'] = bm_linalg.matrix_rank
linalg.__dict__['norm'] = bm_linalg.norm
linalg.__dict__['pinv'] = bm_linalg.pinv
linalg.__dict__['qr'] = bm_linalg.qr
linalg.__dict__['solve'] = bm_linalg.solve
linalg.__dict__['slogdet'] = bm_linalg.slogdet
linalg.__dict__['tensorinv'] = bm_linalg.tensorinv
linalg.__dict__['tensorsolve'] = bm_linalg.tensorsolve
linalg.__dict__['multi_dot'] = bm_linalg.multi_dot
del bm_linalg
