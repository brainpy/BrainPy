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
from . import sharding

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

from brainpy._src.deprecations import deprecation_getattr
__deprecations = {
  "sparse_matmul": ("brainpy.math.sparse_matmul is deprecated. Use brainpy.math.sparse.seg_matmul instead.",
                    sparse.seg_matmul),
  'csr_matvec': ("brainpy.math.csr_matvec is deprecated. Use brainpy.math.sparse.csrmv instead.",
                 sparse.csrmv),
  'event_matvec_prob_conn_homo_weight': ("brainpy.math.event_matvec_prob_conn_homo_weight is deprecated. "
                                         "Use brainpy.math.jitconn.event_mv_prob_homo instead.",
                                         jitconn.event_mv_prob_homo),
  'event_matvec_prob_conn_uniform_weight': ("brainpy.math.event_matvec_prob_conn_uniform_weight is deprecated. "
                                            "Use brainpy.math.jitconn.event_mv_prob_uniform instead.",
                                            jitconn.event_mv_prob_uniform),
  'event_matvec_prob_conn_normal_weight': ("brainpy.math.event_matvec_prob_conn_normal_weight is deprecated. "
                                           "Use brainpy.math.jitconn.event_mv_prob_normal instead.",
                                           jitconn.event_mv_prob_normal),
  'matvec_prob_conn_homo_weight': ("brainpy.math.matvec_prob_conn_homo_weight is deprecated. "
                                   "Use brainpy.math.jitconn.mv_prob_homo instead.",
                                   jitconn.mv_prob_homo),
  'matvec_prob_conn_uniform_weight': ("brainpy.math.matvec_prob_conn_uniform_weight is deprecated. "
                                      "Use brainpy.math.jitconn.mv_prob_uniform instead.",
                                      jitconn.mv_prob_uniform),
  'matvec_prob_conn_normal_weight': ("brainpy.math.matvec_prob_conn_normal_weight is deprecated. "
                                     "Use brainpy.math.jitconn.mv_prob_normal instead.",
                                     jitconn.mv_prob_normal),
  'cusparse_csr_matvec': ("brainpy.math.cusparse_csr_matvec is deprecated. "
                          "Use brainpy.math.sparse.csrmv instead.",
                          sparse.csrmv),
  'cusparse_coo_matvec': ("brainpy.math.cusparse_coo_matvec is deprecated. "
                          "Use brainpy.math.sparse.coomv instead.",
                          sparse.coomv),
  'coo_to_csr': ("brainpy.math.coo_to_csr is deprecated. "
                 "Use brainpy.math.sparse.coo_to_csr instead.",
                 sparse.coo_to_csr),
  'csr_to_coo': ("brainpy.math.csr_to_coo is deprecated. "
                 "Use brainpy.math.sparse.csr_to_coo instead.",
                 sparse.csr_to_coo),
  'csr_to_dense': ("brainpy.math.csr_to_dense is deprecated. "
                   "Use brainpy.math.sparse.csr_to_dense instead.",
                   sparse.csr_to_dense),
  'event_csr_matvec': ("brainpy.math.event_csr_matvec is deprecated. "
                       "Use brainpy.math.event.csr_to_dense instead.",
                       event.csrmv),
  'event_info': ("brainpy.math.event_info is deprecated. "
                 "Use brainpy.math.event.info instead.",
                 event.info),
}
__getattr__ = deprecation_getattr(__name__, __deprecations)
del deprecation_getattr
