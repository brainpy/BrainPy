# -*- coding: utf-8 -*-

# data structure
from .ndarray import *
from .delayvars import *
from .interoperability import *
from .datatypes import *
from .compat_numpy import *
from .compat_tensorflow import *
from .compat_pytorch import *
from .einops import *

# functions
from .activations import *
from . import activations

# operators
from .pre_syn_post import *
from .op_register import *
from . import surrogate, event, sparse, jitconn

# Variable and Objects for object-oriented JAX transformations
from .oo_transform import *

# environment settings
from .modes import *
from .environment import *
from .scales import *
from .others import *

# high-level numpy operations
from . import fft
from . import linalg
from . import random

# taichi operations
from . import tifunc

# others
from . import sharding

import jax.numpy as jnp
from jax import config

del jnp, config

from brainpy._src.math import defaults
from brainpy._src.deprecations import deprecation_getattr
from brainpy._src.dependency_check import import_taichi, import_numba

import_taichi(error_if_not_found=False)
import_numba(error_if_not_found=False)

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
}

__getattr__ = deprecation_getattr(__name__, __deprecations, redirects=defaults.__all__, redirect_module=defaults)
del deprecation_getattr, defaults
