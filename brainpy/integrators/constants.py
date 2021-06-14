# -*- coding: utf-8 -*-


__all__ = [
  'SUPPORTED_VAR_TYPE',
  'SCALAR_VAR',
  'POPU_VAR',
  'SYSTEM_VAR',

  'SUPPORTED_WIENER_TYPE',
  'SCALAR_WIENER',
  'VECTOR_WIENER',

  'SUPPORTED_SDE_TYPE',
  'ITO_SDE',
  'STRA_SDE',

  'DE_PREFIX',
]

# Ito SDE
# ---
# The SDE integral proposed by Ito in 1940s.
ITO_SDE = 'Ito'

# Stratonovich SDE
# ---
# The SDE integral proposed by Stratonovich in 1960s.
STRA_SDE = 'Stratonovich'

SUPPORTED_SDE_TYPE = [
  ITO_SDE,
  STRA_SDE
]

# ------------------------------------------------------

# Scalar Wiener process
# ----
#
SCALAR_WIENER = 'scalar'

# Vector Wiener process
# ----
#
VECTOR_WIENER = 'vector'

SUPPORTED_WIENER_TYPE = [
  SCALAR_WIENER,
  VECTOR_WIENER
]

# ------------------------------------------------------

# Denotes each variable is a scalar variable
# -------
# For example:
#
# def derivative(a, b, t):
#     ...
#     return da, db
#
# The "a" and "b" are scalars: a=1, b=2
#
SCALAR_VAR = 'scalar'

# Denotes each variable is a homogeneous population
# -------
# For example:
#
# def derivative(a, b, t):
#     ...
#     return da, db
#
# The "a" and "b" are vectors or matrix:
#    a = np.array([1,2]),  b = np.array([3,4])
# or,
#    a = np.array([[1,2], [2,1]]),  b=np.array([[3,4], [4,3]])
#
POPU_VAR = 'population'

# Denotes each variable is a system
# ------
# For example, the above defined differential equations can be defined as:
#
# def derivative(x, t):
#     a, b = x
#     ...
#     dx = np.array([da, db])
#     return dx
SYSTEM_VAR = 'system'

SUPPORTED_VAR_TYPE = [
  SCALAR_VAR,
  POPU_VAR,
  SYSTEM_VAR,
]

# ------------------------------------------------------

# Differential equation type
# ----------
#

DE_PREFIX = '_brainpy_intg_of_'
ODE_PREFIX = 'ode_brainpy_intg_of_'
SDE_PREFIX = 'sde_brainpy_intg_of_'
