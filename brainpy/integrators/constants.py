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

    'NAME_PREFIX',
]

# Ito SDE
# ---
#
ITO_SDE = 'Ito'

# Stratonovich SDE
# ---
#
STRA_SDE = 'Stratonovich'

SUPPORTED_SDE_TYPE = [
    ITO_SDE,
    STRA_SDE
]

# Scalar Wiener process
# ----
#
SCALAR_WIENER = 'scalar_wiener'

# Vector Wiener process
# ----
#
VECTOR_WIENER = 'vector_wiener'

SUPPORTED_WIENER_TYPE = [
    SCALAR_WIENER,
    VECTOR_WIENER
]

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

NAME_PREFIX = '_brainpy_numint_of_'
