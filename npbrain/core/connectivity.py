# -*- coding: utf-8 -*-

"""
Connection toolkit.
"""

import numba as nb
import numpy as onp

from .. import _numpy as np
from .. import profile

__all__ = [
    # conn formatter
    'from_matrix',
    'from_ij',
    'pre2post',
    'post2pre',
    'pre2syn',
    'post2syn',

]


# -----------------------------------
# formatter of conn
# -----------------------------------
