# -*- coding: utf-8 -*-


"""
This module provides APIs for parallel brain simulations.
"""

from . import jax_multiprocessing
from . import native_multiprocessing
from . import pathos_multiprocessing
from . import constants


__all__ = (native_multiprocessing.__all__ +
           pathos_multiprocessing.__all__ +
           jax_multiprocessing.__all__ +
           constants.__all__)


from .jax_multiprocessing import *
from .native_multiprocessing import *
from .pathos_multiprocessing import *
from .constants import *

