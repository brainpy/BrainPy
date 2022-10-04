# -*- coding: utf-8 -*-

"""
This module provides analysis tools for differential equations.

- The ``symbolic`` module use SymPy symbolic inference to make analysis of
  low-dimensional dynamical system (only sypport ODEs).
- The ``numeric`` module use numerical optimization function to make analysis
  of high-dimensional dynamical system (support ODEs and discrete systems).
- The ``continuation`` module is the analysis package with numerical continuation methods.
- Moreover, we provide several useful functions in ``stability`` module which may
  help your dynamical system analysis.

Details in the following.
"""

from .base import *

from .highdim.slow_points import *

from .lowdim.lowdim_phase_plane import *
from .lowdim.lowdim_bifurcation import *

from .constants import *
from . import constants as C, stability, plotstyle, utils
