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

from .symbolic.old_phase_plane import *
from .symbolic.old_bifurcation import *

from .symbolic.lowdim_phase_plane import *
from .symbolic.lowdim_bifurcation import *

from .numeric.fixed_points import *
from .numeric.lowdim_phase_plane import *
from .numeric.lowdim_bifurcation import *

from . import constants
C = constants
from . import stability
from . import utils
