# -*- coding: utf-8 -*-

"""
This module provides analysis tools for differential equations.

- The ``symbolic`` module use SymPy symbolic inference to make analysis of
  low-dimensional dynamical system (only sypport ODEs).
- The ``numeric`` module use numerical optimization function to make analysis
  of high-dimensional dynamical system (support ODEs and discrete systems).
- The ``continuation`` module is the analysis package with numerical continuation methods.
- Moreover, we provide several useful functions in ``stability`` module which may
  help your dynamical system analysis, like:

  >>> get_1d_stability_types()
  ['saddle node', 'stable point', 'unstable point']

Details in the following.
"""

from . import symbolic
from . import continuation
from . import solver
from . import stability
from .solver import *
from .stability import *
