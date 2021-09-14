# -*- coding: utf-8 -*-

"""
This module provides analysis tools for ordinary differential equations.

The first part is the analysis package with SymPy symbolic inference.

The second part is the analysis package with numerical continuation methods.

Moreover, we provide several useful functions which may help your dynamical
system analysis, like:

>>> get_1d_stability_types()
['saddle node', 'stable point', 'unstable point']

"""

from . import sym_analysis
from . import continuation
from . import solver
from . import stability
from .solver import *
from .stability import *
