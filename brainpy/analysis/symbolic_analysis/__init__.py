# -*- coding: utf-8 -*-


"""
Dynamics analysis with the aid of `SymPy <https://www.sympy.org/en/index.html>`_  symbolic_analysis inference.

This module provide basic dynamics analysis for low-dimensional dynamical systems, including

- phase plane analysis (1d or 2d systems)
- bifurcation analysis (1d or 2d systems)
- fast slow bifurcation analysis (2d or 3d systems)

"""

from .base import *
from .bifurcation import *
from .phase_plane import *
from .trajectory import *
