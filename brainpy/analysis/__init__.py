# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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

from . import constants as C, stability, plotstyle, utils
from .base import *
from .constants import *
from .constants import *
from .highdim.slow_points import *
from .lowdim.lowdim_bifurcation import *
from .lowdim.lowdim_phase_plane import *
