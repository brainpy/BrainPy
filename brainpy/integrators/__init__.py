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
This module provides numerical solvers for various differential equations,
including:

- ordinary differential equations (ODEs)
- stochastic differential equations (SDEs)
- fractional differential equations (FDEs)
- delay differential equations (DDEs)

Details please see the following.
"""

# FDE tools
from . import fde
# ODE tools
from . import ode
# PDE tools
from . import pde
# SDE tools
from . import sde
# basic tools
from .base import *
from .constants import *
from .fde.base import FDEIntegrator
from .fde.generic import (fdeint,
                          get_default_fdeint,
                          set_default_fdeint,
                          register_fde_integrator)
from .joint_eq import *
from .ode.base import ODEIntegrator
from .ode.generic import (odeint,
                          get_default_odeint,
                          set_default_odeint,
                          register_ode_integrator)
from .runner import *
from .sde.base import SDEIntegrator
from .sde.generic import (sdeint,
                          get_default_sdeint,
                          set_default_sdeint,
                          register_sde_integrator)
