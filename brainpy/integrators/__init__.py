# -*- coding: utf-8 -*-

"""
This module provides numerical solvers for various differential equations,
including:

- ordinary differential equations (ODEs)
- stochastic differential equations (SDEs)
- delay differential equations (DDEs)
- fractional differential equations (FDEs)

Details please see the following.
"""

# basic tools
from .base import *
from .constants import *
from .joint_eq import *
from .runner import *

# ODE tools
from . import ode
from .ode.base import ODEIntegrator
from .ode.generic import (odeint,
                          get_default_odeint,
                          set_default_odeint,
                          register_ode_integrator)

# SDE tools
from . import sde
from .sde.base import SDEIntegrator
from .sde.generic import (sdeint,
                          get_default_sdeint,
                          set_default_sdeint,
                          register_sde_integrator)

# DDE tools
from . import dde
from .dde.base import DDEIntegrator
from .dde.generic import (ddeint,
                          get_default_ddeint,
                          set_default_ddeint,
                          register_dde_integrator)

# FDE tools
from . import fde
from .fde.base import FDEIntegrator
from .fde.generic import (fdeint,
                          get_default_fdeint,
                          set_default_fdeint,
                          register_fde_integrator)


# PDE tools
from . import pde
