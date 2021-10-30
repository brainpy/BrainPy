# -*- coding: utf-8 -*-

"""
This module provides numerical solvers for various differential equations,
including:

- ordinary differential equations (ODEs)
- stochastic differential equations (SDEs)

Details please see the following.
"""


from . import dde
from . import fde
from . import ode
from . import sde
from .ode import odeint, get_default_odeint, set_default_odeint, ODEIntegrator
from .sde import sdeint, get_default_sdeint, set_default_sdeint, SDEIntegrator
from .constants import *
from .delay_vars import *
