# -*- coding: utf-8 -*-

"""
This module provides numerical solvers for various differential equations.


"""


from . import dde
from . import fde
from . import ode
from . import sde
from .ode import odeint, get_default_odeint, set_default_odeint
from .sde import sdeint, get_default_sdeint, set_default_sdeint
from .constants import *
from .delay_vars import *
