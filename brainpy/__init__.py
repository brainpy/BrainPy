# -*- coding: utf-8 -*-

__version__ = "1.1.3"

try:
  import jax
  jax.config.update('jax_platform_name', 'cpu')
except ModuleNotFoundError:
  pass

# "base" module
from . import base
from .base.base import Base
from .base.collector import Collector, TensorCollector


# "math" module
from . import math


# "integrators" module
from . import integrators
from .integrators import ode
from .integrators import sde
from .integrators.ode import odeint
from .integrators.ode import set_default_odeint
from .integrators.ode import get_default_odeint
from .integrators.sde import sdeint
from .integrators.sde import set_default_sdeint
from .integrators.sde import get_default_sdeint


# "simulation" module
from . import simulation
from .simulation.brainobjects import *
from .simulation.monitor import *
# submodules
from .simulation import brainobjects
from .simulation import connect
from .simulation import initialize
from .simulation import layers
from .simulation import inputs
from .simulation import measure


# "analysis" module
from . import analysis
from .analysis import continuation
from .analysis import numeric
from .analysis import symbolic


# "visualization" module
from . import visualization as visualize


# other modules
from . import errors
from . import running
from . import tools

