# -*- coding: utf-8 -*-

__version__ = "1.1.0-alpha"


# "math" module
from . import math


# "integrators" module
from . import integrators
from .integrators import ode
from .integrators import sde
from .integrators import dde
from .integrators import fde
from .integrators.wrapper import *


# "simulation" module
from . import simulation
from .simulation import connectivity as connect
from .simulation.brainobjects import *
from .simulation.every import *
from .simulation.monitor import *


# "dnn" module
if math.get_backend_name() == 'jax':
  from . import dnn


# "analysis" module
from . import analysis


# "visualization" module
from . import visualization as visualize


# other modules
from . import errors
from . import inputs
from . import measure
from . import running
from . import tools


# deprecated
from . import backend
from . import ops
