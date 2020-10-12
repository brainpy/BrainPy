# -*- coding: utf-8 -*-

__version__ = "1.0.0"

# "profile" module
from . import profile

# "_numpy" module
from . import numpy as numpy

# "connectivity" module
from . import connectivity as connect

# "core_system" module
from . import core_system as core
from .core_system.base_objects import *
from .core_system.neuron_group import *
from .core_system.synapse_connection import *
from .core_system.network import *
from .core_system import types

# "dynamics" module
from . import integration
from .integration import integrate
from .integration import DiffEquation
from .integration.integrator import *

# "tools" module
from . import tools

# "visualization" module
from . import visualization as visualize

# other modules
from . import inputs
from . import measure
from . import running
