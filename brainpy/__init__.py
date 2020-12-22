# -*- coding: utf-8 -*-

__version__ = "0.2.6"

# "profile" module
from . import profile

# "numpy" module
from . import numpy as numpy

# "connectivity" module
from . import connectivity as connect

# "core_system" module
from . import core_system as core
from .core_system.base import BaseType
from .core_system.base import BaseEnsemble
from .core_system.neurons import NeuType
from .core_system.neurons import NeuGroup
from .core_system.synapses import SynType
from .core_system.synapses import SynConn
from .core_system.synapses import delayed
from .core_system.network import Network
from .core_system import types

# "integration" module
from . import integration
from .integration import integrate

# "dynamics" module
from . import dynamics
from .dynamics import PhasePortraitAnalyzer
from .dynamics import BifurcationAnalyzer

# "tools" module
from . import tools

# "visualization" module
from . import visualization as visualize

# other modules
from . import inputs
from . import measure
from . import running
