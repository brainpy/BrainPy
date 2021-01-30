# -*- coding: utf-8 -*-

__version__ = "0.3.5"

# "profile" module
from . import profile

# "backend" module
from . import backend

# "connectivity" module
from . import connectivity as connect

# "core" module
from . import core as core
from .core.base import ObjType
from .core.base import Ensemble
from .core.neurons import NeuType
from .core.neurons import NeuGroup
from .core.synapses import SynType
from .core.synapses import SynConn
from .core.synapses import delayed
from .core.network import Network
from .core import types
from .core.types import ObjState
from .core.types import NeuState
from .core.types import SynState

# "integration" module
from . import integration
from .integration import integrate

# "analysis" module
from . import analysis

# "tools" module
from . import tools

# "visualization" module
from . import visualization as visualize

# other modules
from . import inputs
from . import measure
from . import running

