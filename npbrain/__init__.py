# -*- coding: utf-8 -*-

__version__ = "1.0.0"

# "profile" module
from . import profile

# "core" module
from . import core
from .core import integrator
from .core.integrator import integrate
from .core.network import *
from .core.neuron_group import *
from .core.synapse_connection import *
from .core import types

# "utils" module
from . import utils
from .utils import helper
from .utils import inputs
from .utils import measure

#
# # reload functions
# def _reload():
#     global judge_spike
#     global clip
#
#     judge_spike = get_spike_judger()
#     clip = get_clip()
#
#

