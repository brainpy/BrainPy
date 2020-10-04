# -*- coding: utf-8 -*-

"""
This module defines the core of the framework, including the
abstraction of ``Neurons``, ``Synapses``, ``Monitor``, ``Network``,
and numerical integrator methods.

The core is so small, and the overall framework is easy to
understand. Using it, you can easily write your own neurons,
synapses, etc.
"""


from .base import *
from .types import *
from .neuron_group import *
from .synapse_connection import *
from .network import *



