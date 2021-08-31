# -*- coding: utf-8 -*-

"""
The base module for whole BrainPy ecosystem.

This module provide the most fundamental class ``Base``,
and its associated helper class ``Collector``.

For each instance of "Base" class, users can retrieve all
the variables (or trainable variables), integrators, and nodes.

This module also provide a ``Function`` class to wrap user-defined
functions. In each function, maybe several nodes are used, and
users can initialize a ``Function`` by providing the nodes used
in the function. Unfortunately, ``Function`` class does not have
the ability to gather nodes automatically.
"""

from brainpy.base.base import *
from brainpy.base.collector import *
from brainpy.base.function import *
