# -*- coding: utf-8 -*-
"""
The ``brainpy_object`` module for whole BrainPy ecosystem.

- This module provides the most fundamental class ``BrainPyObject``,
  and its associated helper class ``Collector`` and ``ArrayCollector``.
- For each instance of "BrainPyObject" class, users can retrieve all
  the variables (or trainable variables), integrators, and nodes.
- This module also provides a ``FunAsObject`` class to wrap user-defined
  functions. In each function, maybe several nodes are used, and
  users can initialize a ``FunAsObject`` by providing the nodes used
  in the function. Unfortunately, ``FunAsObject`` class does not have
  the ability to gather nodes automatically.

Details please see the following.
"""

from . import (
  base_object,
  base_transform,
  collector,
  autograd,
  controls,
  jit,
  function,
)

__all__ = (
    autograd.__all__
    + controls.__all__
    + jit.__all__
    + function.__all__
    + base_object.__all__
    + base_transform.__all__
    + collector.__all__
)

from .autograd import *
from .controls import *
from .jit import *
from .function import *
from .base_object import *
from .base_transform import *
from .collector import *
