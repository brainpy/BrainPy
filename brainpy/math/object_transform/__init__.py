# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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

from brainstate.transform import ProgressBar

from .autograd import *
from .base import *
from .collectors import *
from .controls import *
from .function import *
from .jit import *
from .naming import *
from .variables import *

if __name__ == '__main__':
    ProgressBar
