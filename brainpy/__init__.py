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

__version__ = "2.7.5"
__version_info__ = tuple(map(int, __version__.split(".")))

from brainpy import _errors as errors
# fundamental supporting modules
from brainpy import check, tools
#  Part: Math Foundation  #
# ----------------------- #
# math foundation
from brainpy import math
from brainpy import mixin
#  Part: Toolbox  #
# --------------- #
# modules of toolbox
from . import (
    connect,  # synaptic connection
    initialize,  # weight initialization
    optim,  # gradient descent optimizers
    losses,  # loss functions
    measure,  # methods for data analysis
    inputs,  # methods for generating input currents
    encoding,  # encoding schema
    checkpoints,  # checkpoints
    check,  # error checking
    algorithms,  # online or offline training algorithms
)
from .math import BrainPyObject

# convenient alias
conn = connect
init = initialize

# numerical integrators
from brainpy import integrators
from brainpy.integrators import ode, sde, fde
from brainpy.integrators.base import (Integrator as Integrator)
from brainpy.integrators.joint_eq import (JointEq as JointEq)
from brainpy.integrators.runner import (IntegratorRunner as IntegratorRunner)
from brainpy.integrators.ode.generic import (odeint as odeint)
from brainpy.integrators.sde.generic import (sdeint as sdeint)
from brainpy.integrators.fde.generic import (fdeint as fdeint)

#  Part: Models  #
# -------------- #

# base classes
from brainpy.dynsys import (
    DynamicalSystem as DynamicalSystem,
    DynSysGroup as DynSysGroup,  # collectors
    Sequential as Sequential,
    Dynamic as Dynamic,  # category
    Projection as Projection,
    receive_update_input,  # decorators
    receive_update_output,
    not_receive_update_input,
    not_receive_update_output,
)

DynamicalSystemNS = DynamicalSystem
Network = DynSysGroup
# delays
from brainpy.delay import (
    VarDelay as VarDelay,
)

# building blocks
from brainpy import (
    dnn, layers,  # module for dnn layers
    dyn,  # module for modeling dynamics
)

NeuGroup = NeuGroupNS = dyn.NeuDyn
dyn.DynamicalSystem = DynamicalSystem

# common tools
from brainpy.context import (share as share)
from brainpy.helpers import (
    reset_level as reset_level,
    reset_state as reset_state,
    save_state as save_state,
    load_state as load_state,
    clear_input as clear_input
)

#  Part: Running  #
# --------------- #
from brainpy.runners import (DSRunner as DSRunner)
from brainpy.transform import (LoopOverTime as LoopOverTime, )
from brainpy import (running as running)

#  Part: Training  #
# ---------------- #
from brainpy.train.base import (DSTrainer as DSTrainer, )
from brainpy.train.back_propagation import (BPTT as BPTT,
                                            BPFF as BPFF, )
from brainpy.train.online import (OnlineTrainer as OnlineTrainer,
                                  ForceTrainer as ForceTrainer, )
from brainpy.train.offline import (OfflineTrainer as OfflineTrainer,
                                   RidgeTrainer as RidgeTrainer, )

#  Part: Analysis  #
# ---------------- #
from brainpy import (analysis as analysis)

#  Part: Others    #
# ---------------- #
import brainpy.visualization as visualize

#  Part: Deprecations  #
# -------------------- #
from brainpy import train
from brainpy import (
    channels,  # channel models
    neurons,  # neuron groups
    synapses,  # synapses
    rates,  # rate models
    synouts,  # synaptic output
    synplast,  # synaptic plasticity
)
from brainpy.math.object_transform.base import (
    Base as Base,
)

from brainpy.math.object_transform.collectors import (
    ArrayCollector as ArrayCollector,
    Collector as Collector,
)

from brainpy.deprecations import deprecation_getattr

optimizers = optim


# New package
from brainpy import state

