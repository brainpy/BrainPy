# -*- coding: utf-8 -*-


from brainpy import _errors as errors
# fundamental supporting modules
from brainpy.version2 import check, tools
#  Part: Math Foundation  #
# ----------------------- #
# math foundation
from brainpy.version2 import math
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
    mixin,  # mixin classes
    algorithms,  # online or offline training algorithms
)
from .math import BrainPyObject

# convenient alias
conn = connect
init = initialize

# numerical integrators
from brainpy.version2 import integrators
from brainpy.version2.integrators import ode, sde, fde
from brainpy.version2.integrators.base import (Integrator as Integrator)
from brainpy.version2.integrators.joint_eq import (JointEq as JointEq)
from brainpy.version2.integrators.runner import (IntegratorRunner as IntegratorRunner)
from brainpy.version2.integrators.ode.generic import (odeint as odeint)
from brainpy.version2.integrators.sde.generic import (sdeint as sdeint)
from brainpy.version2.integrators.fde.generic import (fdeint as fdeint)

#  Part: Models  #
# -------------- #

# base classes
from brainpy.version2.dynsys import (
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
from brainpy.version2.delay import (
    VarDelay as VarDelay,
)

# building blocks
from brainpy.version2 import (
    dnn, layers,  # module for dnn layers
    dyn,  # module for modeling dynamics
)

NeuGroup = NeuGroupNS = dyn.NeuDyn

# common tools
from brainpy.version2.context import (share as share)
from brainpy.version2.helpers import (
    reset_level as reset_level,
    reset_state as reset_state,
    save_state as save_state,
    load_state as load_state,
    clear_input as clear_input
)

#  Part: Running  #
# --------------- #
from brainpy.version2.runners import (DSRunner as DSRunner)
from brainpy.version2.transform import (LoopOverTime as LoopOverTime, )
from brainpy.version2 import (running as running)

#  Part: Training  #
# ---------------- #
from brainpy.version2.train.base import (DSTrainer as DSTrainer, )
from brainpy.version2.train.back_propagation import (BPTT as BPTT,
                                                     BPFF as BPFF, )
from brainpy.version2.train.online import (OnlineTrainer as OnlineTrainer,
                                           ForceTrainer as ForceTrainer, )
from brainpy.version2.train.offline import (OfflineTrainer as OfflineTrainer,
                                            RidgeTrainer as RidgeTrainer, )

#  Part: Analysis  #
# ---------------- #
from brainpy.version2 import (analysis as analysis)

#  Part: Others    #
# ---------------- #
import brainpy.version2.visualization as visualize

#  Part: Deprecations  #
# -------------------- #
from brainpy.version2 import train
from brainpy.version2 import (
    channels,  # channel models
    neurons,  # neuron groups
    synapses,  # synapses
    rates,  # rate models
    experimental,
    synouts,  # synaptic output
    synplast,  # synaptic plasticity
)
from brainpy.version2 import modes
from brainpy.version2.math.object_transform.base import (
    Base as Base,
)

from brainpy.version2.math.object_transform.collectors import (
    ArrayCollector as ArrayCollector,
    Collector as Collector,
)


if __name__ == '__main__':
    connect
    initialize,  # weight initialization
    optim,  # gradient descent optimizers
    losses,  # loss functions
    measure,  # methods for data analysis
    inputs,  # methods for generating input currents
    encoding,  # encoding schema
    checkpoints,  # checkpoints
    check,  # error checking
    mixin,  # mixin classes
    algorithms,  # online or offline training algorithms
    check, tools, errors, math
    BrainPyObject,
    integrators, ode, sde, fde
    Integrator, JointEq, IntegratorRunner, odeint, sdeint, fdeint
    DynamicalSystem, DynSysGroup, Sequential, Dynamic, Projection
    receive_update_input, receive_update_output, not_receive_update_input, not_receive_update_output
    VarDelay
    dnn, layers, dyn
    NeuGroup, NeuGroupNS
    share
    reset_level, reset_state, save_state, load_state, clear_input
    DSRunner, LoopOverTime, running
    DSTrainer, BPTT, BPFF, OnlineTrainer, ForceTrainer,
    OfflineTrainer, RidgeTrainer
    analysis
    visualize
    train
    channels, neurons, synapses, rates, experimental, synouts, synplast
    modes
    Base
    ArrayCollector, Collector
