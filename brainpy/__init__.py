# -*- coding: utf-8 -*-

__version__ = "2.4.3"

# fundamental supporting modules
from brainpy import errors, check, tools

try:
  import jaxlib
  del jaxlib
except ModuleNotFoundError:
  raise ModuleNotFoundError(tools.jaxlib_install_info) from None

#  Part: Math Foundation  #
# ----------------------- #

# math foundation
from brainpy import math
from .math import BrainPyObject

#  Part: Toolbox  #
# --------------- #
# modules of toolbox
from brainpy import (
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

# convenient alias
conn = connect
init = initialize

# numerical integrators
from brainpy import integrators
from brainpy.integrators import ode, sde, fde
from brainpy._src.integrators.base import (Integrator as Integrator)
from brainpy._src.integrators.joint_eq import (JointEq as JointEq)
from brainpy._src.integrators.runner import (IntegratorRunner as IntegratorRunner)
from brainpy._src.integrators.ode.generic import (odeint as odeint)
from brainpy._src.integrators.sde.generic import (sdeint as sdeint)
from brainpy._src.integrators.fde.generic import (fdeint as fdeint)


#  Part: Models  #
# -------------- #

# base classes
from brainpy._src.dynsys import (
  DynamicalSystem as DynamicalSystem,
  DynSysGroup as DynSysGroup,  # collectors
  Sequential as Sequential,
  Dynamic as Dynamic,  # category
  Projection as Projection,
)
DynamicalSystemNS = DynamicalSystem
Network = DynSysGroup
# delays
from brainpy._src.delay import (
  VarDelay as VarDelay,
)

# building blocks
from brainpy import (
  dnn, layers,  # module for dnn layers
  dyn,  # module for modeling dynamics
)
NeuGroup = NeuGroupNS = dyn.NeuDyn

# shared parameters
from brainpy._src.context import (share as share)
from brainpy._src.dynsys import not_pass_shared


#  Part: Running  #
# --------------- #
from brainpy._src.runners import (DSRunner as DSRunner)
from brainpy._src.transform import (LoopOverTime as LoopOverTime, )
from brainpy import (running as running)


#  Part: Training  #
# ---------------- #
from brainpy._src.train.base import (DSTrainer as DSTrainer, )
from brainpy._src.train.back_propagation import (BPTT as BPTT,
                                                 BPFF as BPFF,)
from brainpy._src.train.online import (OnlineTrainer as OnlineTrainer,
                                       ForceTrainer as ForceTrainer, )
from brainpy._src.train.offline import (OfflineTrainer as OfflineTrainer,
                                        RidgeTrainer as RidgeTrainer, )


#  Part: Analysis  #
# ---------------- #
from brainpy import (analysis as analysis)


#  Part: Others    #
# ---------------- #
from brainpy._src.visualization import (visualize as visualize)


#  Part: Deprecations  #
# -------------------- #
from brainpy._src import base, train
from brainpy import (
  channels,  # channel models
  neurons,  # neuron groups
  synapses,  # synapses
  rates,  # rate models
  experimental,
  synouts,  # synaptic output
  synplast,  # synaptic plasticity
)
from brainpy._src import modes
from brainpy._src.math.object_transform.base import (Base as Base,
                                                     ArrayCollector as ArrayCollector,
                                                     Collector as Collector, )

# deprecated
from brainpy._add_deprecations import deprecation_getattr2

__deprecations = {
  'Module': ('brainpy.Module', 'brainpy.DynamicalSystem', DynamicalSystem),
  'Channel': ('brainpy.Channel', 'brainpy.dyn.IonChannel', dyn.IonChannel),
  'SynConn': ('brainpy.SynConn', 'brainpy.dyn.SynConn', dyn.SynConn),
  'Container': ('brainpy.Container', 'brainpy.DynSysGroup', DynSysGroup),

  'optimizers': ('brainpy.optimizers', 'brainpy.optim', optim),
  'TensorCollector': ('brainpy.TensorCollector', 'brainpy.ArrayCollector', ArrayCollector),
  'SynSTP': ('brainpy.SynSTP', 'brainpy.synapses.SynSTP', synapses.SynSTP),
  'SynOut': ('brainpy.SynOut', 'brainpy.synapses.SynOut', synapses.SynOut),
  'TwoEndConn': ('brainpy.TwoEndConn', 'brainpy.synapses.TwoEndConn', synapses.TwoEndConn),
  'CondNeuGroup': ('brainpy.CondNeuGroup', 'brainpy.syn.CondNeuGroup', dyn.CondNeuGroup),
}
__getattr__ = deprecation_getattr2('brainpy', __deprecations)

del deprecation_getattr2

