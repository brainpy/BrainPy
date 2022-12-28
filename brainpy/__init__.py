# -*- coding: utf-8 -*-

__version__ = "2.3.1"


# fundamental modules
from . import errors, check, tools

# math foundation
from . import math

# toolboxes
from . import (
  connect,  # synaptic connection
  initialize,  # weight initialization
  optimizers,  # gradient descent optimizers
  losses,  # loss functions
  measure,  # methods for data analysis
  inputs,  # methods for generating input currents
  algorithms,  # online or offline training algorithms
  encoding,  # encoding schema
  checkpoints,  # checkpoints
  check,  # error checking
)

# numerical integrators
from . import integrators
from .integrators import (
  # sub-modules
  ode,
  sde,
  fde,

  # functions
  odeint,
  sdeint,
  fdeint,

  # classes
  JointEq,
  IntegratorRunner,
)


# dynamics simulation
from . import dyn
from .dyn import (
  # sub-modules
  channels,  # channel models
  layers,  # ANN layers
  networks,  # network models
  neurons,  # neuron groups
  rates,  # rate models
  synapses,  # synaptic dynamics
  synouts,  # synaptic output
  synplast,  # synaptic plasticity

  # brainpy_object classes
  DynamicalSystem,
  Container,
  Sequential,
  Network,
  NeuGroup,
  SynConn,
  SynOut,
  SynSTP,
  SynLTP,
  TwoEndConn,
  CondNeuGroup,
  Channel,

  # runner
  DSRunner,

  # transformations
  NoSharedArg,
  LoopOverTime,
)

# dynamics training
from . import train
from .train import (
  DSTrainer,
  OnlineTrainer, ForceTrainer,
  OfflineTrainer, RidgeTrainer,
  BPFF,
  BPTT,
)

# automatic dynamics analysis
from . import analysis
from .analysis import (
  DSAnalyzer,
  PhasePlane1D, PhasePlane2D,
  Bifurcation1D, Bifurcation2D,
  FastSlow1D, FastSlow2D,
  SlowPointFinder,
)

# running
from . import running
from .running import (Runner)

# "visualization" module, will be removed soon
from .visualization import visualize

# convenient access
conn = connect
init = initialize
optim = optimizers

from . import experimental


# deprecated
from . import base
# use ``brainpy.math.*`` instead
from brainpy.math.object_transform.base_object import (Base, BrainPyObject,)
# use ``brainpy.math.*`` instead
from brainpy.math.object_transform.collector import (Collector, ArrayCollector, TensorCollector,)
# use ``brainpy.math.*`` instead
from . import modes
