# -*- coding: utf-8 -*-

__version__ = "2.3.0"


# fundamental modules
from . import errors, tools, check, modes

# "base" module
from . import base
from .base.base import (BrainPyObject, Base)
from .base.collector import (
  Collector,
  ArrayCollector,
  TensorCollector
)

# math foundation
from . import math

# toolboxes
from . import (
  connect,  # synaptic connection
  initialize,  # weight initialization
  optimizers,  # gradient descent optimizers
  losses,  # loss functions
  measure,  # methods for data analysis
  datasets,  # methods for generating data
  inputs,  # methods for generating input currents
  algorithms,  # online or offline training algorithms
)

# numerical integrators
from . import integrators
from .integrators import ode
from .integrators import sde
from .integrators import fde
from .integrators.ode import odeint
from .integrators.sde import sdeint
from .integrators.fde import fdeint
from .integrators.joint_eq import JointEq

# dynamics simulation
from . import dyn
from .dyn import (
  channels,  # channel models
  layers,  # ANN layers
  networks,  # network models
  neurons,  # neuron groups
  rates,  # rate models
  synapses,  # synaptic dynamics
  synouts,  # synaptic output
  synplast,  # synaptic plasticity
)
from .dyn.base import (
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
)
from .dyn.runners import *

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

# "visualization" module, will be removed soon
from .visualization import visualize

# convenient access
conn = connect
init = initialize
optim = optimizers
