# -*- coding: utf-8 -*-

__version__ = "2.2.3.6"

try:
  import jaxlib

  del jaxlib
except ModuleNotFoundError:
  raise ModuleNotFoundError(
    '''

BrainPy needs jaxlib, please install jaxlib. 

1. If you are using Windows system, install jaxlib through

   >>> pip install jaxlib -f https://whls.blob.core.windows.net/unstable/index.html

2. If you are using macOS platform, install jaxlib through

   >>> pip install jaxlib -f https://storage.googleapis.com/jax-releases/jax_releases.html

3. If you are using Linux platform, install jaxlib through

   >>> pip install jaxlib -f https://storage.googleapis.com/jax-releases/jax_releases.html

4. If you are using Linux + CUDA platform, install jaxlib through

   >>> pip install jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Note that the versions of "jax" and "jaxlib" should be consistent, like "jax=0.3.14" and "jaxlib=0.3.14".  

For more detail installation instructions, please see https://brainpy.readthedocs.io/en/latest/quickstart/installation.html#dependency-2-jax 
    
    ''') from None

# fundamental modules
from . import errors, tools, check, modes

# "base" module
from . import base
from .base.base import Base
from .base.collector import Collector, TensorCollector

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
  OnlineBPTT,
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
