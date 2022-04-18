# -*- coding: utf-8 -*-

__version__ = "2.1.5"


try:
  import jaxlib
  del jaxlib
except ModuleNotFoundError:
  raise ModuleNotFoundError(
    'Please install jaxlib. See '
    'https://brainpy.readthedocs.io/en/latest/quickstart/installation.html#dependency-2-jax '
    'for installation instructions.'
  )


# fundamental modules
from . import errors, tools, check


# "base" module
from . import base
from .base.base import Base
from .base.collector import Collector, TensorCollector


# math foundation
from . import math


# toolboxes
from . import connect, initialize, optimizers, measure, losses, datasets, inputs


# numerical integrators
from . import integrators
from .integrators import ode
from .integrators import sde
from .integrators import dde
from .integrators import fde
from .integrators.ode import odeint
from .integrators.sde import sdeint
from .integrators.dde import ddeint
from .integrators.fde import fdeint
from .integrators.joint_eq import JointEq


# dynamics simulation
from . import dyn


# neural networks modeling
from . import nn


# running
from . import running


# automatic dynamics analysis
from . import analysis


# "visualization" module, will be removed soon
from .visualization import visualize


# compatible interface
from .compat import *  # compat


# convenient access
conn = connect
init = initialize
optim = optimizers
