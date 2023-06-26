# -*- coding: utf-8 -*-

__version__ = "2.4.2"

# fundamental supporting modules
from brainpy import errors, check, tools

try:
  import jaxlib

  del jaxlib
except ModuleNotFoundError:
  raise ModuleNotFoundError(tools.jaxlib_install_info) from None

#  Part 1: Math Foundation  #
# ------------------------- #

# math foundation
from brainpy import math
from .math import BrainPyObject

#  Part 2: Toolbox  #
# ----------------- #

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
)
from . import algorithms  # online or offline training algorithms

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

#  Part 3: Models  #
# ---------------- #

from brainpy import (
  channels,  # channel models
  neurons,  # neuron groups
  synapses,  # synapses
  rates,  # rate models
  experimental,

  dnn, layers,  # deep neural network module
  dyn,  # dynamics module
  # delay,  # delay module
)

from brainpy.synapses import (
  synouts,  # synaptic output
  synplast,  # synaptic plasticity
)

from brainpy._src.dynsys import (
  DynamicalSystem as DynamicalSystem,
  Container as Container,
  Sequential as Sequential,
  Network as Network,
  NeuGroup as NeuGroup,
  SynConn as SynConn,
  SynOut as SynOut,
  SynSTP as SynSTP,
  SynLTP as SynLTP,
  TwoEndConn as TwoEndConn,
  CondNeuGroup as CondNeuGroup,
  Channel as Channel
)

# shared parameters
from brainpy._src.context import share
from brainpy._src.dynsys import not_pass_shared

# running
from brainpy._src.runners import (DSRunner as DSRunner)
from brainpy._src.transform import (LoopOverTime as LoopOverTime, )

# DynamicalSystem base classes
from brainpy._src.dynsys import (
  DynamicalSystemNS as DynamicalSystemNS,
  NeuGroupNS as NeuGroupNS,
  TwoEndConnNS as TwoEndConnNS,
)
from brainpy._src.synapses_v2.base import (SynOutNS as SynOutNS,
                                           SynSTPNS as SynSTPNS,
                                           SynConnNS as SynConnNS, )

#  Part 4: Training  #
# ------------------ #

from brainpy._src.train.base import (DSTrainer as DSTrainer, )
from brainpy._src.train.back_propagation import (BPTT as BPTT,
                                                 BPFF as BPFF, )
from brainpy._src.train.online import (OnlineTrainer as OnlineTrainer,
                                       ForceTrainer as ForceTrainer, )
from brainpy._src.train.offline import (OfflineTrainer as OfflineTrainer,
                                        RidgeTrainer as RidgeTrainer, )

#  Part 6: Others    #
# ------------------ #

from brainpy import running, testing, analysis
from brainpy._src.visualization import (visualize as visualize)
from brainpy._src import base, train

#  Part 7: Deprecations  #
# ---------------------- #

from brainpy._src import modes
from brainpy._src.math.object_transform.base import (Base as Base,
                                                     ArrayCollector,
                                                     Collector as Collector, )

# deprecated
from brainpy._src import checking
from brainpy._src.synapses import compat
from brainpy._src.deprecations import deprecation_getattr2

__deprecations = {
  'optimizers': ('brainpy.optimizers', 'brainpy.optim', optim),
  'TensorCollector': ('brainpy.TensorCollector', 'brainpy.ArrayCollector', ArrayCollector),
}
__getattr__ = deprecation_getattr2('brainpy', __deprecations)

tools.__deprecations = {
  'clear_name_cache': ('brainpy.tools.clear_name_cache', 'brainpy.math.clear_name_cache', math.clear_name_cache),
  'checking': ('brainpy.tools.checking', 'brainpy.checking', checking),
}
tools.__getattr__ = deprecation_getattr2('brainpy.tools', tools.__deprecations)

integrators.__deprecations = {
  'Integrator': ('brainpy.integrators.Integrator', 'brainpy.Integrator', Integrator),
  'odeint': ('brainpy.integrators.odeint', 'brainpy.odeint', odeint),
  'sdeint': ('brainpy.integrators.sdeint', 'brainpy.sdeint', sdeint),
  'fdeint': ('brainpy.integrators.fdeint', 'brainpy.fdeint', fdeint),
  'IntegratorRunner': ('brainpy.integrators.IntegratorRunner', 'brainpy.IntegratorRunner', IntegratorRunner),
  'JointEq': ('brainpy.integrators.JointEq', 'brainpy.JointEq', JointEq),
}
integrators.__getattr__ = deprecation_getattr2('brainpy.integrators', integrators.__deprecations)

train.__deprecations = {
  'DSTrainer': ('brainpy.train.DSTrainer', 'brainpy.DSTrainer', DSTrainer),
  'BPTT': ('brainpy.train.BPTT', 'brainpy.BPTT', BPTT),
  'BPFF': ('brainpy.train.BPFF', 'brainpy.BPFF', BPFF),
  'OnlineTrainer': ('brainpy.train.OnlineTrainer', 'brainpy.OnlineTrainer', OnlineTrainer),
  'ForceTrainer': ('brainpy.train.ForceTrainer', 'brainpy.ForceTrainer', ForceTrainer),
  'OfflineTrainer': ('brainpy.train.OfflineTrainer', 'brainpy.OfflineTrainer', OfflineTrainer),
  'RidgeTrainer': ('brainpy.train.RidgeTrainer', 'brainpy.RidgeTrainer', RidgeTrainer),
}
train.__getattr__ = deprecation_getattr2('brainpy.train', train.__deprecations)

ode.__deprecations = {'odeint': ('brainpy.ode.odeint', 'brainpy.odeint', odeint)}
ode.__getattr__ = deprecation_getattr2('brainpy.ode', ode.__deprecations)

sde.__deprecations = {'sdeint': ('brainpy.sde.sdeint', 'brainpy.sdeint', sdeint)}
sde.__getattr__ = deprecation_getattr2('brainpy.sde', sde.__deprecations)

fde.__deprecations = {'fdeint': ('brainpy.fde.fdeint', 'brainpy.fdeint', fdeint)}
fde.__getattr__ = deprecation_getattr2('brainpy.fde', sde.__deprecations)

dyn.__deprecations = {
  # module
  # 'channels': ('brainpy.dyn.channels', 'brainpy.channels', channels),
  # 'neurons': ('brainpy.dyn.neurons', 'brainpy.neurons', neurons),
  'rates': ('brainpy.dyn.rates', 'brainpy.rates', rates),
  # 'synapses': ('brainpy.dyn.synapses', 'brainpy.synapses', synapses),
  'synouts': ('brainpy.dyn.synouts', 'brainpy.synapses', synouts),
  'synplast': ('brainpy.dyn.synplast', 'brainpy.synapses', synplast),

  # models
  'DynamicalSystem': ('brainpy.dyn.DynamicalSystem', 'brainpy.DynamicalSystem', DynamicalSystem),
  'Container': ('brainpy.dyn.Container', 'brainpy.Container', Container),
  'Sequential': ('brainpy.dyn.Sequential', 'brainpy.Sequential', Sequential),
  'Network': ('brainpy.dyn.Network', 'brainpy.Network', Network),
  'NeuGroup': ('brainpy.dyn.NeuGroup', 'brainpy.NeuGroup', NeuGroup),
  'SynConn': ('brainpy.dyn.SynConn', 'brainpy.SynConn', SynConn),
  # 'SynOut': ('brainpy.dyn.SynOut', 'brainpy.SynOut', SynOut),
  'SynLTP': ('brainpy.dyn.SynLTP', 'brainpy.SynLTP', SynLTP),
  'SynSTP': ('brainpy.dyn.SynSTP', 'brainpy.SynSTP', SynSTP),
  'TwoEndConn': ('brainpy.dyn.TwoEndConn', 'brainpy.TwoEndConn', TwoEndConn),
  'CondNeuGroup': ('brainpy.dyn.CondNeuGroup', 'brainpy.CondNeuGroup', CondNeuGroup),
  'Channel': ('brainpy.dyn.Channel', 'brainpy.Channel', Channel),
  'LoopOverTime': ('brainpy.dyn.LoopOverTime', 'brainpy.LoopOverTime', LoopOverTime),
  'DSRunner': ('brainpy.dyn.DSRunner', 'brainpy.DSRunner', DSRunner),

  # neurons
  'HH': ('brainpy.dyn.HH', 'brainpy.neurons.HH', neurons.HH),
  'MorrisLecar': ('brainpy.dyn.MorrisLecar', 'brainpy.neurons.MorrisLecar', neurons.MorrisLecar),
  'PinskyRinzelModel': ('brainpy.dyn.PinskyRinzelModel', 'brainpy.neurons.PinskyRinzelModel',
                        neurons.PinskyRinzelModel),
  'FractionalFHR': ('brainpy.dyn.FractionalFHR', 'brainpy.neurons.FractionalFHR', neurons.FractionalFHR),
  'FractionalIzhikevich': ('brainpy.dyn.FractionalIzhikevich', 'brainpy.neurons.FractionalIzhikevich',
                           neurons.FractionalIzhikevich),
  'LIF': ('brainpy.dyn.LIF', 'brainpy.neurons.LIF', neurons.LIF),
  'ExpIF': ('brainpy.dyn.ExpIF', 'brainpy.neurons.ExpIF', neurons.ExpIF),
  'AdExIF': ('brainpy.dyn.AdExIF', 'brainpy.neurons.AdExIF', neurons.AdExIF),
  'QuaIF': ('brainpy.dyn.QuaIF', 'brainpy.neurons.QuaIF', neurons.QuaIF),
  'AdQuaIF': ('brainpy.dyn.AdQuaIF', 'brainpy.neurons.AdQuaIF', neurons.AdQuaIF),
  'GIF': ('brainpy.dyn.GIF', 'brainpy.neurons.GIF', neurons.GIF),
  'Izhikevich': ('brainpy.dyn.Izhikevich', 'brainpy.neurons.Izhikevich', neurons.Izhikevich),
  'HindmarshRose': ('brainpy.dyn.HindmarshRose', 'brainpy.neurons.HindmarshRose', neurons.HindmarshRose),
  'FHN': ('brainpy.dyn.FHN', 'brainpy.neurons.FHN', neurons.FHN),
  'SpikeTimeGroup': ('brainpy.dyn.SpikeTimeGroup', 'brainpy.neurons.SpikeTimeGroup', neurons.SpikeTimeGroup),
  'PoissonGroup': ('brainpy.dyn.PoissonGroup', 'brainpy.neurons.PoissonGroup', neurons.PoissonGroup),
  'OUProcess': ('brainpy.dyn.OUProcess', 'brainpy.neurons.OUProcess', neurons.OUProcess),

  # synapses
  'DeltaSynapse': ('brainpy.dyn.DeltaSynapse', 'brainpy.synapses.Delta', compat.DeltaSynapse),
  'ExpCUBA': ('brainpy.dyn.ExpCUBA', 'brainpy.synapses.Exponential', compat.ExpCUBA),
  'ExpCOBA': ('brainpy.dyn.ExpCOBA', 'brainpy.synapses.Exponential', compat.ExpCOBA),
  'DualExpCUBA': ('brainpy.dyn.DualExpCUBA', 'brainpy.synapses.DualExponential', compat.DualExpCUBA),
  'DualExpCOBA': ('brainpy.dyn.DualExpCOBA', 'brainpy.synapses.DualExponential', compat.DualExpCOBA),
  'AlphaCUBA': ('brainpy.dyn.AlphaCUBA', 'brainpy.synapses.Alpha', compat.AlphaCUBA),
  'AlphaCOBA': ('brainpy.dyn.AlphaCOBA', 'brainpy.synapses.Alpha', compat.AlphaCOBA),
  # 'NMDA': ('brainpy.dyn.NMDA', 'brainpy.synapses.NMDA', compat.NMDA),
}
dyn.__getattr__ = deprecation_getattr2('brainpy.dyn', dyn.__deprecations)

del deprecation_getattr2, checking, compat
