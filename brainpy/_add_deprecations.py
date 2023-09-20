
from ._src import checking, train, integrators
from . import tools, math, integrators, dyn, dnn, neurons, synapses, layers, connect
from .integrators import ode, fde, sde
from brainpy._src.integrators.base import Integrator
from brainpy._src.integrators.runner import IntegratorRunner
from brainpy._src.integrators.joint_eq import JointEq
from brainpy._src.integrators.ode.generic import odeint
from brainpy._src.integrators.sde.generic import sdeint
from brainpy._src.integrators.fde.generic import fdeint
from brainpy._src.dynsys import (DynamicalSystem, DynSysGroup, Sequential, Network)
from brainpy._src.dyn.base import NeuDyn, IonChaDyn
from brainpy._src.runners import DSRunner
from brainpy._src.deprecations import deprecation_getattr2

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
  'DSTrainer': ('brainpy.train.DSTrainer', 'brainpy.DSTrainer', train.base.DSTrainer),
  'BPTT': ('brainpy.train.BPTT', 'brainpy.BPTT', train.back_propagation.BPTT),
  'BPFF': ('brainpy.train.BPFF', 'brainpy.BPFF', train.back_propagation.BPFF),
  'OnlineTrainer': ('brainpy.train.OnlineTrainer', 'brainpy.OnlineTrainer', train.online.OnlineTrainer),
  'ForceTrainer': ('brainpy.train.ForceTrainer', 'brainpy.ForceTrainer', train.online.ForceTrainer),
  'OfflineTrainer': ('brainpy.train.OfflineTrainer', 'brainpy.OfflineTrainer', train.offline.OfflineTrainer),
  'RidgeTrainer': ('brainpy.train.RidgeTrainer', 'brainpy.RidgeTrainer', train.offline.RidgeTrainer),
}
train.__getattr__ = deprecation_getattr2('brainpy.train', train.__deprecations)


neurons.__deprecations = {
  'OUProcess': ('brainpy.neurons.OUProcess', 'brainpy.dyn.OUProcess', dyn.OUProcess),
  'Leaky': ('brainpy.neurons.Leaky', 'brainpy.dyn.Leaky', dyn.Leaky),
  'Integrator': ('brainpy.neurons.Integrator', 'brainpy.dyn.Integrator', dyn.Integrator),
  'InputGroup': ('brainpy.neurons.InputGroup', 'brainpy.dyn.InputGroup', dyn.InputGroup),
  'OutputGroup': ('brainpy.neurons.OutputGroup', 'brainpy.dyn.OutputGroup', dyn.OutputGroup),
  'SpikeTimeGroup': ('brainpy.neurons.SpikeTimeGroup', 'brainpy.dyn.SpikeTimeGroup', dyn.SpikeTimeGroup),
  'PoissonGroup': ('brainpy.neurons.PoissonGroup', 'brainpy.dyn.PoissonGroup', dyn.PoissonGroup),
}
neurons.__getattr__ = deprecation_getattr2('brainpy.neurons', neurons.__deprecations)


synapses.__deprecations = {
  'PoissonInput': ('brainpy.synapses.PoissonInput', 'brainpy.dyn.PoissonInput', dyn.PoissonInput),
  'DiffusiveCoupling': ('brainpy.synapses.DiffusiveCoupling', 'brainpy.dyn.DiffusiveCoupling', dyn.DiffusiveCoupling),
  'AdditiveCoupling': ('brainpy.synapses.AdditiveCoupling', 'brainpy.dyn.AdditiveCoupling', dyn.AdditiveCoupling),
}
synapses.__getattr__ = deprecation_getattr2('brainpy.synapses', synapses.__deprecations)


ode.__deprecations = {
  'odeint': ('brainpy.ode.odeint', 'brainpy.odeint', odeint)
}
ode.__getattr__ = deprecation_getattr2('brainpy.ode', ode.__deprecations)

sde.__deprecations = {
  'sdeint': ('brainpy.sde.sdeint', 'brainpy.sdeint', sdeint)
}
sde.__getattr__ = deprecation_getattr2('brainpy.sde', sde.__deprecations)

fde.__deprecations = {
  'fdeint': ('brainpy.fde.fdeint', 'brainpy.fdeint', fdeint)
}
fde.__getattr__ = deprecation_getattr2('brainpy.fde', sde.__deprecations)

dyn.__deprecations = {
  # models
  'DynamicalSystem': ('brainpy.dyn.DynamicalSystem', 'brainpy.DynamicalSystem', DynamicalSystem),
  'Container': ('brainpy.dyn.Container', 'brainpy.DynSysGroup', DynSysGroup),
  'Sequential': ('brainpy.dyn.Sequential', 'brainpy.Sequential', Sequential),
  'Network': ('brainpy.dyn.Network', 'brainpy.Network', Network),
  'Channel': ('brainpy.dyn.Channel', 'brainpy.IonChaDyn', IonChaDyn),
  'DSRunner': ('brainpy.dyn.DSRunner', 'brainpy.DSRunner', DSRunner),

  # neurons
  'NeuGroup': ('brainpy.dyn.NeuGroup', 'brainpy.dyn.NeuDyn', NeuDyn),

  # synapses
  'TwoEndConn': ('brainpy.dyn.TwoEndConn', 'brainpy.synapses.TwoEndConn', synapses.TwoEndConn),
  'SynSTP': ('brainpy.dyn.SynSTP', 'brainpy.synapses.SynSTP', synapses.SynSTP),
  'DeltaSynapse': ('brainpy.dyn.DeltaSynapse', 'brainpy.synapses.Delta', synapses.DeltaSynapse),
  'ExpCUBA': ('brainpy.dyn.ExpCUBA', 'brainpy.synapses.Exponential', synapses.ExpCUBA),
  'ExpCOBA': ('brainpy.dyn.ExpCOBA', 'brainpy.synapses.Exponential', synapses.ExpCOBA),
  'DualExpCUBA': ('brainpy.dyn.DualExpCUBA', 'brainpy.synapses.DualExponential', synapses.DualExpCUBA),
  'DualExpCOBA': ('brainpy.dyn.DualExpCOBA', 'brainpy.synapses.DualExponential', synapses.DualExpCOBA),
  'AlphaCUBA': ('brainpy.dyn.AlphaCUBA', 'brainpy.synapses.Alpha', synapses.AlphaCUBA),
  'AlphaCOBA': ('brainpy.dyn.AlphaCOBA', 'brainpy.synapses.Alpha', synapses.AlphaCOBA),
}
dyn.__getattr__ = deprecation_getattr2('brainpy.dyn', dyn.__deprecations)

dnn.__deprecations = {
  'NVAR': ('brainpy.dnn.NVAR', 'brainpy.dyn.NVAR', dyn.NVAR),
  'Reservoir': ('brainpy.dnn.Reservoir', 'brainpy.dyn.Reservoir', dyn.Reservoir),
  'RNNCell': ('brainpy.dnn.RNNCell', 'brainpy.dyn.RNNCell', dyn.RNNCell),
  'GRUCell': ('brainpy.dnn.GRUCell', 'brainpy.dyn.GRUCell', dyn.GRUCell),
  'LSTMCell': ('brainpy.dnn.LSTMCell', 'brainpy.dyn.LSTMCell', dyn.LSTMCell),
  'Conv1dLSTMCell': ('brainpy.dnn.Conv1dLSTMCell', 'brainpy.dyn.Conv1dLSTMCell', dyn.Conv1dLSTMCell),
  'Conv2dLSTMCell': ('brainpy.dnn.Conv2dLSTMCell', 'brainpy.dyn.Conv2dLSTMCell', dyn.Conv2dLSTMCell),
  'Conv3dLSTMCell': ('brainpy.dnn.Conv3dLSTMCell', 'brainpy.dyn.Conv3dLSTMCell', dyn.Conv3dLSTMCell),
}
dnn.__getattr__ = deprecation_getattr2('brainpy.dnn', dnn.__deprecations)

layers.__deprecations = {
  'NVAR': ('brainpy.layers.NVAR', 'brainpy.dyn.NVAR', dyn.NVAR),
  'Reservoir': ('brainpy.layers.Reservoir', 'brainpy.dyn.Reservoir', dyn.Reservoir),
  'RNNCell': ('brainpy.layers.RNNCell', 'brainpy.dyn.RNNCell', dyn.RNNCell),
  'GRUCell': ('brainpy.layers.GRUCell', 'brainpy.dyn.GRUCell', dyn.GRUCell),
  'LSTMCell': ('brainpy.layers.LSTMCell', 'brainpy.dyn.LSTMCell', dyn.LSTMCell),
  'Conv1dLSTMCell': ('brainpy.layers.Conv1dLSTMCell', 'brainpy.dyn.Conv1dLSTMCell', dyn.Conv1dLSTMCell),
  'Conv2dLSTMCell': ('brainpy.layers.Conv2dLSTMCell', 'brainpy.dyn.Conv2dLSTMCell', dyn.Conv2dLSTMCell),
  'Conv3dLSTMCell': ('brainpy.layers.Conv3dLSTMCell', 'brainpy.dyn.Conv3dLSTMCell', dyn.Conv3dLSTMCell),
}
layers.__getattr__ = deprecation_getattr2('brainpy.layers', layers.__deprecations)


connect.__deprecations = {
    'one2one': ('brainpy.connect.one2one', 'brainpy.connect.One2One', connect.One2One),
    'all2all': ('brainpy.connect.all2all', 'brainpy.connect.All2All', connect.All2All),
    'grid_four': ('brainpy.connect.grid_four', 'brainpy.connect.GridFour', connect.GridFour),
    'grid_eight': ('brainpy.connect.grid_eight', 'brainpy.connect.GridEight', connect.GridEight),
}
connect.__getattr__ = deprecation_getattr2('brainpy.connect', connect.__deprecations)

