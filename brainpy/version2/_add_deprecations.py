
from . import check, train, integrators, tools, math, integrators, dyn, dnn, neurons, synapses, layers, connect
from .integrators import ode, fde, sde
from brainpy.version2.integrators.base import Integrator
from brainpy.version2.integrators.runner import IntegratorRunner
from brainpy.version2.integrators.joint_eq import JointEq
from brainpy.version2.integrators.ode.generic import odeint
from brainpy.version2.integrators.sde.generic import sdeint
from brainpy.version2.integrators.fde.generic import fdeint
from brainpy.version2.dynsys import (DynamicalSystem, DynSysGroup, Sequential, Network)
from brainpy.version2.dyn.base import NeuDyn, IonChaDyn
from brainpy.version2.runners import DSRunner
from brainpy.version2.deprecations import deprecation_getattr2

tools.__deprecations = {
  'clear_name_cache': ('brainpy.version2.tools.clear_name_cache', 'brainpy.version2.math.clear_name_cache', math.clear_name_cache),
  'checking': ('brainpy.version2.tools.checking', 'brainpy.version2.checking', check),
}
tools.__getattr__ = deprecation_getattr2('brainpy.version2.tools', tools.__deprecations)

integrators.__deprecations = {
  'Integrator': ('brainpy.version2.integrators.Integrator', 'brainpy.version2.Integrator', Integrator),
  'odeint': ('brainpy.version2.integrators.odeint', 'brainpy.version2.odeint', odeint),
  'sdeint': ('brainpy.version2.integrators.sdeint', 'brainpy.version2.sdeint', sdeint),
  'fdeint': ('brainpy.version2.integrators.fdeint', 'brainpy.version2.fdeint', fdeint),
  'IntegratorRunner': ('brainpy.version2.integrators.IntegratorRunner', 'brainpy.version2.IntegratorRunner', IntegratorRunner),
  'JointEq': ('brainpy.version2.integrators.JointEq', 'brainpy.version2.JointEq', JointEq),
}
integrators.__getattr__ = deprecation_getattr2('brainpy.version2.integrators', integrators.__deprecations)

train.__deprecations = {
  'DSTrainer': ('brainpy.version2.train.DSTrainer', 'brainpy.version2.DSTrainer', train.base.DSTrainer),
  'BPTT': ('brainpy.version2.train.BPTT', 'brainpy.version2.BPTT', train.back_propagation.BPTT),
  'BPFF': ('brainpy.version2.train.BPFF', 'brainpy.version2.BPFF', train.back_propagation.BPFF),
  'OnlineTrainer': ('brainpy.version2.train.OnlineTrainer', 'brainpy.version2.OnlineTrainer', train.online.OnlineTrainer),
  'ForceTrainer': ('brainpy.version2.train.ForceTrainer', 'brainpy.version2.ForceTrainer', train.online.ForceTrainer),
  'OfflineTrainer': ('brainpy.version2.train.OfflineTrainer', 'brainpy.version2.OfflineTrainer', train.offline.OfflineTrainer),
  'RidgeTrainer': ('brainpy.version2.train.RidgeTrainer', 'brainpy.version2.RidgeTrainer', train.offline.RidgeTrainer),
}
train.__getattr__ = deprecation_getattr2('brainpy.version2.train', train.__deprecations)


neurons.__deprecations = {
  'OUProcess': ('brainpy.version2.neurons.OUProcess', 'brainpy.version2.dyn.OUProcess', dyn.OUProcess),
  'Leaky': ('brainpy.version2.neurons.Leaky', 'brainpy.version2.dyn.Leaky', dyn.Leaky),
  'Integrator': ('brainpy.version2.neurons.Integrator', 'brainpy.version2.dyn.Integrator', dyn.Integrator),
  'InputGroup': ('brainpy.version2.neurons.InputGroup', 'brainpy.version2.dyn.InputGroup', dyn.InputGroup),
  'OutputGroup': ('brainpy.version2.neurons.OutputGroup', 'brainpy.version2.dyn.OutputGroup', dyn.OutputGroup),
  'SpikeTimeGroup': ('brainpy.version2.neurons.SpikeTimeGroup', 'brainpy.version2.dyn.SpikeTimeGroup', dyn.SpikeTimeGroup),
  'PoissonGroup': ('brainpy.version2.neurons.PoissonGroup', 'brainpy.version2.dyn.PoissonGroup', dyn.PoissonGroup),
}
neurons.__getattr__ = deprecation_getattr2('brainpy.version2.neurons', neurons.__deprecations)


synapses.__deprecations = {
  'PoissonInput': ('brainpy.version2.synapses.PoissonInput', 'brainpy.version2.dyn.PoissonInput', dyn.PoissonInput),
  'DiffusiveCoupling': ('brainpy.version2.synapses.DiffusiveCoupling', 'brainpy.version2.dyn.DiffusiveCoupling', dyn.DiffusiveCoupling),
  'AdditiveCoupling': ('brainpy.version2.synapses.AdditiveCoupling', 'brainpy.version2.dyn.AdditiveCoupling', dyn.AdditiveCoupling),
}
synapses.__getattr__ = deprecation_getattr2('brainpy.version2.synapses', synapses.__deprecations)


ode.__deprecations = {
  'odeint': ('brainpy.version2.ode.odeint', 'brainpy.version2.odeint', odeint)
}
ode.__getattr__ = deprecation_getattr2('brainpy.version2.ode', ode.__deprecations)

sde.__deprecations = {
  'sdeint': ('brainpy.version2.sde.sdeint', 'brainpy.version2.sdeint', sdeint)
}
sde.__getattr__ = deprecation_getattr2('brainpy.version2.sde', sde.__deprecations)

fde.__deprecations = {
  'fdeint': ('brainpy.version2.fde.fdeint', 'brainpy.version2.fdeint', fdeint)
}
fde.__getattr__ = deprecation_getattr2('brainpy.version2.fde', sde.__deprecations)

dyn.__deprecations = {
  # models
  'DynamicalSystem': ('brainpy.version2.dyn.DynamicalSystem', 'brainpy.version2.DynamicalSystem', DynamicalSystem),
  'Container': ('brainpy.version2.dyn.Container', 'brainpy.version2.DynSysGroup', DynSysGroup),
  'Sequential': ('brainpy.version2.dyn.Sequential', 'brainpy.version2.Sequential', Sequential),
  'Network': ('brainpy.version2.dyn.Network', 'brainpy.version2.Network', Network),
  'Channel': ('brainpy.version2.dyn.Channel', 'brainpy.version2.IonChaDyn', IonChaDyn),
  'DSRunner': ('brainpy.version2.dyn.DSRunner', 'brainpy.version2.DSRunner', DSRunner),

  # neurons
  'NeuGroup': ('brainpy.version2.dyn.NeuGroup', 'brainpy.version2.dyn.NeuDyn', NeuDyn),

  # projections
  'ProjAlignPostMg1': ('brainpy.version2.dyn.ProjAlignPostMg1', 'brainpy.version2.dyn.HalfProjAlignPostMg', dyn.HalfProjAlignPostMg),
  'ProjAlignPostMg2': ('brainpy.version2.dyn.ProjAlignPostMg2', 'brainpy.version2.dyn.FullProjAlignPostMg', dyn.FullProjAlignPostMg),
  'ProjAlignPost1': ('brainpy.version2.dyn.ProjAlignPost1', 'brainpy.version2.dyn.HalfProjAlignPost', dyn.HalfProjAlignPost),
  'ProjAlignPost2': ('brainpy.version2.dyn.ProjAlignPost2', 'brainpy.version2.dyn.FullProjAlignPost', dyn.FullProjAlignPost),
  'ProjAlignPreMg1': ('brainpy.version2.dyn.ProjAlignPreMg1', 'brainpy.version2.dyn.FullProjAlignPreSDMg', dyn.FullProjAlignPreSDMg),
  'ProjAlignPreMg2': ('brainpy.version2.dyn.ProjAlignPreMg2', 'brainpy.version2.dyn.FullProjAlignPreDSMg', dyn.FullProjAlignPreDSMg),
  'ProjAlignPre1': ('brainpy.version2.dyn.ProjAlignPre1', 'brainpy.version2.dyn.FullProjAlignPreSD', dyn.FullProjAlignPreSD),
  'ProjAlignPre2': ('brainpy.version2.dyn.ProjAlignPre2', 'brainpy.version2.dyn.FullProjAlignPreDS', dyn.FullProjAlignPreDS),

  # synapses
  'TwoEndConn': ('brainpy.version2.dyn.TwoEndConn', 'brainpy.version2.synapses.TwoEndConn', synapses.TwoEndConn),
  'SynSTP': ('brainpy.version2.dyn.SynSTP', 'brainpy.version2.synapses.SynSTP', synapses.SynSTP),
  'DeltaSynapse': ('brainpy.version2.dyn.DeltaSynapse', 'brainpy.version2.synapses.Delta', synapses.DeltaSynapse),
  'ExpCUBA': ('brainpy.version2.dyn.ExpCUBA', 'brainpy.version2.synapses.Exponential', synapses.ExpCUBA),
  'ExpCOBA': ('brainpy.version2.dyn.ExpCOBA', 'brainpy.version2.synapses.Exponential', synapses.ExpCOBA),
  'DualExpCUBA': ('brainpy.version2.dyn.DualExpCUBA', 'brainpy.version2.synapses.DualExponential', synapses.DualExpCUBA),
  'DualExpCOBA': ('brainpy.version2.dyn.DualExpCOBA', 'brainpy.version2.synapses.DualExponential', synapses.DualExpCOBA),
  'AlphaCUBA': ('brainpy.version2.dyn.AlphaCUBA', 'brainpy.version2.synapses.Alpha', synapses.AlphaCUBA),
  'AlphaCOBA': ('brainpy.version2.dyn.AlphaCOBA', 'brainpy.version2.synapses.Alpha', synapses.AlphaCOBA),
}
dyn.__getattr__ = deprecation_getattr2('brainpy.version2.dyn', dyn.__deprecations)

dnn.__deprecations = {
  'NVAR': ('brainpy.version2.dnn.NVAR', 'brainpy.version2.dyn.NVAR', dyn.NVAR),
  'Reservoir': ('brainpy.version2.dnn.Reservoir', 'brainpy.version2.dyn.Reservoir', dyn.Reservoir),
  'RNNCell': ('brainpy.version2.dnn.RNNCell', 'brainpy.version2.dyn.RNNCell', dyn.RNNCell),
  'GRUCell': ('brainpy.version2.dnn.GRUCell', 'brainpy.version2.dyn.GRUCell', dyn.GRUCell),
  'LSTMCell': ('brainpy.version2.dnn.LSTMCell', 'brainpy.version2.dyn.LSTMCell', dyn.LSTMCell),
  'Conv1dLSTMCell': ('brainpy.version2.dnn.Conv1dLSTMCell', 'brainpy.version2.dyn.Conv1dLSTMCell', dyn.Conv1dLSTMCell),
  'Conv2dLSTMCell': ('brainpy.version2.dnn.Conv2dLSTMCell', 'brainpy.version2.dyn.Conv2dLSTMCell', dyn.Conv2dLSTMCell),
  'Conv3dLSTMCell': ('brainpy.version2.dnn.Conv3dLSTMCell', 'brainpy.version2.dyn.Conv3dLSTMCell', dyn.Conv3dLSTMCell),
}
dnn.__getattr__ = deprecation_getattr2('brainpy.version2.dnn', dnn.__deprecations)

layers.__deprecations = {
  'NVAR': ('brainpy.version2.layers.NVAR', 'brainpy.version2.dyn.NVAR', dyn.NVAR),
  'Reservoir': ('brainpy.version2.layers.Reservoir', 'brainpy.version2.dyn.Reservoir', dyn.Reservoir),
  'RNNCell': ('brainpy.version2.layers.RNNCell', 'brainpy.version2.dyn.RNNCell', dyn.RNNCell),
  'GRUCell': ('brainpy.version2.layers.GRUCell', 'brainpy.version2.dyn.GRUCell', dyn.GRUCell),
  'LSTMCell': ('brainpy.version2.layers.LSTMCell', 'brainpy.version2.dyn.LSTMCell', dyn.LSTMCell),
  'Conv1dLSTMCell': ('brainpy.version2.layers.Conv1dLSTMCell', 'brainpy.version2.dyn.Conv1dLSTMCell', dyn.Conv1dLSTMCell),
  'Conv2dLSTMCell': ('brainpy.version2.layers.Conv2dLSTMCell', 'brainpy.version2.dyn.Conv2dLSTMCell', dyn.Conv2dLSTMCell),
  'Conv3dLSTMCell': ('brainpy.version2.layers.Conv3dLSTMCell', 'brainpy.version2.dyn.Conv3dLSTMCell', dyn.Conv3dLSTMCell),
}
layers.__getattr__ = deprecation_getattr2('brainpy.version2.layers', layers.__deprecations)


connect.__deprecations = {
    'one2one': ('brainpy.version2.connect.one2one', 'brainpy.version2.connect.One2One', connect.One2One),
    'all2all': ('brainpy.version2.connect.all2all', 'brainpy.version2.connect.All2All', connect.All2All),
    'grid_four': ('brainpy.version2.connect.grid_four', 'brainpy.version2.connect.GridFour', connect.GridFour),
    'grid_eight': ('brainpy.version2.connect.grid_eight', 'brainpy.version2.connect.GridEight', connect.GridEight),
}
connect.__getattr__ = deprecation_getattr2('brainpy.version2.connect', connect.__deprecations)

