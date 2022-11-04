# -*- coding: utf-8 -*-

import importlib
import inspect
import os

from brainpy.math import (activations, autograd, controls, function,
                          jit, parallels, setting, delayvars, operators)

block_list = ['test', 'register_pytree_node', 'call', 'namedtuple', 'jit', 'wraps', 'index', 'function']
for module in [jit, autograd, function, controls, activations, parallels, setting, delayvars, operators]:
  for k in dir(module):
    if (not k.startswith('_')) and (not inspect.ismodule(getattr(module, k))):
      block_list.append(k)


def get_class_funcs(module):
  classes, functions, others = [], [], []
  # Solution from: https://stackoverflow.com/questions/43059267/how-to-do-from-module-import-using-importlib
  if "__all__" in module.__dict__:
    names = module.__dict__["__all__"]
  else:
    names = [x for x in module.__dict__ if not x.startswith("_")]
  for k in names:
    data = getattr(module, k)
    if not inspect.ismodule(data) and not k.startswith("_"):
      if inspect.isfunction(data):
        functions.append(k)
      elif isinstance(data, type):
        classes.append(k)
      else:
        others.append(k)

  return classes, functions, others


def write_module(module_name, filename, header=None):
  module = importlib.import_module(module_name)
  classes, functions, others = get_class_funcs(module)

  fout = open(filename, 'w')
  # write header
  if header is None:
    header = f'``{module_name}`` module'
  else:
    header = header
  fout.write(header + '\n')
  fout.write('=' * len(header) + '\n\n')
  fout.write(f'.. currentmodule:: {module_name} \n')
  fout.write(f'.. automodule:: {module_name} \n\n')

  # write autosummary
  fout.write('.. autosummary::\n')
  fout.write('   :toctree: generated/\n\n')
  for m in functions:
    fout.write(f'   {m}\n')
  for m in classes:
    fout.write(f'   {m}\n')
  for m in others:
    fout.write(f'   {m}\n')

  fout.close()


def write_submodules(module_name, filename, header=None, submodule_names=(), section_names=()):
  fout = open(filename, 'w')
  # write header
  if header is None:
    header = f'``{module_name}`` module'
  else:
    header = header
  fout.write(header + '\n')
  fout.write('=' * len(header) + '\n\n')
  fout.write(f'.. currentmodule:: {module_name} \n')
  fout.write(f'.. automodule:: {module_name} \n\n')

  # whole module
  for i, name in enumerate(submodule_names):
    module = importlib.import_module(module_name + '.' + name)
    classes, functions, others = get_class_funcs(module)

    fout.write(section_names[i] + '\n')
    fout.write('-' * len(section_names[i]) + '\n\n')

    # write autosummary
    fout.write('.. autosummary::\n')
    fout.write('   :toctree: generated/\n\n')
    for m in functions:
      fout.write(f'   {m}\n')
    for m in classes:
      fout.write(f'   {m}\n')
    for m in others:
      fout.write(f'   {m}\n')

    fout.write(f'\n\n')

  fout.close()


def _get_functions(obj):
  return set([n for n in dir(obj)
              if (n not in block_list  # not in blacklist
                  and callable(getattr(obj, n))  # callable
                  and not isinstance(getattr(obj, n), type)  # not class
                  and n[0].islower()  # starts with lower char
                  and not n.startswith('__')  # not special methods
                  )
              ])


def _import(mod, klass=None, is_jax=False):
  obj = importlib.import_module(mod)
  if klass:
    obj = getattr(obj, klass)
    return obj, ':meth:`{}.{}.{{}}`'.format(mod, klass)
  else:
    if not is_jax:
      return obj, ':obj:`{}.{{}}`'.format(mod)
    else:
      from docs import implemented_jax_funcs
      return implemented_jax_funcs, ':obj:`{}.{{}}`'.format(mod)


def _generate_comparison_rst(numpy_mod, brainpy_mod, jax_mod, klass=None, header=', , ', is_jax=False):
  np_obj, np_fmt = _import(numpy_mod, klass)
  np_funcs = _get_functions(np_obj)

  bm_obj, bm_fmt = _import(brainpy_mod, klass)
  bm_funcs = _get_functions(bm_obj)

  jax_obj, jax_fmt = _import(jax_mod, klass, is_jax=is_jax)
  jax_funcs = _get_functions(jax_obj)

  buf = []
  buf += [
    '.. csv-table::',
    '   :header: {}'.format(header),
    '',
  ]
  for f in sorted(np_funcs):
    np_cell = np_fmt.format(f)
    bm_cell = bm_fmt.format(f) if f in bm_funcs else r'\-'
    jax_cell = jax_fmt.format(f) if f in jax_funcs else r'\-'
    line = '   {}, {}, {}'.format(np_cell, bm_cell, jax_cell)
    buf.append(line)

  unique_names = bm_funcs - np_funcs
  for f in sorted(unique_names):
    np_cell = r'\-'
    bm_cell = bm_fmt.format(f) if f in bm_funcs else r'\-'
    jax_cell = jax_fmt.format(f) if f in jax_funcs else r'\-'
    line = '   {}, {}, {}'.format(np_cell, bm_cell, jax_cell)
    buf.append(line)

  buf += [
    '',
    '**Summary**\n',
    '- Number of NumPy functions: {}\n'.format(len(np_funcs)),
    '- Number of functions covered by ``brainpy.math``: {}\n'.format(len(bm_funcs & np_funcs)),
    '- Number of functions unique in ``brainpy.math``: {}\n'.format(len(bm_funcs - np_funcs)),
    '- Number of functions covered by ``jax.numpy``: {}\n'.format(len(jax_funcs & np_funcs)),
  ]
  return buf


def _section(header, numpy_mod, brainpy_mod, jax_mod, klass=None, is_jax=False):
  buf = [header, '-' * len(header), '', ]
  header2 = 'NumPy, brainpy.math, jax.numpy'
  buf += _generate_comparison_rst(numpy_mod, brainpy_mod, jax_mod, klass=klass, header=header2, is_jax=is_jax)
  buf += ['']
  return buf


def generate_algorithm_docs(path='apis/auto/algorithms/'):
  if not os.path.exists(path): os.makedirs(path)

  write_module(module_name='brainpy.algorithms.offline',
               filename=os.path.join(path, 'offline.rst'),
               header='Offline Training Algorithms')
  write_module(module_name='brainpy.algorithms.online',
               filename=os.path.join(path, 'online.rst'),
               header='Online Training Algorithms')
  write_module(module_name='brainpy.algorithms.utils',
               filename=os.path.join(path, 'utils.rst'),
               header='Training Algorithm Utilities')


def generate_analysis_docs(path='apis/auto/analysis/'):
  if not os.path.exists(path):
    os.makedirs(path)
  write_module(module_name='brainpy.analysis.lowdim',
               filename=os.path.join(path, 'lowdim.rst'),
               header='Low-dimensional Analyzers')
  write_module(module_name='brainpy.analysis.highdim',
               filename=os.path.join(path, 'highdim.rst'),
               header='High-dimensional Analyzers')
  write_module(module_name='brainpy.analysis.stability',
               filename=os.path.join(path, 'stability.rst'),
               header='Stability Analysis')


def generate_train_docs(path='apis/auto/train/'):
  if not os.path.exists(path):
    os.makedirs(path)
  write_module(module_name='brainpy.train.base',
               filename=os.path.join(path, 'base.rst'),
               header='Base Training Class')
  write_module(module_name='brainpy.train.online',
               filename=os.path.join(path, 'online.rst'),
               header='Online Training Method')
  write_module(module_name='brainpy.train.offline',
               filename=os.path.join(path, 'offline.rst'),
               header='Offline Training Method')
  write_module(module_name='brainpy.train.back_propagation',
               filename=os.path.join(path, 'back_propagation.rst'),
               header='Back-propagation Training Method')


def generate_base_docs(path='apis/auto/'):
  if not os.path.exists(path):
    os.makedirs(path)

  module_and_name = [
    ('base', 'Base Class'),
    ('function', 'Function Wrapper'),
    ('collector', 'Collectors'),
    ('io', 'Exporting and Loading'),
    ('naming', 'Naming Tools')]
  write_submodules(module_name='brainpy.base',
                   filename=os.path.join(path, 'base.rst'),
                   header='``brainpy.base`` module',
                   submodule_names=[k[0] for k in module_and_name],
                   section_names=[k[1] for k in module_and_name])


def generate_connect_docs(path='apis/auto/'):
  if not os.path.exists(path):
    os.makedirs(path)

  module_and_name = [('base', 'Base Class'),
                     ('custom_conn', 'Custom Connections'),
                     ('random_conn', 'Random Connections'),
                     ('regular_conn', 'Regular Connections'), ]
  write_submodules(module_name='brainpy.connect',
                   filename=os.path.join(path, 'connect.rst'),
                   header='``brainpy.connect`` module',
                   submodule_names=[a[0] for a in module_and_name],
                   section_names=[a[1] for a in module_and_name])


def generate_datasets_docs(path='apis/auto/datasets/'):
  if not os.path.exists(path):
    os.makedirs(path)

  write_module(module_name='brainpy.datasets.chaos',
               filename=os.path.join(path, 'chaos.rst'),
               header='Chaotic Systems')
  write_module(module_name='brainpy.datasets.vision',
               filename=os.path.join(path, 'vision.rst'),
               header='Vision Datasets')


def generate_dyn_docs(path='apis/auto/dyn/'):
  if not os.path.exists(path):
    os.makedirs(path)

  write_module(module_name='brainpy.dyn.base',
               filename=os.path.join(path, 'base.rst'),
               header='Base Class')
  write_module(module_name='brainpy.dyn.runners',
               filename=os.path.join(path, 'runners.rst'),
               header='Runners')

  # "channels" module
  write_module(module_name='brainpy.dyn.channels.base',
               filename=os.path.join(path, 'channel_base.rst'),
               header='Basic Channel Classes')
  write_module(module_name='brainpy.dyn.channels.Na',
               filename=os.path.join(path, 'channel_sodium.rst'),
               header='Voltage-dependent Sodium Channel Models')
  write_module(module_name='brainpy.dyn.channels.K',
               filename=os.path.join(path, 'channel_potassium.rst'),
               header='Voltage-dependent Potassium Channel Models')
  write_module(module_name='brainpy.dyn.channels.Ca',
               filename=os.path.join(path, 'channel_calcium.rst'),
               header='Voltage-dependent Calcium Channel Models')
  write_module(module_name='brainpy.dyn.channels.KCa',
               filename=os.path.join(path, 'channel_potassium_calcium.rst'),
               header='Calcium-dependent Potassium Channel Models')
  write_module(module_name='brainpy.dyn.channels.IH',
               filename=os.path.join(path, 'channel_Ih.rst'),
               header='Hyperpolarization-activated Cation Channel Models')
  write_module(module_name='brainpy.dyn.channels.leaky',
               filename=os.path.join(path, 'channel_leaky.rst'),
               header='Leakage Channel Models')

  # "neurons" module
  module_and_name = [
    ('biological_models', 'Biological Models'),
    ('fractional_models', 'Fractional-order Models'),
    ('reduced_models', 'Reduced Models'),
    ('noise_groups', 'Noise Models'),
    ('input_groups', 'Input Models'),
  ]
  write_submodules(module_name='brainpy.dyn.neurons',
                   filename=os.path.join(path, 'neurons.rst'),
                   header='Neuron Models',
                   submodule_names=[a[0] for a in module_and_name],
                   section_names=[a[1] for a in module_and_name])

  # "layers" module
  module_and_name = [
    ('conv', 'Convolutional Layers'),
    ('dropout', 'Dropout Layers'),
    ('linear', 'Dense Connection Layers'),
    ('nvar', 'NVAR Layers'),
    ('reservoir', 'Reservoir Layers'),
    ('normalization', 'Normalization Layers'),
    ('pooling', 'Pooling Layers'),
    ('rnncells', 'Artificial Recurrent Layers'),
  ]
  write_submodules(module_name='brainpy.dyn.layers',
                   filename=os.path.join(path, 'layers.rst'),
                   header='Artificial Layers',
                   submodule_names=[a[0] for a in module_and_name],
                   section_names=[a[1] for a in module_and_name])

  # "synapses" module
  module_and_name = [
    ('abstract_models', 'Abstract Models'),
    ('biological_models', 'Biological Models'),
    ('delay_couplings', 'Coupling Models'),
    ('gap_junction', 'Gap Junction Models'),
    ('learning_rules', 'Learning Rule Models'),
  ]
  write_submodules(module_name='brainpy.dyn.synapses',
                   filename=os.path.join(path, 'synapses.rst'),
                   header='Synapse Dynamics',
                   submodule_names=[a[0] for a in module_and_name],
                   section_names=[a[1] for a in module_and_name])

  # "synouts" module
  write_module(module_name='brainpy.dyn.synouts',
               filename=os.path.join(path, 'synouts.rst'),
               header='Synaptic Outputs')

  # "synplast" module
  write_module(module_name='brainpy.dyn.synplast',
               filename=os.path.join(path, 'synplast.rst'),
               header='Synaptic Plasticity')

  # "rates" module
  module_and_name = [
    ('populations', 'Population Models'),
  ]
  write_submodules(module_name='brainpy.dyn.rates',
                   filename=os.path.join(path, 'rates.rst'),
                   header='Rate Models',
                   submodule_names=[a[0] for a in module_and_name],
                   section_names=[a[1] for a in module_and_name])


def generate_initialize_docs(path='apis/auto/'):
  if not os.path.exists(path):
    os.makedirs(path)

  module_and_name = [('base', 'Base Class'),
                     ('regular_inits', 'Regular Initializers'),
                     ('random_inits', 'Random Initializers'),
                     ('decay_inits', 'Decay Initializers'), ]
  write_submodules(module_name='brainpy.initialize',
                   filename=os.path.join(path, 'initialize.rst'),
                   header='``brainpy.initialize`` module',
                   submodule_names=[a[0] for a in module_and_name],
                   section_names=[a[1] for a in module_and_name])


def generate_inputs_docs(path='apis/auto/'):
  if not os.path.exists(path):
    os.makedirs(path)

  write_module(module_name='brainpy.inputs',
               filename=os.path.join(path, 'inputs.rst'),
               header='``brainpy.inputs`` module')


def generate_integrators_doc(path='apis/auto/integrators/'):
  if not os.path.exists(path):
    os.makedirs(path)

  # ODE
  write_module(module_name='brainpy.integrators.ode.base',
               filename=os.path.join(path, 'ode_base.rst'),
               header='Base Integrator')
  write_module(module_name='brainpy.integrators.ode.generic',
               filename=os.path.join(path, 'ode_generic.rst'),
               header='Generic Functions')
  write_module(module_name='brainpy.integrators.ode.explicit_rk',
               filename=os.path.join(path, 'ode_explicit_rk.rst'),
               header='Explicit Runge-Kutta Methods')
  write_module(module_name='brainpy.integrators.ode.adaptive_rk',
               filename=os.path.join(path, 'ode_adaptive_rk.rst'),
               header='Adaptive Runge-Kutta Methods')
  write_module(module_name='brainpy.integrators.ode.exponential',
               filename=os.path.join(path, 'ode_exponential.rst'),
               header='Exponential Integrators')

  # SDE
  write_module(module_name='brainpy.integrators.sde.base',
               filename=os.path.join(path, 'sde_base.rst'),
               header='Base Integrator')
  write_module(module_name='brainpy.integrators.sde.generic',
               filename=os.path.join(path, 'sde_generic.rst'),
               header='Generic Functions')
  write_module(module_name='brainpy.integrators.sde.normal',
               filename=os.path.join(path, 'sde_normal.rst'),
               header='Normal Methods')
  write_module(module_name='brainpy.integrators.sde.srk_scalar',
               filename=os.path.join(path, 'sde_srk_scalar.rst'),
               header='SRK methods for scalar Wiener process')

  # FDE
  write_module(module_name='brainpy.integrators.fde.base',
               filename=os.path.join(path, 'fde_base.rst'),
               header='Base Integrator')
  write_module(module_name='brainpy.integrators.fde.generic',
               filename=os.path.join(path, 'fde_generic.rst'),
               header='Generic Functions')
  write_module(module_name='brainpy.integrators.fde.Caputo',
               filename=os.path.join(path, 'fde_Caputo.rst'),
               header='Methods for Caputo Fractional Derivative')
  write_module(module_name='brainpy.integrators.fde.GL',
               filename=os.path.join(path, 'fde_GL.rst'),
               header='Methods for Riemann-Liouville Fractional Derivative')

  # Others
  write_module(module_name='brainpy.integrators.joint_eq',
               filename=os.path.join(path, 'joint_eq.rst'),
               header='Joint Equation')
  write_module(module_name='brainpy.integrators.runner',
               filename=os.path.join(path, 'runner.rst'),
               header='Integrator Runner')


def generate_losses_docs(path='apis/auto/'):
  if not os.path.exists(path):
    os.makedirs(path)

  module_and_name = [
    ('comparison', 'Comparison', ),
    ('regularization', 'Regularization', ),
  ]
  write_submodules(module_name='brainpy.losses',
                   filename=os.path.join(path, 'losses.rst'),
                   header='``brainpy.losses`` module',
                   submodule_names=[k[0] for k in module_and_name],
                   section_names=[k[1] for k in module_and_name])


def generate_math_docs(path='apis/auto/math/'):
  if not os.path.exists(path):
    os.makedirs(path)

  buf = []
  buf += _section(header='Multi-dimensional Array',
                  numpy_mod='numpy',
                  brainpy_mod='brainpy.math',
                  jax_mod='jax.numpy',
                  klass='ndarray', )
  buf += _section(header='Array Operations',
                  numpy_mod='numpy',
                  brainpy_mod='brainpy.math',
                  jax_mod='jax.numpy',
                  is_jax=True)
  buf += _section(header='Linear Algebra',
                  numpy_mod='numpy.linalg',
                  brainpy_mod='brainpy.math.linalg',
                  jax_mod='jax.numpy.linalg', )
  buf += _section(header='Discrete Fourier Transform',
                  numpy_mod='numpy.fft',
                  brainpy_mod='brainpy.math.fft',
                  jax_mod='jax.numpy.fft', )
  buf += _section(header='Random Sampling',
                  numpy_mod='numpy.random',
                  brainpy_mod='brainpy.math.random',
                  jax_mod='jax.random',)
  codes = '\n'.join(buf)

  if not os.path.exists(path):
    os.makedirs(path)
  with open(os.path.join(path, 'comparison_table.rst.inc'), 'w') as f:
    f.write(codes)

  module_and_name = [
    ('pre_syn_post',   '``pre-syn-post`` Transformations',),
    ('sparse_matmul', 'Sparse Matrix Multiplication',),
    ('event_matmul', 'Event-based Matrix Multiplication',),
    ('spikegrad',      'Surrogate Gradients for Spike Operation',),
    ('op_register',    'Operator Registration',),
    ('wrap_jax',       'Other Operators',),
  ]
  write_submodules(module_name='brainpy.math.operators',
                   filename=os.path.join(path, 'operators.rst'),
                   header='Sparse & Event-based Operators',
                   submodule_names=[k[0] for k in module_and_name],
                   section_names=[k[1] for k in module_and_name])

  write_module(module_name='brainpy.math.activations',
               filename=os.path.join(path, 'activations.rst'),
               header='Activation Functions')
  write_module(module_name='brainpy.math.autograd',
               filename=os.path.join(path, 'autograd.rst'),
               header='Automatic Differentiation')
  write_module(module_name='brainpy.math.controls',
               filename=os.path.join(path, 'controls.rst'),
               header='Control Flows')

  write_module(module_name='brainpy.math.parallels',
               filename=os.path.join(path, 'parallels.rst'),
               header='Parallel Compilation')
  write_module(module_name='brainpy.math.jit',
               filename=os.path.join(path, 'jit.rst'),
               header='JIT Compilation')
  write_module(module_name='brainpy.math.jaxarray',
               filename=os.path.join(path, 'variables.rst'),
               header='Math Variables')
  write_module(module_name='brainpy.math.setting',
               filename=os.path.join(path, 'setting.rst'),
               header='Setting')
  write_module(module_name='brainpy.math.function',
               filename=os.path.join(path, 'function.rst'),
               header='Function')
  write_module(module_name='brainpy.math.delayvars',
               filename=os.path.join(path, 'delay_vars.rst'),
               header='Delay Variables')


def generate_measure_docs(path='apis/auto/'):
  if not os.path.exists(path):
    os.makedirs(path)

  write_module(module_name='brainpy.measure',
               filename=os.path.join(path, 'measure.rst'),
               header='``brainpy.measure`` module')



def generate_optimizers_docs(path='apis/auto/'):
  if not os.path.exists(path):
    os.makedirs(path)

  module_and_name = [
    ('optimizer', 'Optimizers'),
    ('scheduler', 'Schedulers'),
  ]
  write_submodules(module_name='brainpy.optimizers',
                   filename=os.path.join(path, 'optimizers.rst'),
                   header='``brainpy.optimizers`` module',
                   submodule_names=[k[0] for k in module_and_name],
                   section_names=[k[1] for k in module_and_name])


def generate_running_docs(path='apis/auto/'):
  if not os.path.exists(path):
    os.makedirs(path)

  module_and_name = [
    ('multiprocess', 'Parallel Pool'),
    ('runner', 'Runners')
  ]
  write_submodules(module_name='brainpy.running',
                   filename=os.path.join(path, 'running.rst'),
                   header='``brainpy.running`` module',
                   submodule_names=[k[0] for k in module_and_name],
                   section_names=[k[1] for k in module_and_name])


def generate_tools_docs(path='apis/auto/tools/'):
  if not os.path.exists(path):
    os.makedirs(path)

  write_module(module_name='brainpy.tools.checking',
               filename=os.path.join(path, 'checking.rst'),
               header='Type Checking')
  write_module(module_name='brainpy.tools.codes',
               filename=os.path.join(path, 'codes.rst'),
               header='Code Tools')
  write_module(module_name='brainpy.tools.others',
               filename=os.path.join(path, 'others.rst'),
               header='Other Tools')
  write_module(module_name='brainpy.tools.errors',
               filename=os.path.join(path, 'errors.rst'),
               header='Error Tools')


# ---------- #
# Deprecated #
# ---------- #

def generate_nn_docs(path='apis/auto/nn/'):
  if not os.path.exists(path):
    os.makedirs(path)

  write_module(module_name='brainpy.nn.base',
               filename=os.path.join(path, 'base.rst'),
               header='Base Classes')
  write_module(module_name='brainpy.nn.operations',
               filename=os.path.join(path, 'operations.rst'),
               header='Node Operations')
  write_module(module_name='brainpy.nn.graph_flow',
               filename=os.path.join(path, 'graph_flow.rst'),
               header='Node Graph Tools')
  write_module(module_name='brainpy.nn.datatypes',
               filename=os.path.join(path, 'data_types.rst'),
               header='Data Types')

  module_and_name = [
    ('rnn_runner', 'Base RNN Runner'),
    ('rnn_trainer', 'Base RNN Trainer'),
    ('online_trainer', 'Online RNN Trainer'),
    ('offline_trainer', 'Offline RNN Trainer'),
    ('back_propagation', 'Back-propagation Trainer'),
  ]
  write_submodules(module_name='brainpy.nn.runners',
                   filename=os.path.join(path, 'runners.rst'),
                   header='Runners and Trainers',
                   submodule_names=[k[0] for k in module_and_name],
                   section_names=[k[1] for k in module_and_name])

  module_and_name = [
    ('online', 'Online Training Algorithms'),
    ('offline', 'Offline Training Algorithms'),
  ]
  write_submodules(module_name='brainpy.nn.algorithms',
                   filename=os.path.join(path, 'algorithms.rst'),
                   header='Training Algorithms',
                   submodule_names=[k[0] for k in module_and_name],
                   section_names=[k[1] for k in module_and_name])

  write_module(module_name='brainpy.nn.nodes.base',
               filename=os.path.join(path, 'nodes_base.rst'),
               header='Nodes: basic')
  write_module(module_name='brainpy.nn.nodes.ANN',
               filename=os.path.join(path, 'nodes_ANN.rst'),
               header='Nodes: artificial neural network ')
  write_module(module_name='brainpy.nn.nodes.RC',
               filename=os.path.join(path, 'nodes_RC.rst'),
               header='Nodes: reservoir computing')

def generate_compact_docs(path='apis/auto/compat/'):
  if not os.path.exists(path):
    os.makedirs(path)

  write_module(module_name='brainpy.compat.brainobjects',
               filename=os.path.join(path, 'brainobjects.rst'),
               header='Brain Objects')
  write_module(module_name='brainpy.compat.integrators',
               filename=os.path.join(path, 'integrators.rst'),
               header='Integrators')
  write_module(module_name='brainpy.compat.layers',
               filename=os.path.join(path, 'layers.rst'),
               header='Layers')
  write_module(module_name='brainpy.compat.monitor',
               filename=os.path.join(path, 'monitor.rst'),
               header='Monitor')
  write_module(module_name='brainpy.compat.runners',
               filename=os.path.join(path, 'runners.rst'),
               header='Runners')

  write_module(module_name='brainpy.compat.nn.base',
               filename=os.path.join(path, 'nn_base.rst'),
               header='Base Classes')
  write_module(module_name='brainpy.compat.nn.operations',
               filename=os.path.join(path, 'nn_operations.rst'),
               header='Node Operations')
  write_module(module_name='brainpy.compat.nn.graph_flow',
               filename=os.path.join(path, 'nn_graph_flow.rst'),
               header='Node Graph Tools')
  write_module(module_name='brainpy.compat.nn.datatypes',
               filename=os.path.join(path, 'nn_data_types.rst'),
               header='Data Types')
  module_and_name = [
    ('rnn_runner', 'Base RNN Runner'),
    ('rnn_trainer', 'Base RNN Trainer'),
    ('online_trainer', 'Online RNN Trainer'),
    ('offline_trainer', 'Offline RNN Trainer'),
    ('back_propagation', 'Back-propagation Trainer'),
  ]
  write_submodules(module_name='brainpy.compat.nn.runners',
                   filename=os.path.join(path, 'nn_runners.rst'),
                   header='Runners and Trainers',
                   submodule_names=[k[0] for k in module_and_name],
                   section_names=[k[1] for k in module_and_name])
  module_and_name = [
    ('online', 'Online Training Algorithms'),
    ('offline', 'Offline Training Algorithms'),
  ]
  write_submodules(module_name='brainpy.compat.nn.algorithms',
                   filename=os.path.join(path, 'nn_algorithms.rst'),
                   header='Training Algorithms',
                   submodule_names=[k[0] for k in module_and_name],
                   section_names=[k[1] for k in module_and_name])
  write_module(module_name='brainpy.compat.nn.nodes.base',
               filename=os.path.join(path, 'nn_nodes_base.rst'),
               header='Nodes: basic')
  write_module(module_name='brainpy.compat.nn.nodes.ANN',
               filename=os.path.join(path, 'nn_nodes_ANN.rst'),
               header='Nodes: artificial neural network ')
  write_module(module_name='brainpy.compat.nn.nodes.RC',
               filename=os.path.join(path, 'nn_nodes_RC.rst'),
               header='Nodes: reservoir computing')


def generate_math_compact_docs(path='apis/auto/math/'):
  if not os.path.exists(path):
    os.makedirs(path)


  write_module(module_name='brainpy.math.compat.optimizers',
               filename=os.path.join(path, 'optimizers.rst'),
               header='Optimizers')

  write_module(module_name='brainpy.math.compat.losses',
               filename=os.path.join(path, 'losses.rst'),
               header='Losses')
