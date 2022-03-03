# -*- coding: utf-8 -*-

import importlib
import inspect
import os

from brainpy.math import (activations, autograd, controls, function,
                          jit, operators, parallels, setting)
from brainpy import optimizers


def get_class_funcs(module):
  classes, functions = [], []
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

  return classes, functions


def write_module(module_name, filename, header=None):
  module = importlib.import_module(module_name)
  classes, functions = get_class_funcs(module)

  # if '__all__' not in module.__dict__:
  #   raise ValueError(f'Only support auto generate APIs in a module has __all__ '
  #                    f'specification, while __all__ is not specified in {module_name}')

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

  # # write autoclass
  # fout.write('\n')
  # for m in classes:
  #   fout.write(f'.. autoclass:: {m}\n')
  #   fout.write(f'   :members:\n\n')

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
    classes, functions = get_class_funcs(module)

    fout.write(section_names[i] + '\n')
    fout.write('-' * len(section_names[i]) + '\n\n')

    # write autosummary
    fout.write('.. autosummary::\n')
    fout.write('   :toctree: generated/\n\n')
    for m in functions:
      fout.write(f'   {m}\n')
    for m in classes:
      fout.write(f'   {m}\n')

    fout.write(f'\n\n')

    # # write autoclass
    # fout.write('\n')
    # for m in classes:
    #   fout.write(f'.. autoclass:: {m}\n')
    #   fout.write(f'   :members:\n\n')

  fout.close()


block_list = ['test', 'register_pytree_node']
for module in [jit, autograd, function,
               controls, activations, optimizers,
               operators, parallels, setting]:
  for k in dir(module):
    if (not k.startswith('_')) and (not inspect.ismodule(getattr(module, k))):
      block_list.append(k)


def _get_functions(obj):
  return set([n for n in dir(obj)
              if (n not in block_list  # not in blacklist
                  and callable(getattr(obj, n))  # callable
                  and not isinstance(getattr(obj, n), type)  # not class
                  and n[0].islower()  # starts with lower char
                  and not n.startswith('__')  # not special methods
                  )
              ])


def _import(mod, klass):
  obj = importlib.import_module(mod)
  if klass:
    obj = getattr(obj, klass)
    return obj, ':meth:`{}.{}.{{}}`'.format(mod, klass)
  else:
    # ufunc is not a function
    return obj, ':obj:`{}.{{}}`'.format(mod)


def _generate_comparison_rst(numpy_mod, brainpy_jax, klass, header=', , '):
  np_obj, np_fmt = _import(numpy_mod, klass)
  np_funcs = _get_functions(np_obj)
  brainpy_jax_obj, brainpy_jax_fmt = _import(brainpy_jax, klass)
  brainpy_funcs = _get_functions(brainpy_jax_obj)

  buf = []
  buf += [
    '.. csv-table::',
    '   :header: {}'.format(header),
    '',
  ]
  for f in sorted(np_funcs):
    np_cell = np_fmt.format(f)
    brainpy_cell = brainpy_jax_fmt.format(f) if f in brainpy_funcs else r'\-'
    line = '   {}, {}'.format(np_cell, brainpy_cell)
    buf.append(line)

  unique_names = brainpy_funcs - np_funcs
  for f in sorted(unique_names):
    np_cell = r'\-'
    brainpy_cell = brainpy_jax_fmt.format(f) if f in brainpy_funcs else r'\-'
    line = '   {}, {}'.format(np_cell, brainpy_cell)
    buf.append(line)

  buf += [
    '',
    '**Summary**\n',
    '- Number of NumPy functions: {}\n'.format(len(np_funcs)),
    '- Number of functions covered by ``brainpy.math``: {}\n'.format(
      len(brainpy_funcs & np_funcs)),
  ]
  return buf


def _section(header, numpy_mod, brainpy_jax, klass=None):
  buf = [header, '-' * len(header), '', ]
  header2 = 'NumPy, brainpy.math'
  buf += _generate_comparison_rst(numpy_mod, brainpy_jax, klass, header=header2)
  buf += ['']
  return buf


def generate_analysis_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  # module_and_name = [
  #   ('lowdim', 'Low-Dimensional Analyzers'),
  #   ('highdim', 'High-Dimensional Analyzers'),
  #   ('stability', 'Stability Analysis'), ]
  # write_submodules(module_name='brainpy.analysis',
  #                  filename=os.path.join(path, 'analysis.rst'),
  #                  header='``brainpy.analysis`` module',
  #                  submodule_names=[k[0] for k in module_and_name],
  #                  section_names=[k[1] for k in module_and_name])

  write_module(module_name='brainpy.analysis.lowdim',
               filename=os.path.join(path, 'lowdim.rst'),
               header='Low-dimensional Analyzers')

  write_module(module_name='brainpy.analysis.highdim',
               filename=os.path.join(path, 'highdim.rst'),
               header='High-dimensional Analyzers')

  # write_module(module_name='brainpy.analysis.continuation',
  #              filename=os.path.join(path, 'continuation.rst'),
  #              header='Continuation Analysis')

  write_module(module_name='brainpy.analysis.stability',
               filename=os.path.join(path, 'stability.rst'),
               header='Stability Analysis')


def generate_base_docs(path):
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


def generate_connect_docs(path):
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


def generate_datasets_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  module_and_name = [('chaotic_system', 'Chaotic Systems'), ]
  write_submodules(module_name='brainpy.datasets',
                   filename=os.path.join(path, 'datasets.rst'),
                   header='``brainpy.datasets`` module',
                   submodule_names=[a[0] for a in module_and_name],
                   section_names=[a[1] for a in module_and_name])


def generate_dyn_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  write_module(module_name='brainpy.dyn.base',
               filename=os.path.join(path, 'base.rst'),
               header='Base Class')

  module_and_name = [('biological_models', 'Biological Models'),
                     ('IF_models', 'Integrate-and-Fire Models'),
                     ('input_models', 'Input Models'),
                     ('rate_models', 'Rate Models'),
                     ('reduced_models', 'Reduced Models'), ]
  write_submodules(module_name='brainpy.dyn.neurons',
                   filename=os.path.join(path, 'neurons.rst'),
                   header='Neuron Models',
                   submodule_names=[a[0] for a in module_and_name],
                   section_names=[a[1] for a in module_and_name])

  module_and_name = [('biological_models', 'Biological Models'),
                     ('abstract_models', 'Abstract Models'),
                     ('learning_rules', 'Learning Rules'), ]
  write_submodules(module_name='brainpy.dyn.synapses',
                   filename=os.path.join(path, 'synapses.rst'),
                   header='Synapse Models',
                   submodule_names=[a[0] for a in module_and_name],
                   section_names=[a[1] for a in module_and_name])

  write_module(module_name='brainpy.dyn.runners',
               filename=os.path.join(path, 'runners.rst'),
               header='Runners')


def generate_initialize_docs(path):
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


def generate_inputs_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  write_module(module_name='brainpy.inputs',
               filename=os.path.join(path, 'inputs.rst'),
               header='``brainpy.input`` module')


def generate_integrators_doc(path):
  if not os.path.exists(path):
    os.makedirs(path)

  write_module(module_name='brainpy.integrators.ode.explicit_rk',
               filename=os.path.join(path, 'ode_explicit_rk.rst'),
               header='Explicit Runge-Kutta Methods')
  write_module(module_name='brainpy.integrators.ode.adaptive_rk',
               filename=os.path.join(path, 'ode_adaptive_rk.rst'),
               header='Adaptive Runge-Kutta Methods')
  write_module(module_name='brainpy.integrators.ode.exponential',
               filename=os.path.join(path, 'ode_exponential.rst'),
               header='Exponential Integrators')
  write_module(module_name='brainpy.integrators.joint_eq',
               filename=os.path.join(path, 'joint_eq.rst'),
               header='Joint Equation')


def generate_losses_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  write_module(module_name='brainpy.losses',
               filename=os.path.join(path, 'losses.rst'),
               header='``brainpy.losses`` module')


def generate_math_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  buf = []
  buf += _section(header='Multi-dimensional Array',
                  numpy_mod='numpy',
                  brainpy_jax='brainpy.math',
                  klass='ndarray')
  buf += _section(header='Array Operations',
                  numpy_mod='numpy',
                  brainpy_jax='brainpy.math')
  buf += _section(header='Linear Algebra',
                  numpy_mod='numpy.linalg',
                  brainpy_jax='brainpy.math.linalg')
  buf += _section(header='Discrete Fourier Transform',
                  numpy_mod='numpy.fft',
                  brainpy_jax='brainpy.math.fft')
  buf += _section(header='Random Sampling',
                  numpy_mod='numpy.random',
                  brainpy_jax='brainpy.math.random')
  codes = '\n'.join(buf)

  if not os.path.exists(path):
    os.makedirs(path)
  with open(os.path.join(path, 'comparison_table.rst.inc'), 'w') as f:
    f.write(codes)

  write_module(module_name='brainpy.math.activations',
               filename=os.path.join(path, 'activations.rst'),
               header='Activation Functions')
  write_module(module_name='brainpy.math.autograd',
               filename=os.path.join(path, 'autograd.rst'),
               header='Automatic Differentiation')
  write_module(module_name='brainpy.math.controls',
               filename=os.path.join(path, 'controls.rst'),
               header='Control Flows')
  write_module(module_name='brainpy.math.operators',
               filename=os.path.join(path, 'operators.rst'),
               header='Operators')
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


def generate_measure_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  write_module(module_name='brainpy.measure',
               filename=os.path.join(path, 'measure.rst'),
               header='``brainpy.measure`` module')


def generate_nn_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  write_module(module_name='brainpy.nn.base',
               filename=os.path.join(path, 'base.rst'),
               header='Base Class')

  module_and_name = [
    ('conv', 'Convolution Layers'),
    ('dropout', 'Dropout Layers'),
    ('rnn_cells', 'RNN Cells'),
  ]
  write_submodules(module_name='brainpy.nn.nodes.ANN',
               filename=os.path.join(path, 'nodes_ANN.rst'),
               header='Artificial Neural Network Nodes',
               submodule_names=[k[0] for k in module_and_name],
               section_names=[k[1] for k in module_and_name])

  module_and_name = [
    ('activation', 'Activations'),
    ('dense', 'Dense Layers'),
    ('io', 'Exporting and Loading'),
    ('ops', 'Operators')
  ]
  write_submodules(module_name='brainpy.nn.nodes.base',
                   filename=os.path.join(path, 'nodes_base.rst'),
                   header='Node Basics',
                   submodule_names=[k[0] for k in module_and_name],
                   section_names=[k[1] for k in module_and_name])

  module_and_name = [
    ('linear_readout', 'Linear Readout'),
    ('nvar', 'Nonlinear Vector Autoregression'),
    ('reservoir', 'Reservoir'),
  ]
  write_submodules(module_name='brainpy.nn.nodes.RC',
                   filename=os.path.join(path, 'nodes_RC.rst'),
                   header='Reservoir Computing Nodes',
                   submodule_names=[k[0] for k in module_and_name],
                   section_names=[k[1] for k in module_and_name])

  write_module(module_name='brainpy.nn.operations',
               filename=os.path.join(path, 'operations.rst'),
               header='Construction Operations')

  # write_module(module_name='brainpy.nn.runners',
  #              filename=os.path.join(path, 'runners.rst'),
  #              header='Runners')

  module_and_name = [
    ('back_propagation', 'Back Propagation'),
    ('force_learning', 'Force Learning'),
    ('ridge_regression', 'Ridge Regression'),
    ('rnn_runner', 'Rnn Runner'),
    ('rnn_trainer', 'RNN Trainer'),
  ]
  write_submodules(module_name='brainpy.nn.runners',
                   filename=os.path.join(path, 'runners.rst'),
                   header='Runners',
                   submodule_names=[k[0] for k in module_and_name],
                   section_names=[k[1] for k in module_and_name])

  write_module(module_name='brainpy.nn.graph_flow',
               filename=os.path.join(path, 'graph_flow.rst'),
               header='Graph FLow')

  # module_and_name = [
  #   ('base', 'Base Class'),
  #   ('constants', 'Constants'),
  #   ('graph_flow', 'Graph Flow'),
  #   ('operations', 'Operations'),
  #   ('utils', 'Utilizations'),
  # ]
  # write_submodules(module_name='brainpy.nn',
  #                  filename=os.path.join(path, 'nn.rst'),
  #                  header='``brainpy.nn`` module',
  #                  submodule_names=[k[0] for k in module_and_name],
  #                  section_names=[k[1] for k in module_and_name])


def generate_optimizers_docs(path):
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


def generate_running_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  module_and_name = [
    ('monitor', 'Monitors'),
    ('parallel', 'Parallel Pool'),
    ('runner', 'Runners')
  ]
  write_submodules(module_name='brainpy.running',
                   filename=os.path.join(path, 'running.rst'),
                   header='``brainpy.running`` module',
                   submodule_names=[k[0] for k in module_and_name],
                   section_names=[k[1] for k in module_and_name])


def generate_tools_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  module_and_name = [
    ('ast2code', 'AST-to-Code'),
    ('checking', 'Check Tools'),
    ('codes', 'Code Tools'),
    ('dicts', 'Dict Tools'),
    ('numba_tools', 'Numba Tools'),
    ('others', 'Other Tools'),
  ]
  write_submodules(module_name='brainpy.tools',
                   filename=os.path.join(path, 'tools.rst'),
                   header='``brainpy.tools`` module',
                   submodule_names=[k[0] for k in module_and_name],
                   section_names=[k[1] for k in module_and_name])


def generate_visualization_docs(path):
  # raise NotImplementedError
  if not os.path.exists(path):
    os.makedirs(path)

  module_and_name = [
    ('base', 'Base Class'),
    ('figures', 'Figures'),
    ('plots', 'Plots'),
    ('styles', 'Styles')
  ]
  write_submodules(module_name='brainpy.visualization',
                   filename=os.path.join(path, 'tools.visualization'),
                   header='``brainpy.visualization`` module',
                   submodule_names=[k[0] for k in module_and_name],
                   section_names=[k[1] for k in module_and_name])


def generate_building_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  # submodules in 'simulation' package
  write_module(module_name='brainpy.compact.brainobjects',
               filename=os.path.join(path, 'brainobjects.rst'),
               header='Brain Objects')

  module_and_name = [('neurons', 'Neuron Models'),
                     ('synapses', 'Synapse Models'), ]
  write_submodules(module_name='brainpy.dyn',
                   filename=os.path.join(path, 'models.rst'),
                   header='Brain Models',
                   submodule_names=[a[0] for a in module_and_name],
                   section_names=[a[1] for a in module_and_name])

  module_and_name = [('base', 'Base Class'),
                     ('custom_conn', 'Custom Connections'),
                     ('random_conn', 'Random Connections'),
                     ('regular_conn', 'Regular Connections'), ]
  write_submodules(module_name='brainpy.connect',
                   filename=os.path.join(path, 'connect.rst'),
                   header='Synaptic Connectivity',
                   submodule_names=[a[0] for a in module_and_name],
                   section_names=[a[1] for a in module_and_name])

  write_module(module_name='brainpy.inputs',
               filename=os.path.join(path, 'inputs.rst'),
               header='Current Inputs')

  write_module(module_name='brainpy.training.layers',
               filename=os.path.join(path, 'layers.rst'),
               header='DNN Layers')
  module_and_name = [('base', 'Base Class'),
                     ('regular_inits', 'Regular Initializers'),
                     ('random_inits', 'Random Initializers'),
                     ('decay_inits', 'Decay Initializers'), ]
  write_submodules(module_name='brainpy.initialize',
                   filename=os.path.join(path, 'initialize.rst'),
                   header='Weight Initialization',
                   submodule_names=[a[0] for a in module_and_name],
                   section_names=[a[1] for a in module_and_name])


def generate_simulation_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  # py-files in 'simulation' package
  write_module(module_name='brainpy.measure',
               filename=os.path.join(path, 'measure.rst'),
               header='Measurements')
  write_module(module_name='brainpy.running.monitor',
               filename=os.path.join(path, 'monitor.rst'),
               header='Monitors')
  write_module(module_name='brainpy.running.runner',
               filename=os.path.join(path, 'runner.rst'),
               header='Runners')
  write_module(module_name='brainpy.running.parallel',
               filename=os.path.join(path, 'parallel.rst'),
               header='Parallel Running')


def generate_training_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  # submodules in 'training' package


