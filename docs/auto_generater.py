# -*- coding: utf-8 -*-

import importlib
import inspect
import os

block_list = ['test', 'register_pytree_node', 'call', 'namedtuple', 'jit', 'wraps', 'index', 'function']


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


def _write_module(module_name, filename, header=None, template=False):
  module = importlib.import_module(module_name)
  classes, functions, others = get_class_funcs(module)

  fout = open(filename, 'w')
  # write header
  if header is None:
    header = f'``{module_name}`` module'
  fout.write(header + '\n')
  fout.write('=' * len(header) + '\n\n')
  fout.write(f'.. currentmodule:: {module_name} \n')
  fout.write(f'.. automodule:: {module_name} \n\n')

  # write autosummary
  fout.write('.. autosummary::\n')
  if template:
    fout.write('   :template: class_template.rst\n')
  fout.write('   :toctree: generated/\n\n')
  for m in functions:
    fout.write(f'   {m}\n')
  for m in classes:
    fout.write(f'   {m}\n')
  for m in others:
    fout.write(f'   {m}\n')

  fout.close()


def _write_submodules(module_name, filename, header=None, submodule_names=(), section_names=()):
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


def _write_subsections(module_name,
                       filename,
                       subsections: dict,
                       header: str = None):
  fout = open(filename, 'w')
  header = f'``{module_name}`` module' if header is None else header
  fout.write(header + '\n')
  fout.write('=' * len(header) + '\n\n')
  fout.write(f'.. currentmodule:: {module_name} \n')
  fout.write(f'.. automodule:: {module_name} \n\n')

  fout.write('.. contents::' + '\n')
  fout.write('   :local:' + '\n')
  fout.write('   :depth: 1' + '\n\n')

  for name, values in subsections.items():
    fout.write(name + '\n')
    fout.write('-' * len(name) + '\n\n')
    fout.write('.. autosummary::\n')
    fout.write('   :toctree: generated/\n\n')
    for m in values:
      fout.write(f'   {m}\n')
    fout.write(f'\n\n')

  fout.close()


def _write_subsections_v2(module_path,
                          out_path,
                          filename,
                          subsections: dict,
                          header: str = None):
  fout = open(filename, 'w')
  header = f'``{out_path}`` module' if header is None else header
  fout.write(header + '\n')
  fout.write('=' * len(header) + '\n\n')
  fout.write(f'.. currentmodule:: {out_path} \n')
  fout.write(f'.. automodule:: {out_path} \n\n')

  fout.write('.. contents::' + '\n')
  fout.write('   :local:' + '\n')
  fout.write('   :depth: 1' + '\n\n')

  for name, subheader in subsections.items():
    module = importlib.import_module(f'{module_path}.{name}')
    classes, functions, others = get_class_funcs(module)

    fout.write(subheader + '\n')
    fout.write('-' * len(subheader) + '\n\n')
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


def _write_subsections_v3(module_path,
                          out_path,
                          filename,
                          subsections: dict,
                          header: str = None):
  fout = open(filename, 'w')
  header = f'``{out_path}`` module' if header is None else header
  fout.write(header + '\n')
  fout.write('=' * len(header) + '\n\n')
  fout.write(f'.. currentmodule:: {out_path} \n')
  fout.write(f'.. automodule:: {out_path} \n\n')

  fout.write('.. contents::' + '\n')
  fout.write('   :local:' + '\n')
  fout.write('   :depth: 2' + '\n\n')

  for section in subsections:
    fout.write(subsections[section]['header'] + '\n')
    fout.write('-' * len(subsections[section]['header']) + '\n\n')

    fout.write(f'.. currentmodule:: {out_path}.{section} \n')
    fout.write(f'.. automodule:: {out_path}.{section} \n\n')

    for name, subheader in subsections[section]['content'].items():
      module = importlib.import_module(f'{module_path}.{section}.{name}')
      classes, functions, others = get_class_funcs(module)

      fout.write(subheader + '\n')
      fout.write('~' * len(subheader) + '\n\n')
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


def generate_analysis_docs():
  _write_subsections(
    module_name='brainpy.analysis',
    filename='apis/auto/analysis.rst',
    subsections={
      'Low-dimensional Analyzers': ['PhasePlane1D',
                                    'PhasePlane2D',
                                    'Bifurcation1D',
                                    'Bifurcation2D',
                                    'FastSlow1D',
                                    'FastSlow2D'],
      'High-dimensional Analyzers': ['SlowPointFinder']
    }
  )


def generate_connect_docs():
  _write_subsections_v2(
    'brainpy._src.connect',
    'brainpy.connect',
    'apis/auto/connect.rst',
    subsections={
      'base': 'Base Connection Classes and Tools',
      'custom_conn': 'Custom Connections',
      'random_conn': 'Random Connections',
      'regular_conn': 'Regular Connections',
    }
  )


def generate_channels_docs():
  _write_subsections_v2(
    'brainpy._src.dyn.channels',
    'brainpy.channels',
    'apis/auto/channels.rst',
    subsections={
      'base': 'Basic Channel Classes',
      'Na': 'Voltage-dependent Sodium Channel Models',
      'K': 'Voltage-dependent Potassium Channel Models',
      'Ca': 'Voltage-dependent Calcium Channel Models',
      'KCa': 'Calcium-dependent Potassium Channel Models',
      'IH': 'Hyperpolarization-activated Cation Channel Models',
      'leaky': 'Leakage Channel Models',
    }
  )


def generate_encoding_docs():
  _write_module(module_name='brainpy.encoding',
                filename='apis/auto/encoding.rst',
                header='``brainpy.encoding`` module')


def generate_initialize_docs():
  _write_subsections_v2(
    'brainpy._src.initialize',
    'brainpy.initialize',
    'apis/auto/initialize.rst',
    subsections={
      'base': 'Basic Initialization Classes',
      'regular_inits': 'Regular Initializers',
      'random_inits': 'Random Initializers',
      'decay_inits': 'Decay Initializers',
    }
  )


def generate_inputs_docs():
  _write_module(module_name='brainpy.inputs',
                filename='apis/auto/inputs.rst',
                header='``brainpy.inputs`` module')


def generate_layers_docs():
  _write_subsections_v2(
    'brainpy._src.dyn.layers',
    'brainpy.layers',
    'apis/auto/layers.rst',
    subsections={
      'base': 'Basic ANN Layer Class',
      'conv': 'Convolutional Layers',
      'dropout': 'Dropout Layers',
      'function': 'Function Layers',
      'linear': 'Dense Connection Layers',
      'normalization': 'Normalization Layers',
      'nvar': 'NVAR Layers',
      'pooling': 'Pooling Layers',
      'reservoir': 'Reservoir Layers',
      'rnncells': 'Artificial Recurrent Layers',
    }
  )


def generate_losses_docs():
  _write_subsections_v2(
    'brainpy._src.losses',
    'brainpy.losses',
    'apis/auto/losses.rst',
    subsections={
      'comparison': 'Comparison',
      'regularization': 'Regularization',
    }
  )


def generate_measure_docs():
  _write_module(module_name='brainpy.measure',
                filename='apis/auto/measure.rst',
                header='``brainpy.measure`` module')


def generate_neurons_docs():
  _write_subsections_v2(
    'brainpy._src.dyn.neurons',
    'brainpy.neurons',
    'apis/auto/neurons.rst',
    subsections={
      'biological_models': 'Biological Models',
      'fractional_models': 'Fractional-order Models',
      'reduced_models': 'Reduced Models',
      'noise_groups': 'Noise Models',
      'input_groups': 'Input Models',
    }
  )


def generate_optim_docs():
  _write_subsections_v2(
    'brainpy._src.optimizers',
    'brainpy.optim',
    'apis/auto/optim.rst',
    subsections={
      'optimizer': 'Optimizers',
      'scheduler': 'Schedulers',
    }
  )


def generate_rates_docs():
  _write_module(module_name='brainpy.rates',
                filename='apis/auto/rates.rst',
                header='``brainpy.rates`` module')


def generate_running_docs():
  _write_module(module_name='brainpy.running',
                filename='apis/auto/running.rst',
                header='``brainpy.running`` module')


def generate_synapses_docs():
  _write_subsections_v2(
    'brainpy._src.dyn.synapses',
    'brainpy.synapses',
    'apis/auto/synapses.rst',
    subsections={
      'abstract_models': 'Abstract Models',
      'biological_models': 'Biological Models',
      'delay_couplings': 'Coupling Models',
      'gap_junction': 'Gap Junction Models',
      'learning_rules': 'Learning Rule Models',
    }
  )


def generate_synouts_docs():
  _write_module(module_name='brainpy.synouts',
                filename='apis/auto/synouts.rst',
                header='``brainpy.synouts`` module')


def generate_synplast_docs():
  _write_module(module_name='brainpy.synplast',
                filename='apis/auto/synplast.rst',
                header='``brainpy.synplast`` module')


def generate_brainpy_docs():
  _write_subsections(
    module_name='brainpy',
    filename='apis/auto/brainpy.rst',
    subsections={
      'Numerical Differential Integration': ['Integrator',
                                             'JointEq',
                                             'IntegratorRunner',
                                             'odeint',
                                             'sdeint',
                                             'fdeint'],
      'Building Dynamical System': ['DynamicalSystem',
                                    'Container',
                                    'Sequential',
                                    'Network',
                                    'NeuGroup',
                                    'SynConn',
                                    'SynOut',
                                    'SynSTP',
                                    'SynLTP',
                                    'TwoEndConn',
                                    'CondNeuGroup',
                                    'Channel',
                                    ],
      'Simulating Dynamical System': ['DSRunner'],
      'Training Dynamical System': ['DSTrainer',
                                    'BPTT',
                                    'BPFF',
                                    'OnlineTrainer',
                                    'ForceTrainer',
                                    'OfflineTrainer',
                                    'RidgeTrainer'],
      'Dynamical System Helpers': ['DSPartial', 'NoSharedArg', 'LoopOverTime'],
    }
  )


def generate_integrators_doc():
  _write_subsections_v3(
    'brainpy._src.integrators',
    'brainpy.integrators',
    'apis/auto/integrators.rst',
    subsections={
      'ode': {'header': 'ODE integrators',
              'content': {'base': 'Base ODE Integrator',
                          'generic': 'Generic ODE Functions',
                          'explicit_rk': 'Explicit Runge-Kutta ODE Integrators',
                          'adaptive_rk': 'Adaptive Runge-Kutta ODE Integrators',
                          'exponential': 'Exponential ODE Integrators', }},
      'sde': {'header': 'SDE integrators',
              'content': {'base': 'Base SDE Integrator',
                          'generic': 'Generic SDE Functions',
                          'normal': 'Normal SDE Integrators',
                          'srk_scalar': 'SRK methods for scalar Wiener process'}},
      'fde': {'header': 'FDE integrators',
              'content': {'base': 'Base FDE Integrator',
                          'generic': 'Generic FDE Functions',
                          'Caputo': 'Methods for Caputo Fractional Derivative',
                          'GL': 'Methods for Riemann-Liouville Fractional Derivative'}}

    }
  )


def generate_math_docs():
  _write_subsections_v2(
    'brainpy.math',
    'brainpy.math',
    'apis/auto/math.rst',
    subsections={
      'object_base': 'Basis for Object-oriented Transformations',
      'object_transform': 'Object-oriented Transformations',
      'operators': 'Brain Dynamics Dedicated Operators',
      'activations': 'Activation Functions',
      'arrayoperation': 'Array Operations',
      'delayvars': 'Delay Variables',
      'environment': 'Environment Settings',
      'modes': 'Computing Modes',
    }
  )
  _write_module(
    module_name='brainpy.math.random',
    filename='apis/auto/math_random.rst',
    # header='Random Number Generations'
  )
  _write_module(
    module_name='brainpy.math.surrogate',
    filename='apis/auto/math_surrogate.rst',
    # header='Surrogate Gradient Functions'
  )


def generate_algorithm_docs(path='apis/auto/algorithms/'):
  if not os.path.exists(path): os.makedirs(path)

  module_and_name = [
    ('offline', 'Offline Training Algorithms'),
    ('online', 'Online Training Algorithms'),
    ('utils', 'Training Algorithm Utilities'),
  ]
  _write_submodules(module_name='brainpy.algorithms',
                    filename=os.path.join(path, 'algorithms.rst'),
                    header='``brainpy.algorithms`` module',
                    submodule_names=[k[0] for k in module_and_name],
                    section_names=[k[1] for k in module_and_name])
