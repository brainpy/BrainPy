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
    fout.write('   :template: classtemplate.rst\n')
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
    fout.write('   :toctree: generated/\n')
    fout.write('   :nosignatures:\n')
    fout.write('   :template: classtemplate.rst\n\n')
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
    fout.write('   :toctree: generated/\n')
    fout.write('   :nosignatures:\n')
    fout.write('   :template: classtemplate.rst\n\n')
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
    fout.write('   :toctree: generated/\n')
    fout.write('   :nosignatures:\n')
    fout.write('   :template: classtemplate.rst\n\n')
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
      fout.write('   :toctree: generated/\n')
      fout.write('   :nosignatures:\n')
      fout.write('   :template: classtemplate.rst\n\n')
      for m in functions:
        fout.write(f'   {m}\n')
      for m in classes:
        fout.write(f'   {m}\n')
      for m in others:
        fout.write(f'   {m}\n')
      fout.write(f'\n\n')

  fout.close()


def _write_subsections_v4(module_path,
                          filename,
                          subsections: dict,
                          header: str = None):
  fout = open(filename, 'w')
  header = f'``{module_path}`` module' if header is None else header
  fout.write(header + '\n')
  fout.write('=' * len(header) + '\n\n')

  fout.write('.. contents::' + '\n')
  fout.write('   :local:' + '\n')
  fout.write('   :depth: 1' + '\n\n')

  for name, (subheader, out_path) in subsections.items():

    module = importlib.import_module(f'{module_path}.{name}')
    classes, functions, others = get_class_funcs(module)

    fout.write(subheader + '\n')
    fout.write('-' * len(subheader) + '\n\n')

    fout.write(f'.. currentmodule:: {out_path} \n')
    fout.write(f'.. automodule:: {out_path} \n\n')


    fout.write('.. autosummary::\n')
    fout.write('   :toctree: generated/\n')
    fout.write('   :nosignatures:\n')
    fout.write('   :template: classtemplate.rst\n\n')
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
    '- Number of functions covered by ``brainpy.version2.math``: {}\n'.format(len(bm_funcs & np_funcs)),
    '- Number of functions unique in ``brainpy.version2.math``: {}\n'.format(len(bm_funcs - np_funcs)),
    '- Number of functions covered by ``jax.numpy``: {}\n'.format(len(jax_funcs & np_funcs)),
  ]
  return buf


def _section(header, numpy_mod, brainpy_mod, jax_mod, klass=None, is_jax=False):
  buf = [header, '-' * len(header), '', ]
  header2 = 'NumPy, brainpy.version2.math, jax.numpy'
  buf += _generate_comparison_rst(numpy_mod, brainpy_mod, jax_mod, klass=klass, header=header2, is_jax=is_jax)
  buf += ['']
  return buf


def generate_analysis_docs():
  _write_subsections(
    module_name='brainpy.version2.analysis',
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
    'brainpy.version2.connect',
    'brainpy.version2.connect',
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
    'brainpy.version2.channels',
    'brainpy.version2.channels',
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
  _write_module(module_name='brainpy.version2.encoding',
                filename='apis/auto/encoding.rst',
                header='``brainpy.version2.encoding`` module')


def generate_initialize_docs():
  _write_subsections_v2(
    'brainpy.version2.initialize',
    'brainpy.version2.initialize',
    'apis/auto/initialize.rst',
    subsections={
      'base': 'Basic Initialization Classes',
      'regular_inits': 'Regular Initializers',
      'random_inits': 'Random Initializers',
      'decay_inits': 'Decay Initializers',
    }
  )


def generate_inputs_docs():
  _write_module(module_name='brainpy.version2.inputs',
                filename='apis/auto/inputs.rst',
                header='``brainpy.version2.inputs`` module')


def generate_mixin_docs():
  _write_module(module_name='brainpy.version2.mixin',
                filename='apis/auto/mixin.rst',
                header='``brainpy.version2.mixin`` module')


def generate_dnn_docs():
  _write_subsections_v2(
    'brainpy.version2.dnn',
    'brainpy.version2.dnn',
    'apis/auto/dnn.rst',
    subsections={
      'activations': 'Non-linear Activations',
      'conv': 'Convolutional Layers',
      'linear': 'Dense Connection Layers',
      'normalization': 'Normalization Layers',
      'pooling': 'Pooling Layers',
      'interoperation_flax': 'Interoperation with Flax',
      'others': 'Other Layers',
    }
  )


def generate_dyn_docs():
  _write_subsections_v2(
    'brainpy.version2.dyn',
    'brainpy.version2.dyn',
    'apis/auto/dyn.rst',
    subsections={
      'base': 'Base Classes',
      'ions': 'Ion Dynamics',
      'channels': 'Ion Channel Dynamics',
      'neurons': 'Neuron Dynamics',
      'synapses': 'Synaptic Dynamics',
      'projections': 'Synaptic Projections',
      'others': 'Common Dynamical Models',
      'outs': 'Synaptic Output Models',
      'rates': 'Population Rate Models',
    }
  )


def generate_losses_docs():
  _write_subsections_v2(
    'brainpy.version2.losses',
    'brainpy.version2.losses',
    'apis/auto/losses.rst',
    subsections={
      'comparison': 'Comparison',
      'regularization': 'Regularization',
    }
  )


def generate_measure_docs():
  _write_module(module_name='brainpy.version2.measure',
                filename='apis/auto/measure.rst',
                header='``brainpy.version2.measure`` module')


def generate_neurons_docs():
  _write_subsections_v2(
    'brainpy.version2.neurons',
    'brainpy.version2.neurons',
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
    'brainpy.version2.optim',
    'brainpy.version2.optim',
    'apis/auto/optim.rst',
    subsections={
      'optimizer': 'Optimizers',
      'scheduler': 'Schedulers',
    }
  )


def generate_rates_docs():
  _write_module(module_name='brainpy.version2.rates',
                filename='apis/auto/rates.rst',
                header='``brainpy.version2.rates`` module')


def generate_running_docs():
  _write_module(module_name='brainpy.version2.running',
                filename='apis/auto/running.rst',
                header='``brainpy.version2.running`` module')


def generate_synapses_docs():
  _write_module(module_name='brainpy.version2.synapses',
                filename='apis/auto/synapses.rst',
                header='``brainpy.version2.synapses`` module')

  _write_module(module_name='brainpy.version2.synouts',
                filename='apis/auto/synouts.rst',
                header='``brainpy.version2.synouts`` module')

  _write_module(module_name='brainpy.version2.synplast',
                filename='apis/auto/synplast.rst',
                header='``brainpy.version2.synplast`` module')


def generate_brainpy_docs():
  _write_subsections(
    module_name='brainpy',
    filename='apis/auto/brainpy.version2.rst',
    subsections={
      'Numerical Differential Integration': ['Integrator',
                                             'JointEq',
                                             'IntegratorRunner',
                                             'odeint',
                                             'sdeint',
                                             'fdeint'],
      'Building Dynamical System': ['DynamicalSystem',
                                    'DynSysGroup',
                                    'Sequential',
                                    'Network',
                                    'Dynamic',
                                    'Projection',
                                    ],
      'Simulating Dynamical System': ['DSRunner'],
      'Training Dynamical System': ['DSTrainer',
                                    'BPTT',
                                    'BPFF',
                                    'OnlineTrainer',
                                    'ForceTrainer',
                                    'OfflineTrainer',
                                    'RidgeTrainer'],
      'Dynamical System Helpers': ['LoopOverTime'],
    }
  )


def generate_integrators_doc():
  _write_subsections_v3(
    'brainpy.version2.integrators',
    'brainpy.version2.integrators',
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
  _write_subsections_v4(
    'brainpy.version2.math',
    'apis/auto/math.rst',
    subsections={
      'object_base': ('Objects and Variables', 'brainpy.version2.math'),
      'object_transform': ('Object-oriented Transformations', 'brainpy.version2.math'),
      'environment': ('Environment Settings', 'brainpy.version2.math'),
      # 'compat_numpy': ('Dense Operators with NumPy Syntax', 'brainpy.version2.math'),
      # 'compat_pytorch': ('Dense Operators with PyTorch Syntax', 'brainpy.version2.math'),
      # 'compat_tensorflow': ('Dense Operators with TensorFlow Syntax', 'brainpy.version2.math'),
      'interoperability': ('Array Interoperability', 'brainpy.version2.math'),
      'pre_syn_post': ('Operators for Pre-Syn-Post Conversion', 'brainpy.version2.math'),
      'activations': ('Activation Functions', 'brainpy.version2.math'),
      'delayvars': ('Delay Variables', 'brainpy.version2.math'),
      'modes': ('Computing Modes', 'brainpy.version2.math'),
      'sparse': ('``brainpy.version2.math.sparse`` module: Sparse Operators', 'brainpy.version2.math.sparse'),
      'event': ('``brainpy.version2.math.event`` module: Event-driven Operators', 'brainpy.version2.math.event'),
      'jitconn': ('``brainpy.version2.math.jitconn`` module: Just-In-Time Connectivity Operators', 'brainpy.version2.math.jitconn'),
      'surrogate': ('``brainpy.version2.math.surrogate`` module: Surrogate Gradient Functions', 'brainpy.version2.math.surrogate'),
      'random': ('``brainpy.version2.math.random`` module: Random Number Generations', 'brainpy.version2.math.random'),
      'linalg': ('``brainpy.version2.math.linalg`` module: Linear algebra', 'brainpy.version2.math.linalg'),
      'fft': ('``brainpy.version2.math.fft`` module: Discrete Fourier Transform', 'brainpy.version2.math.fft'),
    }
  )


def generate_algorithm_docs(path='apis/auto/algorithms/'):
  os.makedirs(path, exist_ok=True)

  module_and_name = [
    ('offline', 'Offline Training Algorithms'),
    ('online', 'Online Training Algorithms'),
    ('utils', 'Training Algorithm Utilities'),
  ]
  _write_submodules(module_name='brainpy.version2.algorithms',
                    filename=os.path.join(path, 'algorithms.rst'),
                    header='``brainpy.version2.algorithms`` module',
                    submodule_names=[k[0] for k in module_and_name],
                    section_names=[k[1] for k in module_and_name])
