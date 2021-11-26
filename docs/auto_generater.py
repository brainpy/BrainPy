# -*- coding: utf-8 -*-

import os
import inspect
import importlib
from brainpy.math.numpy import function
from brainpy.math import parallels, operators, activations, controls, autograd, losses, optimizers, compilation


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

  # write autoclass
  fout.write('\n')
  for m in classes:
    fout.write(f'.. autoclass:: {m}\n')
    fout.write(f'   :members:\n\n')

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

    # write autoclass
    fout.write('\n')
    for m in classes:
      fout.write(f'.. autoclass:: {m}\n')
      fout.write(f'   :members:\n\n')

  fout.close()


def generate_analysis_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  write_module(module_name='brainpy.analysis.symbolic',
               filename=os.path.join(path, 'symbolic.rst'),
               header='Dynamics Analysis (Symbolic)')

  write_module(module_name='brainpy.analysis.numeric',
               filename=os.path.join(path, 'numeric.rst'),
               header='Dynamics Analysis (Numeric)')

  write_module(module_name='brainpy.analysis.continuation',
               filename=os.path.join(path, 'continuation.rst'),
               header='Continuation Analysis')

  write_module(module_name='brainpy.analysis.stability',
               filename=os.path.join(path, 'stability.rst'),
               header='Stability Analysis')


def generate_base_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  write_submodules(module_name='brainpy.base',
                   filename=os.path.join(path, 'base.rst'),
                   header='``brainpy.base`` module',
                   submodule_names=['base', 'function', 'collector', 'io'],
                   section_names=['Base Class', 'Function Wrapper', 'Collectors', 'Exporting and Loading'])


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



block_list = ['test', 'control_transform', 'register_pytree_node']
for module in [compilation, autograd, function,
               controls, losses, activations, optimizers,
               operators, parallels]:
  for k in dir(module):
    if (not k.startswith('_') ) and (not inspect.ismodule(getattr(module, k))):
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


def _generate_comparison_rst(numpy_mod, brainpy_np, brainpy_jax, klass, header=', , '):
  np_obj, np_fmt = _import(numpy_mod, klass)
  np_funcs = _get_functions(np_obj)
  brainpy_np_obj, brainpy_np_fmt = _import(brainpy_np, klass)
  brainpy_np_funcs = _get_functions(brainpy_np_obj)
  brainpy_jax_obj, brainpy_jax_fmt = _import(brainpy_jax, klass)
  brainpy_jax_funcs = _get_functions(brainpy_jax_obj)

  buf = []
  buf += [
    '.. csv-table::',
    '   :header: {}'.format(header),
    '',
  ]
  for f in sorted(np_funcs):
    np_cell = np_fmt.format(f)
    brainpy_np_cell = brainpy_np_fmt.format(f) if f in brainpy_np_funcs else r'\-'
    brainpy_jax_cell = brainpy_jax_fmt.format(f) if f in brainpy_jax_funcs else r'\-'
    line = '   {}, {}, {}'.format(np_cell, brainpy_np_cell, brainpy_jax_cell)
    buf.append(line)

  unique_names = brainpy_np_funcs - np_funcs
  unique_names.update(brainpy_jax_funcs - np_funcs)
  for f in sorted(unique_names):
    np_cell = r'\-'
    brainpy_np_cell = brainpy_np_fmt.format(f) if f in brainpy_np_funcs else r'\-'
    brainpy_jax_cell = brainpy_jax_fmt.format(f) if f in brainpy_jax_funcs else r'\-'
    line = '   {}, {}, {}'.format(np_cell, brainpy_np_cell, brainpy_jax_cell)
    buf.append(line)

  buf += [
    '',
    '**Summary**\n',
    '- Number of NumPy functions: {}\n'.format(len(np_funcs)),
    '- Number of functions covered by ``brainpy.math.numpy``: {}\n'.format(len(brainpy_np_funcs & np_funcs)),
    '- Number of functions covered by ``brainpy.math.jax``: {}\n'.format(len(brainpy_jax_funcs & np_funcs)),
  ]
  return buf


def _section(header, numpy_mod, brainpy_np, brainpy_jax, klass=None):
  buf = [header, '-' * len(header), '', ]
  header2 = 'NumPy, brainpy.math.numpy, brainpy.math.jax'
  buf += _generate_comparison_rst(numpy_mod, brainpy_np, brainpy_jax, klass, header=header2)
  buf += ['']
  return buf


def generate_math_docs(path):
  if not os.path.exists(path): os.makedirs(path)

  buf = []
  buf += _section(header='Multi-dimensional Array',
                  numpy_mod='numpy',
                  brainpy_np='brainpy.math.numpy',
                  brainpy_jax='brainpy.math.jax',
                  klass='ndarray')
  buf += _section(header='Array Operations',
                  numpy_mod='numpy',
                  brainpy_np='brainpy.math.numpy',
                  brainpy_jax='brainpy.math.jax')
  buf += _section(header='Linear Algebra',
                  numpy_mod='numpy.linalg',
                  brainpy_np='brainpy.math.numpy.linalg',
                  brainpy_jax='brainpy.math.jax.linalg')
  buf += _section(header='Discrete Fourier Transform',
                  numpy_mod='numpy.fft',
                  brainpy_np='brainpy.math.numpy.fft',
                  brainpy_jax='brainpy.math.jax.fft')
  buf += _section(header='Random Sampling',
                  numpy_mod='numpy.random',
                  brainpy_np='brainpy.math.numpy.random',
                  brainpy_jax='brainpy.math.jax.random')
  codes = '\n'.join(buf)

  with open(os.path.join(path, 'comparison_table.rst.inc'), 'w') as f:
    f.write(codes)

  path = os.path.join(path, 'jax_math/')
  if not os.path.exists(path): os.makedirs(path)
  write_module(module_name='brainpy.math.jax.optimizers',
               filename=os.path.join(path, 'optimizers.rst'),
               header='Optimizers')
  write_module(module_name='brainpy.math.jax.losses',
               filename=os.path.join(path, 'losses.rst'),
               header='Loss Functions')
  write_module(module_name='brainpy.math.jax.activations',
               filename=os.path.join(path, 'activations.rst'),
               header='Activation Functions')
  write_module(module_name='brainpy.math.jax.autograd',
               filename=os.path.join(path, 'autograd.rst'),
               header='Automatic Differentiation')
  write_module(module_name='brainpy.math.jax.controls',
               filename=os.path.join(path, 'controls.rst'),
               header='Control Flows')
  write_module(module_name='brainpy.math.jax.operators',
               filename=os.path.join(path, 'operators.rst'),
               header='Operators')
  write_module(module_name='brainpy.math.jax.parallels',
               filename=os.path.join(path, 'parallels.rst'),
               header='Parallel Compilation')

def generate_simulation_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  # submodules in 'simulation' package
  write_module(module_name='brainpy.simulation.brainobjects',
               filename=os.path.join(path, 'brainobjects.rst'),
               header='Brain Objects')
  write_module(module_name='brainpy.simulation.layers',
               filename=os.path.join(path, 'layers.rst'),
               header='DNN Layers')
  module_and_name = [
    ('base', 'Base Class'),
    ('custom_conn', 'Custom Connections'),
    ('random_conn', 'Random Connections'),
    ('regular_conn', 'Regular Connections'),
    ('formatter', 'Formatter Functions'),
  ]
  write_submodules(module_name='brainpy.simulation.connect',
                   filename=os.path.join(path, 'connect.rst'),
                   header='Synaptic Connectivity',
                   submodule_names=[a[0] for a in module_and_name],
                   section_names=[a[1] for a in module_and_name])
  write_submodules(module_name='brainpy.simulation.initialize',
                   filename=os.path.join(path, 'initialize.rst'),
                   header='Weight Initialization',
                   submodule_names=['base', 'regular_inits', 'random_inits', 'decay_inits'],
                   section_names=['Base Class', 'Regular Initializers', 'Random Initializers', 'Decay Initializers'])

  # py-files in 'simulation' package
  write_module(module_name='brainpy.simulation.inputs',
               filename=os.path.join(path, 'inputs.rst'),
               header='Current Inputs')
  write_module(module_name='brainpy.simulation.measure',
               filename=os.path.join(path, 'measure.rst'),
               header='Measurements')
  write_module(module_name='brainpy.simulation.monitor',
               filename=os.path.join(path, 'monitor.rst'),
               header='Monitors')


def generate_tools_docs(path):
  if not os.path.exists(path):
    os.makedirs(path)

  module_and_name = [
      ('ast2code', 'AST-to-Code'),
      ('codes', 'Code Tools'),
      ('dicts', 'New Dict'),
      ('namechecking', 'Name Checking'),
      ('others', 'Other Tools'),
    ]
  modules = [k[0] for k in module_and_name]
  names = [k[1] for k in module_and_name]

  write_submodules(module_name='brainpy.tools',
                   filename=os.path.join(path, 'tools.rst'),
                   submodule_names=modules,
                   section_names=names)


def generate_visualization_docs(path):
  if not os.path.exists(path): os.makedirs(path)
  write_module(module_name='brainpy.visualization',
               filename=os.path.join(path, 'visualization.rst'))


def generate_jaxsetting_docs(path):
  if not os.path.exists(path): os.makedirs(path)
  write_module(module_name='brainpy.jaxsetting',
               filename=os.path.join(path, 'jaxsetting.rst'))
