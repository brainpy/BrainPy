import importlib
import inspect
import os

from brainpy.math.numpy import function
from brainpy.math.jax import controls, losses, activations, optimizers, gradient, compilation
from docs.apis.auto_generater import write_module


block_list = ['test', 'control_transform', 'register_pytree_node']
for module in [compilation, gradient, function,
               controls, losses, activations, optimizers]:
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


def generate(path):
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
  write_module(module_name='brainpy.math.jax.gradient',
               filename=os.path.join(path, 'gradient.rst'),
               header='Automatic Differentiation')


