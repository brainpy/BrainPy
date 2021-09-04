import importlib


block_list = ['test', 'pmap', 'vmap', 'jit', 'grad', 'value_and_grad',
              'control_transform', 'register_pytree_node']


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
    brainpy_np_cell = r'\-'
    if f in brainpy_np_funcs:
      brainpy_np_cell = brainpy_np_fmt.format(f)
    brainpy_jax_cell = r'\-'
    if f in brainpy_jax_funcs:
      brainpy_jax_cell = brainpy_jax_fmt.format(f)
    line = '   {}, {}, {}'.format(np_cell, brainpy_np_cell, brainpy_jax_cell)
    buf.append(line)

  buf += [
    '',
    '**Summary**\n',
    '- Number of NumPy functions: {}\n'.format(len(np_funcs)),
    '- Number of functions covered by ``brainpy.math.numpy``: {}\n'.format(len(brainpy_np_funcs & np_funcs)),
    '- Number of functions covered by ``brainpy.math.jax``: {}\n'.format(len(brainpy_jax_funcs & np_funcs)),
  ]
  buf += ['- ``brainpy.math.numpy`` specific functions:\n',]
  buf += [f'  - {i}. {f}' for i, f in enumerate(brainpy_np_funcs - np_funcs)]
  buf += ['']
  buf += ['- ``brainpy.math.jax`` specific functions:\n', ]
  buf += [f'  - {i}. {f}' for i, f in enumerate(brainpy_jax_funcs - np_funcs)]
  return buf


def _section(header, numpy_mod, brainpy_np, brainpy_jax, klass=None):
  buf = [header, '-' * len(header), '', ]
  header2 = 'NumPy, brainpy.math.numpy, brainpy.math.jax'
  buf += _generate_comparison_rst(numpy_mod, brainpy_np, brainpy_jax, klass, header=header2)
  buf += ['']
  return buf


def generate():
  buf = []

  buf += ['NumPy / JAX / BrainPy APIs',
          '--------------------------',
          '', ]
  buf = []
  buf += _section('Multi-dimensional Array', 'numpy', 'brainpy.math.numpy', 'brainpy.math.jax', klass='ndarray')
  buf += _section('Array Operations', 'numpy', 'brainpy.math.numpy', 'brainpy.math.jax')
  buf += _section('Linear Algebra', 'numpy.linalg', 'brainpy.math.numpy.linalg', 'brainpy.math.jax.linalg')
  buf += _section('Discrete Fourier Transform', 'numpy.fft', 'brainpy.math.numpy.fft', 'brainpy.math.jax.fft')
  buf += _section('Random Sampling', 'numpy.random', 'brainpy.math.numpy.random', 'brainpy.math.jax.random')

  return '\n'.join(buf)

# generate()
