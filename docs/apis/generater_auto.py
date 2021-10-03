# -*- coding: utf-8 -*-

import importlib
import inspect
import os


__all__ = [
  'write_module',
  'write_submodules',
]


def write_module(module_name, filename, header=None):
  module = importlib.import_module(module_name)
  classes, functions = [], []
  for k in dir(module):
    data = getattr(module, k)
    if not k.startswith('__') and not inspect.ismodule(data):
      if inspect.isfunction(data):
        functions.append(k)
      elif isinstance(data, type):
        classes.append(k)

  fout = open(filename, 'w')
  # write_module header
  if header is None:
    header = f'``{module_name}`` module'
  else:
    header = header
  fout.write(header + '\n')
  fout.write('=' * len(header) + '\n\n')
  fout.write(f'.. currentmodule:: {module_name} \n')
  fout.write(f'.. automodule:: {module_name} \n\n')

  # write_module autosummary
  fout.write('.. autosummary::\n')
  fout.write('   :toctree: generated/\n\n')
  for m in functions:
    fout.write(f'   {m}\n')
  for m in classes:
    fout.write(f'   {m}\n')

  # write_module autoclass
  fout.write('\n')
  for m in classes:
    fout.write(f'.. autoclass:: {m}\n')
    fout.write(f'   :members:\n\n')

  fout.close()


def write_submodules(module_name, filename, header=None,
                     submodule_names=(), section_names=()):
  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

  fout = open(filename, 'w')
  # write_module header
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

    fout.write(section_names[i] + '\n')
    fout.write('-' * len(section_names[i]) + '\n\n')

    # functions and classes
    classes, functions = [], []
    for k in dir(module):
      data = getattr(module, k)
      if not k.startswith('__') and not inspect.ismodule(data):
        if inspect.isfunction(data):
          functions.append(k)
        elif isinstance(data, type):
          classes.append(k)

    # write_module autosummary
    fout.write('.. autosummary::\n')
    fout.write('   :toctree: generated/\n\n')
    for m in functions:
      fout.write(f'   {m}\n')
    for m in classes:
      fout.write(f'   {m}\n')

    # write_module autoclass
    fout.write('\n')
    for m in classes:
      fout.write(f'.. autoclass:: {m}\n')
      fout.write(f'   :members:\n\n')

  fout.close()

