# -*- coding: utf-8 -*-

import importlib
import inspect
import os

__all__ = [
  'write_module',
  'write_submodules',
]


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
    classes, functions = get_class_funcs(module)

    fout.write(section_names[i] + '\n')
    fout.write('-' * len(section_names[i]) + '\n\n')

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
