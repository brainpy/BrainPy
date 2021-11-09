# -*- coding: utf-8 -*-

import os
from docs.apis.auto_generater import *


def generate(path):
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
