# -*- coding: utf-8 -*-

import os
from docs.apis.generater_auto import *


def generate(path):
  if not os.path.exists(path):
    os.makedirs(path)

  write_submodules(module_name='brainpy.base',
                   filename=os.path.join(path, 'base.rst'),
                   header='``brainpy.base`` module',
                   submodule_names=['base', 'function', 'collector', 'io'],
                   section_names=['Base Class', 'Function Wrapper', 'Collectors', 'Exporting and Loading'])
