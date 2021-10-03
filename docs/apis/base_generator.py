# -*- coding: utf-8 -*-

import os
from docs.apis.generater_auto import *


def generate(path):
  if not os.path.exists(path):
    os.makedirs(path)

  # py-files in 'integrators' package
  write_module(module_name='brainpy.base',
               filename=os.path.join(path, 'base.rst'))
