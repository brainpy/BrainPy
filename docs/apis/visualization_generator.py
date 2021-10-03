# -*- coding: utf-8 -*-

import os

from docs.apis.generater_auto import *


def generate(path):
  if not os.path.exists(path): os.makedirs(path)
  write_module(module_name='brainpy.visualization',
               filename=os.path.join(path, 'visualization.rst'))
