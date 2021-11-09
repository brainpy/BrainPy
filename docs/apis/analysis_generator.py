# -*- coding: utf-8 -*-

import os

from docs.apis.auto_generater import *


def generate(path):
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
