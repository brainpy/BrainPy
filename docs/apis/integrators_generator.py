# -*- coding: utf-8 -*-

import os
from docs.apis.generater_auto import *


def generate(path):
  if not os.path.exists(path):
    os.makedirs(path)

  # py-files in 'integrators' package
  write_module(module_name='brainpy.integrators.wrapper',
               filename=os.path.join(path, 'general.rst'),
               header='General Functions')
  write_module(module_name='brainpy.integrators.ode.explicit_rk',
               filename=os.path.join(path, 'ode_explicit_rk.rst'),
               header='Explicit Runge-Kutta Methods')
  write_module(module_name='brainpy.integrators.ode.adaptive_rk',
               filename=os.path.join(path, 'ode_adaptive_rk.rst'),
               header='Adaptive Runge-Kutta Methods')
  write_module(module_name='brainpy.integrators.ode.exponential',
               filename=os.path.join(path, 'ode_exponential.rst'),
               header='Exponential Integrators')

