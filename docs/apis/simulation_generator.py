# -*- coding: utf-8 -*-

import os

from docs.apis.auto_generater import *


def generate(path):
  if not os.path.exists(path):
    os.makedirs(path)

  # submodules in 'simulation' package
  write_module(module_name='brainpy.simulation.brainobjects',
               filename=os.path.join(path, 'brainobjects.rst'),
               header='Brain Objects')
  write_module(module_name='brainpy.simulation.layers',
               filename=os.path.join(path, 'layers.rst'),
               header='DNN Layers')
  module_and_name = [
    ('base', 'Base Class'),
    ('custom_conn', 'Custom Connections'),
    ('random_conn', 'Random Connections'),
    ('regular_conn', 'Regular Connections'),
    ('formatter', 'Formatter Functions'),
  ]
  write_submodules(module_name='brainpy.simulation.connect',
                   filename=os.path.join(path, 'connect.rst'),
                   header='Synaptic Connectivity',
                   submodule_names=[a[0] for a in module_and_name],
                   section_names=[a[1] for a in module_and_name])
  write_submodules(module_name='brainpy.simulation.initialize',
                   filename=os.path.join(path, 'initialize.rst'),
                   header='Weight Initialization',
                   submodule_names=['base', 'regular_inits', 'random_inits', 'decay_inits'],
                   section_names=['Base Class', 'Regular Initializers', 'Random Initializers', 'Decay Initializers'])

  # py-files in 'simulation' package
  write_module(module_name='brainpy.simulation.inputs',
               filename=os.path.join(path, 'inputs.rst'),
               header='Current Inputs')
  write_module(module_name='brainpy.simulation.measure',
               filename=os.path.join(path, 'measure.rst'),
               header='Measurements')
  write_module(module_name='brainpy.simulation.monitor',
               filename=os.path.join(path, 'monitor.rst'),
               header='Monitors')

