# -*- coding: utf-8 -*-

import os

from docs.apis.generater_auto import *


def generate(path):
  if not os.path.exists(path):
    os.makedirs(path)

  # submodules in 'simulation' package
  write_module(module_name='brainpy.simulation.brainobjects',
               filename=os.path.join(path, 'brainobjects.rst'),
               header='Brain Objects')
  write_module(module_name='brainpy.simulation.layers',
               filename=os.path.join(path, 'layers.rst'),
               header='Brain Layers')
  write_module(module_name='brainpy.simulation.nets',
               filename=os.path.join(path, 'nets.rst'),
               header='Brain Networks')
  write_submodules(module_name='brainpy.simulation.connect',
                   filename=os.path.join(path, 'connect.rst'),
                   header='Synaptic Connectivity',
                   submodule_names=['base', 'custom_conn', 'random_conn', 'regular_conn', 'formatter'],
                   section_names=['Base Class', 'Custom Connections', 'Random Connections',
                                  'Regular Connections', 'Formatter Functions'])
  write_submodules(module_name='brainpy.simulation.initialize',
                   filename=os.path.join(path, 'initialize.rst'),
                   header='Weight Initialization',
                   submodule_names=['base', 'regular_inits', 'random_inits', 'decay_inits'],
                   section_names=['Base Class', 'Regular Initializers', 'Random Initializers', 'Decay Initializers'])

  # py-files in 'simulation' package
  write_module(module_name='brainpy.simulation.inputs',
               filename=os.path.join(path, 'inputs.rst'),
               header='Current Inputs')
  write_module(module_name='brainpy.simulation.losses',
               filename=os.path.join(path, 'losses.rst'),
               header='Loss Functions')
  write_module(module_name='brainpy.simulation.measure',
               filename=os.path.join(path, 'measure.rst'),
               header='Measurements')
  write_module(module_name='brainpy.simulation.monitor',
               filename=os.path.join(path, 'monitor.rst'),
               header='Monitors')
  write_module(module_name='brainpy.simulation.optimizers',
               filename=os.path.join(path, 'optimizers.rst'),
               header='Optimizers')
