# -*- coding: utf-8 -*-

import io
import os
import re

from setuptools import find_packages
from setuptools import setup

try:
  import pkg_resources
  installed_packages = pkg_resources.working_set
  for i in installed_packages:
    if i.key == 'brainpy-simulator':
      raise SystemError('Please uninstall the older version of brainpy '
                        f'package "brainpy-simulator={i.version}" '
                        f'(located in {i.location}) first. \n'
                        '>>> pip uninstall brainpy-simulator')
    if i.key == 'brain-py':
      raise SystemError('Please uninstall the older version of brainpy '
                        f'package "brain-py={i.version}" '
                        f'(located in {i.location}) first. \n'
                        '>>> pip uninstall brain-py')
except ModuleNotFoundError:
  pass


# version
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'brainpy', '__init__.py'), 'r') as f:
  init_py = f.read()
version = re.search('__version__ = "(.*)"', init_py).groups()[0]

# obtain long description from README
with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
  README = f.read()

# setup
setup(
  name='brainpy',
  version=version,
  description='BrainPy: Brain Dynamics Programming in Python',
  long_description=README,
  long_description_content_type="text/markdown",
  author='BrainPy Team',
  author_email='chao.brain@qq.com',
  packages=find_packages(),
  python_requires='>=3.7',
  install_requires=[
    'numpy>=1.15',
    'jax>=0.3.0',
    'tqdm',
  ],
  extras_require={
    'cpu': ['jaxlib>=0.3.0', 'brainpylib>=0.0.4'],
    'cuda': ['jaxlib>=0.3.0', 'brainpylib>=0.0.4'],
    'all': ['jaxlib>=0.3.0', 'brainpylib>=0.0.4',
            'numba>=0.50', 'scipy>=1.1.0',
            'networkx', 'matplotlib']
  },
  url='https://github.com/PKU-NIP-Lab/BrainPy',
  keywords='computational neuroscience, '
           'brain-inspired computation, '
           'dynamical systems, '
           'differential equations, '
           'brain modeling, '
           'brain dynamics programming',
  classifiers=[
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries',
  ],
  license='GPL-3.0 License',
)
