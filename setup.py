# -*- coding: utf-8 -*-

import io
import os
import re

from setuptools import find_packages
from setuptools import setup

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
  name='brain-py',
  version=version,
  description='BrainPy: A flexible and extensible framework for brain modeling',
  long_description=README,
  long_description_content_type="text/markdown",
  author='Chaoming Wang',
  author_email='chao.brain@qq.com',
  packages=find_packages(exclude=['examples*', 'docs*', 'develop*', 'ode*']),
  python_requires='>=3.6',
  install_requires=[
    'numpy>=1.15',
    'matplotlib>=3.4',
  ],
  url='https://github.com/PKU-NIP-Lab/BrainPy',
  keywords='computational neuroscience, '
           'brain-inspired computation, '
           'dynamical systems, '
           'differential equations, '
           'numerical integration',
  classifiers=[
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries',
  ]
)
