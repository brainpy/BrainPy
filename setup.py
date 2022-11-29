# -*- coding: utf-8 -*-

import io
import os
import re
import sys

from setuptools import find_packages
from setuptools import setup

try:
  # require users to uninstall previous brainpy releases.
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

# installation packages
packages = find_packages()
if 'docs' in packages:
  packages.remove('docs')
if 'tests' in packages:
  packages.remove('tests')

# setup
setup(
  name='brainpy',
  version=version,
  description='BrainPy: Brain Dynamics Programming in Python',
  long_description=README,
  long_description_content_type="text/markdown",
  author='BrainPy Team',
  author_email='chao.brain@qq.com',
  packages=packages,
  python_requires='>=3.7',
  install_requires=['numpy>=1.15', 'jax>=0.3.0', 'tqdm', 'brainpylib'],
  url='https://github.com/PKU-NIP-Lab/BrainPy',
  project_urls={
    "Bug Tracker": "https://github.com/PKU-NIP-Lab/BrainPy/issues",
    "Documentation": "https://brainpy.readthedocs.io/",
    "Source Code": "https://github.com/PKU-NIP-Lab/BrainPy",
  },
  keywords=('computational neuroscience, '
            'brain-inspired computation, '
            'dynamical systems, '
            'differential equations, '
            'brain modeling, '
            'brain dynamics modeling, '
            'brain dynamics programming'),
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
    'License :: OSI Approved :: Apache Software License',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries',
  ],
  license='GPL-3.0 License',
)
