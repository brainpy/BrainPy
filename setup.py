# -*- coding: utf-8 -*-

import io
import os
import re
import time
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
if len(sys.argv) > 2 and sys.argv[2] == '--python-tag=py3':
  version = version
else:
  version += '.post{}'.format(time.strftime("%Y%m%d", time.localtime()))

# obtain long description from README
with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
  README = f.read()

# installation packages
packages = find_packages(exclude=['lib*', 'docs', 'tests'])

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
  python_requires='>=3.9',
  install_requires=['numpy>=1.15', 'jax>=0.4.13', 'tqdm'],
  url='https://github.com/brainpy/BrainPy',
  project_urls={
    "Bug Tracker": "https://github.com/brainpy/BrainPy/issues",
    "Documentation": "https://brainpy.readthedocs.io/",
    "Source Code": "https://github.com/brainpy/BrainPy",
  },
  dependency_links=[
    'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
  ],
  extras_require={
    'cpu': ['jaxlib>=0.4.13', 'brainpylib', 'numba', 'braintaichi'],
    'cuda11': ['jaxlib[cuda11_pip]', 'brainpylib', 'numba', 'braintaichi'],
    'cuda12': ['jaxlib[cuda12_pip]', 'brainpylib', 'numba', 'braintaichi'],
    'tpu': ['jaxlib[tpu]', 'numba',],
    'cpu_mini': ['jaxlib>=0.4.13'],
    'cuda11_mini': ['jaxlib[cuda11_pip]'],
    'cuda12_mini': ['jaxlib[cuda12_pip]'],
  },
  keywords=('computational neuroscience, '
            'brain-inspired computation, '
            'brain modeling, '
            'brain dynamics modeling, '
            'brain dynamics programming'),
  classifiers=[
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries',
  ],
  license='GPL-3.0 license',
)
