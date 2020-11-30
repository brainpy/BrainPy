# -*- coding: utf-8 -*-

import io
import os
import re

from setuptools import find_packages
from setuptools import setup


# obtain version string from __init__.py
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'brainpy', '__init__.py'), 'r') as f:
    init_py = f.read()
version = re.search('__version__ = "(.*)"', init_py).groups()[0]

# obtain long description from README and CHANGES
with io.open(os.path.join(here, 'README.rst'), 'r', encoding='utf-8') as f:
    README = f.read()

# setup
setup(
    name='brain.py',
    version=version,
    description='BrainPy: A Just-In-Time compilation approach for neuronal dynamics simulation.',
    long_description=README,
    author='Chaoming Wang',
    author_email='adaduo@outlook.com',
    packages=find_packages(exclude=('examples',
                                    'examples.*',
                                    'docs',
                                    'docs.*',
                                    'develop',
                                    'develop.*',
                                    'tests',
                                    'tests.*')),
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.15',
        'matplotlib>=3.0',
        'sympy>=1.2',
        'autopep8',
    ],
    url='https://github.com/PKU-NIP-Lab/BrainPy',
    keywords='computational neuroscience',
    classifiers=[
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Bio-Informatics'
      ]
)
