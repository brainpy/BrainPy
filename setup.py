# -*- coding: utf-8 -*-

import io
import os
import re

from setuptools import find_packages, setup

# version
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'brainpy', '__init__.py'), 'r') as f:
    init_py = f.read()
version = re.search('__version__ = "(.*)"', init_py).groups()[0]

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
    python_requires='>=3.10',
    install_requires=['numpy>=1.15', 'jax', 'tqdm', 'brainstate>=0.1.6', 'brainunit', 'brainevent'],
    url='https://github.com/brainpy/BrainPy',
    project_urls={
        "Bug Tracker": "https://github.com/brainpy/BrainPy/issues",
        "Documentation": "https://brainpy.readthedocs.io/",
        "Source Code": "https://github.com/brainpy/BrainPy",
    },
    extras_require={
        'cpu': ['jax[cpu]', 'brainstate[cpu]', 'brainunit[cpu]', 'brainevent[cpu]'],
        'cuda12': ['jax[cuda12]', 'brainstate[cuda12]', 'brainunit[cuda12]', 'brainevent[cuda12]'],
        'tpu': ['jax[tpu]', 'brainstate[tpu]', 'brainunit[tpu]', 'brainevent[tpu]'],
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
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
    ],
    license='GPL-3.0 license',
)
