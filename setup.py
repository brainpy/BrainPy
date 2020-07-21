import io
import os
import re

from setuptools import find_packages
from setuptools import setup


# obtain version string from __init__.py
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'npbrain', '__init__.py'), 'r') as f:
    init_py = f.read()
version = re.search('__version__ = "(.*)"', init_py).groups()[0]

# obtain long description from README and CHANGES
with io.open(os.path.join(here, 'README.rst'), 'r', encoding='utf-8') as f:
    README = f.read()

# setup
setup(
    name='npbrain',
    version=version,
    description='NumpyBrain: A lightweight SNN simulation framework.',
    long_description=README,
    author='Chaoming Wang',
    author_email='adaduo@outlook.com',
    packages=find_packages(),
    python_requires='>=3.5',
    install_requires=[
        'numpy>=1.15',
        'matplotlib',
        'numba',
    ],
    url='https://github.com/chaoming0625/NumpyBrain',
    keywords='computational neuroscience simulation',
    classifiers=[
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Bio-Informatics'
      ]
)
