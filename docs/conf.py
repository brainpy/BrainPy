# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# a_list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from shutil import copyfile

import brainpy.math.function
from docs.apis import base_generator
from docs.apis import math_generator
from docs.apis import integrators_generator
from docs.apis import simulation_generator

base_generator.generate('apis/')
math_generator.generate('apis/math/')
integrators_generator.generate('apis/integrators/')
simulation_generator.generate('apis/simulation/')


# -- Project information -----------------------------------------------------

project = 'BrainPy'
copyright = '2021, Chaoming Wang'
author = 'Chaoming Wang'

# The full version, including alpha/beta/rc tags
release = brainpy.__version__

det_changelog = 'apis/changelog.rst'
src_changelog = '../changelog.rst'

if os.path.exists(det_changelog):
    os.remove(det_changelog)
copyfile(src_changelog, det_changelog)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
    "sphinx_rtd_theme",
]
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# source_suffix = '.rst'
autosummary_generate = True

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
