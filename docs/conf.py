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
import brainpy

from docs import auto_generater
auto_generater.generate_base_docs('apis/')
auto_generater.generate_math_docs('apis/math/')
auto_generater.generate_integrators_doc('apis/integrators/')
auto_generater.generate_simulation_docs('apis/simulation/')
auto_generater.generate_training_docs('apis/training/')
auto_generater.generate_analysis_docs('apis/analysis/')
auto_generater.generate_visualization_docs('apis/')
auto_generater.generate_tools_docs('apis/')


import shutil
det_changelog = 'apis/changelog.rst'
src_changelog = '../changelog.rst'
if os.path.exists(det_changelog): os.remove(det_changelog)
shutil.copyfile(src_changelog, det_changelog)


# -- Project information -----------------------------------------------------

project = 'BrainPy'
copyright = '2021, BrainPy'
author = 'BrainPy Team'

# The full version, including alpha/beta/rc tags
release = brainpy.__version__


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
    "sphinx_rtd_theme",
    'sphinx_autodoc_typehints',
    'myst_nb',
    'matplotlib.sphinxext.plot_directive',
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


# -- Options for myst ----------------------------------------------
# Notebook cell execution timeout; defaults to 30.
execution_timeout = 200
jupyter_execute_notebooks = "off"
