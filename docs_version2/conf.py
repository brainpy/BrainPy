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
import shutil
import sys

sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('../'))

import brainpy

os.makedirs('apis/auto/', exist_ok=True)

# import auto_generater
# auto_generater.generate_analysis_docs()
# auto_generater.generate_connect_docs()
# auto_generater.generate_encoding_docs()
# auto_generater.generate_initialize_docs()
# auto_generater.generate_inputs_docs()
# auto_generater.generate_dnn_docs()
# auto_generater.generate_dyn_docs()
# auto_generater.generate_losses_docs()
# auto_generater.generate_measure_docs()
# auto_generater.generate_optim_docs()
# auto_generater.generate_running_docs()
# auto_generater.generate_brainpy_docs()
# auto_generater.generate_integrators_doc()
# auto_generater.generate_math_docs()
# auto_generater.generate_mixin_docs()
# sys.exit()

shutil.copytree('../images/', './_static/logos/', dirs_exist_ok=True)

# -- Project information -----------------------------------------------------

project = 'BrainPy'
copyright = '2020-, BrainPy'
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
    'sphinx_autodoc_typehints',
    'myst_nb',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_thebe',
    'sphinx_design'
    # 'sphinx-mathjax-offline',
]
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.ipynb', '.md']

# source_suffix = '.rst'
autosummary_generate = True

# The master toctree document.
master_doc = 'index'

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.8", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}
nitpick_ignore = [
    ("py:class", "docutils.nodes.document"),
    ("py:class", "docutils.parsers.rst.directives.body.Sidebar"),
]

suppress_warnings = ["myst.domains", "ref.ref"]
numfig = True
myst_enable_extensions = ["dollarmath", "amsmath", "deflist", "colon_fence"]
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# href with no underline and white bold text color
announcement = """
<a href="https://brainpy.readthedocs.io" style="text-decoration: none; color: white;">
  This site covers the old BrainPy 2.0 API. <span style="color: lightgray;">[Explore the new <b>BrainPy 3.0</b> API ✨]</span>
</a>
"""

html_theme_options = {
    'repository_url': 'https://github.com/brainpy/BrainPy',
    'use_repository_button': True,  # add a 'link to repository' button
    'use_issues_button': False,  # add an 'Open an Issue' button
    'path_to_docs': 'docs',  # used to compute the path to launch notebooks in colab
    'launch_buttons': {
        'colab_url': 'https://colab.research.google.com/',
    },
    'prev_next_buttons_location': None,
    'show_navbar_depth': 1,
    'announcement': announcement,
    'logo_only': True,
    'show_toc_level': 2,
}

html_theme = "sphinx_book_theme"
html_logo = "_static/logos/logo.png"
html_title = "BrainPy documentation"
html_copy_source = True
html_sourcelink_suffix = ""
html_favicon = "_static/logos/logo-square.png"
html_last_updated_fmt = ""
html_css_files = ['css/theme.css']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
jupyter_execute_notebooks = "off"
thebe_config = {
    "repository_url": "https://github.com/binder-examples/jupyter-stacks-datascience",
    "repository_branch": "master",
}

# -- Options for myst ----------------------------------------------
# Notebook cell execution timeout; defaults to 30.
execution_timeout = 200

autodoc_default_options = {
    'exclude-members': '....,default_rng',
}
