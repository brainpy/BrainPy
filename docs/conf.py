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

# keep_files = {'highlight_test_lexer.py', 'conf.py', 'make.bat', 'Makefile'}
# for item in os.listdir('../docs'):
#     if item not in keep_files:
#         path = os.path.join('../docs', item)
#         try:
#             if os.path.isfile(path):
#                 os.remove(path)
#             elif os.path.isdir(path):
#                 shutil.rmtree(path)
#         except Exception as e:
#             print(f"Error deleting {item}: {e}")
#
# build_version = os.environ.get('CURRENT_VERSION', 'v2')
# if build_version == 'v2':
#     shutil.copytree(
#         os.path.join(os.path.dirname(__file__), ''),
#         os.path.join(os.path.dirname(__file__)),
#         dirs_exist_ok=True
#     )
# else:
#     shutil.copytree(
#         os.path.join(os.path.dirname(__file__), '../docs_state'),
#         os.path.join(os.path.dirname(__file__)),
#         dirs_exist_ok=True
#     )

sys.path.insert(0, os.path.abspath('../docs/'))
sys.path.insert(0, os.path.abspath('../'))
shutil.copytree('../images/', './_static/logos/', dirs_exist_ok=True)
shutil.copyfile('../changelog.md', './changelog.md')

# -- Project information -----------------------------------------------------

project = 'BrainPy'
copyright = '2020-, BrainPy'
author = 'BrainPy Team'

from highlight_test_lexer import fix_ipython2_lexer_in_notebooks

fix_ipython2_lexer_in_notebooks(os.path.dirname(os.path.abspath(__file__)))

import brainpy

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
    'sphinx_design',
    'sphinx_math_dollar',
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
