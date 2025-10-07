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



build_version = os.environ.get('CURRENT_VERSION', 'v3')
if build_version == 'v2':
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), '../docs_version2'),
        os.path.join(os.path.dirname(__file__), ),
        dirs_exist_ok=True
    )
else:
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), '../docs_version3'),
        os.path.join(os.path.dirname(__file__), ),
    )

sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('../'))

import brainpy

shutil.copytree('../images/', './_static/logos/', dirs_exist_ok=True)
shutil.copyfile('../changelog.md', './changelog.md')
shutil.rmtree('./generated', ignore_errors=True)
shutil.rmtree('./_build', ignore_errors=True)

# -- Project information -----------------------------------------------------

project = 'BrainPy'
copyright = '2020-, BrainPy'
author = 'BrainPy Team'

from highlight_test_lexer import fix_ipython2_lexer_in_notebooks
fix_ipython2_lexer_in_notebooks(os.path.dirname(os.path.abspath(__file__)))

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

# href with no underline and white bold text color

if build_version == 'v2':
    announcement = """
    <a href="https://brainpy-v2.readthedocs.io" style="text-decoration: none; color: white;">
      This site covers the new BrainPy 3.0 API. 
      <span style="color: lightgray;">[Click here for the classical <b>BrainPy 2.0</b> API]</span>
    </a>
    """
else:
    announcement = """
    <a href="https://brainpy-v2.readthedocs.io" style="text-decoration: none; color: white;">
      This site covers the new BrainPy 3.0 API. 
      <span style="color: lightgray;">[Click here for the classical <b>BrainPy 2.0</b> API]</span>
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
