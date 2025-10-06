"""
Wrapper conf.py that dynamically loads the correct configuration
based on the READTHEDOCS_PROJECT environment variable.
"""
import os
import sys

# Determine which docs directory to use based on the project name
project_name = os.environ.get('PROJECT_VERSION', 'brainpy')

if project_name == 'brainpy-version2':
    docs_dir = 'docs_version2'
else:
    docs_dir = 'docs'

# Add the actual docs directory to the path
docs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', docs_dir))
sys.path.insert(0, docs_path)

exec(f'cd {docs_path}', globals())

# Import all settings from the actual conf.py
conf_file = os.path.join(docs_path, 'conf.py')
with open(conf_file, 'r', encoding='utf-8') as f:
    exec(f.read(), globals())
