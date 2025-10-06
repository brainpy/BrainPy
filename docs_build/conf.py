# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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

# Change to the docs directory before executing conf.py
original_dir = os.getcwd()
os.chdir(docs_path)

try:
    # Import all settings from the actual conf.py
    conf_file = os.path.join(docs_path, 'conf.py')
    with open(conf_file, 'r', encoding='utf-8') as f:
        exec(f.read(), globals())
finally:
    # Restore original directory
    os.chdir(original_dir)
