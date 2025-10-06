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

__version__ = "3.0.0"
__version_info__ = (3, 0, 0)

from . import version2
from ._base import *
from ._base import __all__ as base_all
from ._errors import *
from ._errors import __all__ as errors_all
from ._exponential import *
from ._exponential import __all__ as exp_all
from ._inputs import *
from ._inputs import __all__ as inputs_all
from ._lif import *
from ._lif import __all__ as neuron_all
from ._projection import *
from ._projection import __all__ as proj_all
from ._readout import *
from ._readout import __all__ as readout_all
from ._stp import *
from ._stp import __all__ as stp_all
from ._synapse import *
from ._synapse import __all__ as synapse_all
from ._synaptic_projection import *
from ._synaptic_projection import __all__ as synproj_all
from ._synouts import *
from ._synouts import __all__ as synout_all

__main__ = ['version2'] + errors_all + inputs_all + neuron_all + readout_all + stp_all + synapse_all
__main__ = __main__ + synout_all + base_all + exp_all + proj_all + synproj_all
del errors_all, inputs_all, neuron_all, readout_all, stp_all, synapse_all, synout_all, base_all
del exp_all, proj_all, synproj_all


# Deprecation warnings for brainpy.xxx -> brainpy.version2.xxx
def __getattr__(name):
    """Provide deprecation warnings for moved modules."""
    import warnings

    if hasattr(version2, name):
        warnings.warn(
            f"Accessing 'brainpy.{name}' is deprecated. "
            f"Please use 'brainpy.version2.{name}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return getattr(version2, name)

    raise AttributeError(f"'brainpy' has no attribute '{name}'")


def __dir__():
    """Return list of attributes including deprecated ones for tab completion."""
    # Get the default attributes
    default_attrs = list(globals().keys())

    # Add all public attributes from version2 for discoverability
    version2_attrs = [attr for attr in dir(version2) if not attr.startswith('_')]

    # Combine and return unique attributes
    return sorted(set(default_attrs + version2_attrs))


# Register deprecated modules in sys.modules to support "import brainpy.xxx" syntax
import sys as _sys

_deprecated_modules = [
    'math', 'check', 'tools', 'connect', 'initialize', 'init', 'conn',
    'optim', 'losses', 'measure', 'inputs', 'encoding', 'checkpoints',
    'mixin', 'algorithms', 'integrators', 'ode', 'sde', 'fde',
    'dnn', 'layers', 'dyn', 'running', 'train', 'analysis',
    'channels', 'neurons', 'rates', 'synapses', 'synouts', 'synplast',
    'visualization', 'visualize', 'types', 'modes', 'context',
    'helpers', 'delay', 'dynsys', 'runners', 'transform', 'dynold'
]

# Create wrapper modules that show deprecation warnings
for _mod_name in _deprecated_modules:
    if hasattr(version2, _mod_name):
        _sys.modules[f'brainpy.{_mod_name}'] = getattr(version2, _mod_name)

del _sys, _mod_name, _deprecated_modules

version2.__version__ = __version__

