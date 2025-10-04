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

from ._base import *
from ._base import __all__ as base_all
from ._exponential import *
from ._exponential import __all__ as exp_all
from ._inputs import *
from ._inputs import __all__ as inputs_all
from ._lif import *
from ._lif import __all__ as neuron_all
from ._readout import *
from ._readout import __all__ as readout_all
from ._stp import *
from ._stp import __all__ as stp_all
from ._synapse import *
from ._synapse import __all__ as synapse_all
from ._synouts import *
from ._synouts import __all__ as synout_all
from ._errors import *
from ._errors import __all__ as errors_all

__main__ = errors_all + inputs_all + neuron_all + readout_all + stp_all + synapse_all + synout_all + base_all
__main__ = __main__ + exp_all
del errors_all, inputs_all, neuron_all, readout_all, stp_all, synapse_all, synout_all, base_all
del exp_all
