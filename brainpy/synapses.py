# -*- coding: utf-8 -*-
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
This module has been deprecated since brainpy>=2.4.0. Use ``brainpy.dyn`` module instead.
"""

from brainpy.dyn.synapses.delay_couplings import (
    DiffusiveCoupling,
    AdditiveCoupling,
)
from brainpy.dynold.synapses.abstract_models import (
    Delta as Delta,
    Exponential as Exponential,
    DualExponential as DualExponential,
    Alpha as Alpha,
    NMDA as NMDA,
)
from brainpy.dynold.synapses.base import (
    _SynSTP as SynSTP,
    _SynOut as SynOut,
    TwoEndConn as TwoEndConn,
)
from brainpy.dynold.synapses.biological_models import (
    AMPA as AMPA,
    GABAa as GABAa,
    BioNMDA as BioNMDA,
)
from brainpy.dynold.synapses.compat import (
    DeltaSynapse as DeltaSynapse,
    ExpCUBA as ExpCUBA,
    ExpCOBA as ExpCOBA,
    DualExpCUBA as DualExpCUBA,
    DualExpCOBA as DualExpCOBA,
    AlphaCUBA as AlphaCUBA,
    AlphaCOBA as AlphaCOBA,
)
from brainpy.dynold.synapses.gap_junction import (
    GapJunction
)
from brainpy.dynold.synapses.learning_rules import (
    STP as STP,
)

if __name__ == '__main__':
    SynSTP
    SynOut
    TwoEndConn
    AMPA
    GABAa
    BioNMDA
    Delta
    Exponential
    DualExponential
    Alpha
    NMDA
    DeltaSynapse
    ExpCUBA
    ExpCOBA
    DualExpCUBA
    DualExpCOBA
    AlphaCUBA
    AlphaCOBA
    STP
    DiffusiveCoupling
    AdditiveCoupling
    GapJunction
