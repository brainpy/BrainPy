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

r"""
State-based brain dynamics models built on ``brainstate``.

This module provides composable neuron, synapse, projection, readout, and
input models with explicit ``State`` objects, so they compose with
``brainstate`` transformations (``jit``, ``grad``, ``vmap``, ``for_loop``).
"""

from brainpy.state._base import Dynamics, Neuron, Synapse
from brainpy.state.exponential import Expon, DualExpon
from brainpy.state.hh import HH, MorrisLecar, WangBuzsakiHH
from brainpy.state.inputs import SpikeTime, PoissonSpike, PoissonEncoder, PoissonInput, poisson_input
from brainpy.state.inputs import (SectionInput, ConstantInput, StepInput, RampInput,
                                  SinusoidalInput, OUProcessInput, WienerProcessInput)
from brainpy.state.izhikevich import Izhikevich, IzhikevichRef
from brainpy.state.lif import (
    IF, LIF, LIFRef, ALIF,
    ExpIF, ExpIFRef,
    AdExIF, AdExIFRef,
    QuaIF, AdQuaIF, AdQuaIFRef,
    Gif, GifRef,
    CubaLIF, CobaLIF,
)
from brainpy.state.projection import (
    Projection, AlignPostProj, DeltaProj, CurrentProj,
    align_pre_projection, align_post_projection,
)
from brainpy.state.readout import LeakyRateReadout
from brainpy.state.reduced import FitzHughNagumo, HindmarshRose
from brainpy.state.stp import STP, STD
from brainpy.state.synapse import Alpha, AMPA, GABAa, BioNMDA
from brainpy.state.synaptic_projection import SymmetryGapJunction, AsymmetryGapJunction
from brainpy.state.synouts import SynOut, COBA, CUBA, MgBlock

__all__ = [
    # Base models
    'Dynamics', 'Neuron', 'Synapse',

    # Neuron models - LIF family
    'IF', 'LIF', 'LIFRef', 'ALIF',
    'ExpIF', 'ExpIFRef',
    'AdExIF', 'AdExIFRef',
    'QuaIF', 'AdQuaIF', 'AdQuaIFRef',
    'Gif', 'GifRef',
    'CubaLIF', 'CobaLIF',

    # Neuron models - Hodgkin-Huxley family
    'HH', 'MorrisLecar', 'WangBuzsakiHH',

    # Neuron models - Izhikevich family
    'Izhikevich', 'IzhikevichRef',

    # Neuron models - reduced (FitzHugh-Nagumo, Hindmarsh-Rose)
    'FitzHughNagumo', 'HindmarshRose',

    # Synapse models
    'Expon', 'DualExpon',
    'Alpha', 'AMPA', 'GABAa', 'BioNMDA',

    # Synaptic outputs
    'SynOut', 'COBA', 'CUBA', 'MgBlock',

    # Short-term plasticity
    'STP', 'STD',

    # Projections
    'Projection', 'AlignPostProj', 'DeltaProj', 'CurrentProj',
    'align_pre_projection', 'align_post_projection',
    'SymmetryGapJunction', 'AsymmetryGapJunction',

    # Readouts
    'LeakyRateReadout',

    # Input generators
    'SpikeTime', 'PoissonSpike', 'PoissonEncoder', 'PoissonInput', 'poisson_input',
    'SectionInput', 'ConstantInput', 'StepInput', 'RampInput',
    'SinusoidalInput', 'OUProcessInput', 'WienerProcessInput',
]
