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

from brainpy.dyn.others import (
    InputGroup as InputGroup,
    OutputGroup as OutputGroup,
    SpikeTimeGroup as SpikeTimeGroup,
    PoissonGroup as PoissonGroup,
    Leaky as Leaky,
    Integrator as Integrator,
    OUProcess as OUProcess,
)
from brainpy.dynold.neurons.biological_models import (
    HH as HH,
    MorrisLecar as MorrisLecar,
    PinskyRinzelModel as PinskyRinzelModel,
    WangBuzsakiModel as WangBuzsakiModel,
)
from brainpy.dynold.neurons.fractional_models import (
    FractionalNeuron as FractionalNeuron,
    FractionalFHR as FractionalFHR,
    FractionalIzhikevich as FractionalIzhikevich,
)
from brainpy.dynold.neurons.reduced_models import (
    LeakyIntegrator as LeakyIntegrator,
    LIF as LIF,
    ExpIF as ExpIF,
    AdExIF as AdExIF,
    QuaIF as QuaIF,
    AdQuaIF as AdQuaIF,
    GIF as GIF,
    ALIFBellec2020 as ALIFBellec2020,
    Izhikevich as Izhikevich,
    HindmarshRose as HindmarshRose,
    FHN as FHN,
    LIF_SFA_Bellec2020,
)

if __name__ == '__main__':
    HH
    MorrisLecar
    PinskyRinzelModel
    WangBuzsakiModel

    FractionalNeuron
    FractionalFHR
    FractionalIzhikevich

    LeakyIntegrator
    LIF
    ExpIF
    AdExIF
    QuaIF
    AdQuaIF
    GIF
    ALIFBellec2020
    Izhikevich
    HindmarshRose
    FHN
    LIF_SFA_Bellec2020

    InputGroup
    OutputGroup
    SpikeTimeGroup
    PoissonGroup
    Leaky
    Integrator
    OUProcess
