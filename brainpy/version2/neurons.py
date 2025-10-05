# -*- coding: utf-8 -*-

"""
This module has been deprecated since brainpy>=2.4.0. Use ``brainpy.version2.dyn`` module instead.
"""

from brainpy.version2.dynold.neurons.biological_models import (
    HH as HH,
    MorrisLecar as MorrisLecar,
    PinskyRinzelModel as PinskyRinzelModel,
    WangBuzsakiModel as WangBuzsakiModel,
)

from brainpy.version2.dynold.neurons.fractional_models import (
    FractionalNeuron as FractionalNeuron,
    FractionalFHR as FractionalFHR,
    FractionalIzhikevich as FractionalIzhikevich,
)

from brainpy.version2.dynold.neurons.reduced_models import (
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
from brainpy.version2.dyn.others import (
    InputGroup as InputGroup,
    OutputGroup as OutputGroup,
    SpikeTimeGroup as SpikeTimeGroup,
    PoissonGroup as PoissonGroup,
    Leaky as Leaky,
    Integrator as Integrator,
    OUProcess as OUProcess,
)
