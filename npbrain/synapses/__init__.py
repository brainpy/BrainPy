# -*- coding: utf-8 -*-

"""
In this module, we provide several examples for how various synapse models
are implemented in the ``NumpyBrain`` framework. For each synapse model,
three important functions should be implemented, which are ``update_state()``,
``output_synapse()`` and ``collect_spike()``.
"""


# AMPA_synapses.py
from npbrain.synapses.AMPA_synapses import *

# GABA_synapses.py
from npbrain.synapses.GABA_synapses import *

# gap_junction.py
from npbrain.synapses.gap_junction import *

# NMDA_synapses.py
from npbrain.synapses.NMDA_synapses import *

# ordinary_synapses.py
from npbrain.synapses.ordinary_synapses import *

# short_term_plasticity.py
from npbrain.synapses.short_term_plasticity import *


