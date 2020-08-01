# -*- coding: utf-8 -*-

__version__ = "0.2.7.1"

# must import profile first
from npbrain.utils import profile

# module of "utils"
from npbrain.utils import connect
from npbrain.utils import helper
from npbrain.utils import input_factory
from npbrain.utils import measure
from npbrain.utils import visualize
from npbrain.utils import run
from npbrain.utils.helper import *
from npbrain.utils.helper import clip

# module of "core"
from npbrain import core

from npbrain.core import integrator
from npbrain.core.integrator import integrate

from npbrain.core import monitor
from npbrain.core.monitor import *

from npbrain.core import network
from npbrain.core.network import *

from npbrain.core import neuron
from npbrain.core.neuron import *
from npbrain.core.neuron import judge_spike

from npbrain.core import synapse
from npbrain.core.synapse import *


# module of "neurons"
from npbrain import neurons

from npbrain.neurons import HH_model
from npbrain.neurons.HH_model import *

from npbrain.neurons import input_model
from npbrain.neurons.input_model import *

from npbrain.neurons import LIF_model
from npbrain.neurons.LIF_model import *

from npbrain.neurons import Izhikevich_model
from npbrain.neurons.Izhikevich_model import *


# module of "synapse"
from npbrain import synapses

from npbrain.synapses import AMPA_synapses
from npbrain.synapses.AMPA_synapses import *

from npbrain.synapses import GABA_synapses
from npbrain.synapses.GABA_synapses import *

from npbrain.synapses import gap_junction
from npbrain.synapses.gap_junction import *

from npbrain.synapses import NMDA_synapses
from npbrain.synapses.NMDA_synapses import *

from npbrain.synapses import ordinary_synapses
from npbrain.synapses.ordinary_synapses import *

from npbrain.synapses import short_term_plasticity
from npbrain.synapses.short_term_plasticity import *


# reload functions
def _reload():
    global judge_spike
    global clip
    judge_spike = get_spike_judger()
    clip = get_clip()



