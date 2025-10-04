# -*- coding: utf-8 -*-


"""
This module has been deprecated since brainpy>=2.4.0. Use ``brainpy.dyn`` module instead.
"""

from brainpy.version2.dynold.synapses.base import (
    _SynSTP as SynSTP,
    _SynOut as SynOut,
    TwoEndConn as TwoEndConn,
)
from brainpy.version2.dynold.synapses.biological_models import (
    AMPA as AMPA,
    GABAa as GABAa,
    BioNMDA as BioNMDA,
)
from brainpy.version2.dynold.synapses.abstract_models import (
    Delta as Delta,
    Exponential as Exponential,
    DualExponential as DualExponential,
    Alpha as Alpha,
    NMDA as NMDA,
)
from brainpy.version2.dynold.synapses.compat import (
    DeltaSynapse as DeltaSynapse,
    ExpCUBA as ExpCUBA,
    ExpCOBA as ExpCOBA,
    DualExpCUBA as DualExpCUBA,
    DualExpCOBA as DualExpCOBA,
    AlphaCUBA as AlphaCUBA,
    AlphaCOBA as AlphaCOBA,
)
from brainpy.version2.dynold.synapses.learning_rules import (
    STP as STP,
)
from brainpy.version2.dyn.synapses.delay_couplings import (
    DiffusiveCoupling,
    AdditiveCoupling,
)
from brainpy.version2.dynold.synapses.gap_junction import (
    GapJunction
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
