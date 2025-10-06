# -*- coding: utf-8 -*-


"""
This module has been deprecated since brainpy>=2.4.0. Use ``brainpy.version2.dyn`` module instead.
"""

from brainpy.version2.dynold.synouts.conductances import (
    COBA as COBA,
    CUBA as CUBA,
)
from brainpy.version2.dynold.synouts.ions import (
    MgBlock as MgBlock,
)


if __name__ == '__main__':
    COBA, CUBA, MgBlock