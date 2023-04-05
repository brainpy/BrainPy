# -*- coding: utf-8 -*-

"""
Operators for brain dynamics modeling.
"""

from .check import *
from .errors import *
from .tools import *
from .register_custom_calls import *


from . import (
    event_ops,
    sparse_ops,
    jitconn_ops,
    op_registers,
    compat,
    pre_syn_post,
    sparse_matmul
)

__all__ = (
    event_ops.__all__
    + sparse_ops.__all__
    + jitconn_ops.__all__
    + op_registers.__all__
    + compat.__all__
    + pre_syn_post.__all__
    + sparse_matmul.__all__
)


from .event_ops import *
from .sparse_ops import *
from .jitconn_ops import *
from .op_registers import *
from .compat import *
from .sparse_matmul import *
from .pre_syn_post import *