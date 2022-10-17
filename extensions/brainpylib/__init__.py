# -*- coding: utf-8 -*-

__version__ = "0.0.7"

# IMPORTANT, must import first
from . import register_custom_calls

# import operators
from .event_sum import *
from .event_prod import *
from .atomic_sum import *
from .atomic_prod import *
from .custom_op import register_op
