# -*- coding: utf-8 -*-


"""
This module provides various running and training algorithms
for various neural networks.

The supported training algorithms include

- offline training methods, like ridge regression, linear regression, etc.
- online training methods, like recursive least squares (RLS, or Force Learning),
  least mean squares (LMS), etc.
- back-propagation learning method
- and others

The supported neural networks include

- reservoir computing networks,
- artificial recurrent neural networks,
- spiking neural networks,
- and others.
"""


from .base import *
from .back_propagation import *
from .online import *
from .offline import *

