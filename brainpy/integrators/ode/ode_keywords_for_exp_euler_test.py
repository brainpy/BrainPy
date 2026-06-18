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
import unittest

import numpy as np
import pytest

from brainpy import _errors as errors
from brainpy import odeint


class TestExponentialEuler(unittest.TestCase):
    def test1(self):
        def func(m, t, V):
            alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
            beta = 4.0 * np.exp(-(V + 65) / 18)
            dmdt = alpha * (1 - m) - beta * m
            return dmdt

        odeint(method='exponential_euler', show_code=True, f=func)

    def test3(self):
        with pytest.raises(errors.CodeError):
            def func(m, t, dt):
                alpha = 0.1 * (dt + 40) / (1 - np.exp(-(dt + 40) / 10))
                beta = 4.0 * np.exp(-(dt + 65) / 18)
                dmdt = alpha * (1 - m) - beta * m
                return dmdt

            odeint(method='exponential_euler', show_code=True, f=func)
