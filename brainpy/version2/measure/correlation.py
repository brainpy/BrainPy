# -*- coding: utf-8 -*-
# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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
import braintools

__all__ = [
    'cross_correlation',
    'voltage_fluctuation',
    'matrix_correlation',
    'weighted_correlation',
    'functional_connectivity',
    # 'functional_connectivity_dynamics',
]

cross_correlation = braintools.metric.cross_correlation
voltage_fluctuation = braintools.metric.voltage_fluctuation
matrix_correlation = braintools.metric.matrix_correlation
functional_connectivity = braintools.metric.functional_connectivity
weighted_correlation = braintools.metric.weighted_correlation
