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
from typing import Optional

import brainpy.math as bm

__all__ = [
    'get_spk_type',
]


def get_spk_type(spk_type: Optional[type] = None, mode: Optional[bm.Mode] = None):
    if mode is None:
        return bm.bool
    elif isinstance(mode, bm.TrainingMode):
        return bm.float_ if (spk_type is None) else spk_type
    else:
        assert isinstance(mode, bm.Mode)
        return bm.bool if (spk_type is None) else spk_type
