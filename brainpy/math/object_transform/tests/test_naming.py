# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

import brainpy as bp
import brainpy.math as bm


class TestNaming(unittest.TestCase):

    def test_clear_name_cache(self):
        lif = bp.dyn.LifRef(1, name='a')
        with self.assertRaises(bp.errors.UniqueNameError):
            lif = bp.dyn.LifRef(1, name='a')
        bm.clear_name_cache(ignore_warn=True)
        lif = bp.dyn.LifRef(1, name='a')
        bm.clear_name_cache()
        bm.clear_buffer_memory(array=False, compilation=True)
