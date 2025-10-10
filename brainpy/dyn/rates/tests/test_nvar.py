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
from absl.testing import absltest
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm


class Test_NVAR(parameterized.TestCase):
    @parameterized.product(
        mode=[bm.BatchingMode(),
              bm.NonBatchingMode()]
    )
    def test_NVAR(self, mode):
        bm.random.seed()
        input = bm.random.randn(1, 5)
        layer = bp.dyn.NVAR(num_in=5,
                            delay=10,
                            mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)


if __name__ == '__main__':
    absltest.main()
