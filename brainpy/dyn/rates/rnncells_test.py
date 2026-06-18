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


class Test_Rnncells(parameterized.TestCase):
    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(20),
              bm.BatchingMode(),
              bm.BatchingMode(20)
              ]
    )
    def test_RNNCell(self, mode):
        bm.random.seed()
        input = bm.random.randn(20, 10)
        layer = bp.dyn.RNNCell(num_in=10,
                               num_out=64,
                               mode=mode
                               )
        output = layer(input)

    def test_RNNCell_NonBatching(self):
        bm.random.seed()
        input = bm.random.randn(10)
        layer = bp.dyn.RNNCell(num_in=10,
                               num_out=32,
                               mode=bm.NonBatchingMode())
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(50),
              bm.BatchingMode(),
              bm.BatchingMode(50)
              ]
    )
    def test_GRUCell(self, mode):
        bm.random.seed()
        input = bm.random.randn(50, 100)
        layer = bp.dyn.GRUCell(num_in=100,
                               num_out=64,
                               mode=mode)
        output = layer(input)

    def test_GRUCell_NonBatching(self):
        bm.random.seed()
        input = bm.random.randn(10)
        layer = bp.dyn.GRUCell(num_in=10,
                               num_out=12,
                               mode=bm.NonBatchingMode())
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(50),
              bm.BatchingMode(),
              bm.BatchingMode(50)
              ]
    )
    def test_LSTMCell(self, mode):
        bm.random.seed()
        input = bm.random.randn(50, 100)
        layer = bp.dyn.LSTMCell(num_in=100,
                                num_out=64,
                                mode=mode)

        output = layer(input)

    def test_LSTMCell_NonBatching(self):
        bm.random.seed()
        input = bm.random.randn(10)
        layer = bp.dyn.LSTMCell(num_in=10,
                                num_out=5,
                                mode=bm.NonBatchingMode())
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(4),
              bm.BatchingMode(),
              bm.BatchingMode(4)]
    )
    def test_Conv1dLSTMCell(self, mode):
        bm.random.seed()
        input = bm.random.randn(4, 100, 3)
        layer = bp.dyn.Conv1dLSTMCell(input_shape=(100,),
                                      in_channels=3,
                                      out_channels=5,
                                      kernel_size=4,
                                      mode=mode)
        output = layer(input)

    def test_Conv1dLSTMCell_NonBatching(self):
        bm.random.seed()
        input = bm.random.randn(10, 3)
        layer = bp.dyn.Conv1dLSTMCell(input_shape=(10,),
                                      in_channels=3,
                                      out_channels=4,
                                      kernel_size=5,
                                      mode=bm.NonBatchingMode())
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(4),
              bm.BatchingMode(),
              bm.BatchingMode(4)]
    )
    def test_Conv2dLSTMCell(self, mode):
        bm.random.seed()
        input = bm.random.randn(4, 100, 100, 3)
        layer = bp.dyn.Conv2dLSTMCell(input_shape=(100, 100),
                                      in_channels=3,
                                      out_channels=5,
                                      kernel_size=(4, 4),
                                      mode=mode)
        output = layer(input)

    def test_Conv2dLSTMCell_NonBatching(self):
        bm.random.seed()
        input = bm.random.randn(10, 10, 3)
        layer = bp.dyn.Conv2dLSTMCell(input_shape=(10, 10),
                                      in_channels=3,
                                      out_channels=4,
                                      kernel_size=5,
                                      mode=bm.NonBatchingMode())
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(4),
              bm.BatchingMode(),
              bm.BatchingMode(4)]
    )
    def test_Conv3dLSTMCell(self, mode):
        bm.random.seed()
        input = bm.random.randn(4, 100, 100, 100, 3)
        layer = bp.dyn.Conv3dLSTMCell(input_shape=(100, 100, 100),
                                      in_channels=3,
                                      out_channels=5,
                                      kernel_size=(4, 4, 4),
                                      mode=mode)
        output = layer(input)

    def test_Conv3dLSTMCell_NonBatching(self):
        bm.random.seed()
        input = bm.random.randn(10, 10, 10, 3)
        layer = bp.dyn.Conv3dLSTMCell(input_shape=(10, 10, 10),
                                      in_channels=3,
                                      out_channels=4,
                                      kernel_size=5,
                                      mode=bm.NonBatchingMode())
        output = layer(input)


class Test_Rnncells_reset_state(parameterized.TestCase):
    """Regression tests for P10-H1: ``reset_state(None)`` must not crash."""

    @parameterized.product(cls=['RNNCell', 'GRUCell', 'LSTMCell'])
    def test_rnn_cells_reset_state_none_unbatched(self, cls):
        # P10-H1: explicit reset_state(None) on a non-batching cell used to raise
        # ``ValueError: Do not support type <class 'NoneType'>: None`` because the
        # state shape was built as ``(None, num_out)``.
        bm.random.seed()
        cell = getattr(bp.dyn, cls)(num_in=3, num_out=4)
        cell.reset_state(None)
        expected = (8,) if cls == 'LSTMCell' else (4,)
        self.assertTupleEqual(tuple(cell.state.shape), expected)

    @parameterized.product(cls=['RNNCell', 'GRUCell', 'LSTMCell'])
    def test_rnn_cells_reset_state_via_bp_reset_state(self, cls):
        # P10-H1: the public ``bp.reset_state`` path passes ``batch_or_mode=None``.
        bm.random.seed()
        cell = getattr(bp.dyn, cls)(num_in=3, num_out=4)
        bp.reset_state(cell)
        expected = (8,) if cls == 'LSTMCell' else (4,)
        self.assertTupleEqual(tuple(cell.state.shape), expected)

    @parameterized.product(cls=['RNNCell', 'GRUCell', 'LSTMCell'])
    def test_rnn_cells_reset_state_int_batch(self, cls):
        # The int-batch path must still produce a leading batch axis.
        bm.random.seed()
        cell = getattr(bp.dyn, cls)(num_in=3, num_out=4, mode=bm.batching_mode)
        cell.reset_state(2)
        expected = (2, 8) if cls == 'LSTMCell' else (2, 4)
        self.assertTupleEqual(tuple(cell.state.shape), expected)


if __name__ == '__main__':
    absltest.main()
