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


class Test_Conv(parameterized.TestCase):
    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10), ]
    )
    def test_Conv1d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 50, 3)
        layer = bp.dnn.Conv1d(in_channels=3,
                              out_channels=4,
                              kernel_size=5,
                              mode=mode)
        output = layer(input)

    def test_Conv1d_NonBatching(self):
        bm.random.seed()
        input = bm.random.randn(50, 3)
        layer = bp.dnn.Conv1d(in_channels=3,
                              out_channels=4,
                              kernel_size=5,
                              mode=bm.NonBatchingMode())
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10), ]
    )
    def test_Conv2d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 50, 50, 3)
        layer = bp.dnn.Conv2d(in_channels=3,
                              out_channels=4,
                              kernel_size=(5, 5),
                              mode=mode)
        output = layer(input)

    def test_Conv2_NonBatching(self):
        bm.random.seed()
        input = bm.random.randn(10, 10, 3)
        layer = bp.dnn.Conv2d(in_channels=3,
                              out_channels=4,
                              kernel_size=(5, 5),
                              mode=bm.NonBatchingMode())
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10), ]
    )
    def test_Conv3d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 50, 50, 50, 3)
        layer = bp.dnn.Conv3d(in_channels=3,
                              out_channels=4,
                              kernel_size=(5, 5, 5),
                              mode=mode)
        output = layer(input)

    def test_Conv3_NonBatching(self):
        bm.random.seed()
        input = bm.random.randn(10, 10, 10, 3)
        layer = bp.dnn.Conv3d(in_channels=3,
                              out_channels=4,
                              kernel_size=(5, 5, 5),
                              mode=bm.NonBatchingMode())
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10), ]
    )
    def test_ConvTranspose1d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 50, 3)
        layer = bp.dnn.ConvTranspose1d(in_channels=3,
                                       out_channels=4,
                                       kernel_size=5,
                                       mode=mode
                                       )
        output = layer(input)

    def test_ConvTranspose1d_NonBatching(self):
        bm.random.seed()
        input = bm.random.randn(10, 3)
        layer = bp.dnn.ConvTranspose1d(in_channels=3,
                                       out_channels=4,
                                       kernel_size=5,
                                       mode=bm.NonBatchingMode())
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10), ]
    )
    def test_ConvTranspose2d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 50, 50, 3)
        layer = bp.dnn.ConvTranspose2d(in_channels=3,
                                       out_channels=4,
                                       kernel_size=(5, 5),
                                       mode=mode
                                       )
        output = layer(input)

    def test_ConvTranspose2d_NonBatching(self):
        bm.random.seed()
        input = bm.random.randn(10, 10, 3)
        layer = bp.dnn.ConvTranspose2d(in_channels=3,
                                       out_channels=4,
                                       kernel_size=(5, 5),
                                       mode=bm.NonBatchingMode())
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10), ]
    )
    def test_ConvTranspose3d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 50, 50, 50, 3)
        layer = bp.dnn.ConvTranspose3d(in_channels=3,
                                       out_channels=4,
                                       kernel_size=(5, 5, 5),
                                       mode=mode
                                       )
        output = layer(input)

    def test_ConvTranspose3d_NonBatching(self):
        bm.random.seed()
        input = bm.random.randn(10, 10, 10, 3)
        layer = bp.dnn.ConvTranspose3d(in_channels=3,
                                       out_channels=4,
                                       kernel_size=(5, 5, 5),
                                       mode=bm.NonBatchingMode())
        output = layer(input)


class TestPool(parameterized.TestCase):

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_MaxPool(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 5, 5, 4)
        layer = bp.dnn.MaxPool(kernel_size=(3, 3),
                               channel_axis=-1,
                               mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_MinPool(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 5, 5, 4)
        layer = bp.dnn.MaxPool(kernel_size=(3, 3),
                               channel_axis=-1,
                               mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_AvgPool(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 5, 5, 4)
        layer = bp.dnn.AvgPool(kernel_size=(3, 3),
                               channel_axis=-1,
                               mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_AvgPool1d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 5, 4)
        layer = bp.dnn.AvgPool1d(kernel_size=3,
                                 channel_axis=-1,
                                 mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_AvgPool2d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 5, 5, 4)
        layer = bp.dnn.AvgPool2d(kernel_size=(3, 3),
                                 channel_axis=-1,
                                 mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_AvgPool3d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 5, 5, 5, 4)
        layer = bp.dnn.AvgPool3d(kernel_size=(3, 3, 3),
                                 channel_axis=-1,
                                 mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_MaxPool1d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 5, 4)
        layer = bp.dnn.MaxPool1d(kernel_size=3,
                                 channel_axis=-1,
                                 mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_MaxPool2d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 5, 5, 4)
        layer = bp.dnn.MaxPool2d(kernel_size=(3, 3),
                                 channel_axis=-1,
                                 mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_MaxPool3d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 5, 5, 5, 4)
        layer = bp.dnn.MaxPool3d(kernel_size=(3, 3, 3),
                                 channel_axis=-1,
                                 mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_AdaptiveAvgPool1d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 5, 4)
        layer = bp.dnn.AdaptiveAvgPool1d(target_shape=3,
                                         channel_axis=-1,
                                         mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_AdaptiveAvgPool2d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 5, 5, 4)
        layer = bp.dnn.AdaptiveAvgPool2d(target_shape=(3, 3),
                                         channel_axis=-1,
                                         mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_AdaptiveAvgPool3d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 5, 5, 5, 4)
        layer = bp.dnn.AdaptiveAvgPool3d(target_shape=(3, 3, 3),
                                         channel_axis=-1,
                                         mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_AdaptiveMaxPool1d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 5, 4)
        layer = bp.dnn.AdaptiveMaxPool1d(target_shape=3,
                                         channel_axis=-1,
                                         mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_AdaptiveMaxPool2d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 5, 5, 4)
        layer = bp.dnn.AdaptiveMaxPool2d(target_shape=(3, 3),
                                         channel_axis=-1,
                                         mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_AdaptiveMaxPool3d(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 5, 5, 5, 4)
        layer = bp.dnn.AdaptiveMaxPool3d(target_shape=(3, 3, 3),
                                         channel_axis=-1,
                                         mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)


class Test_Dropout(parameterized.TestCase):
    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_Dropout(self, mode):
        bp.share.save(fit=False)
        bm.random.seed()
        input = bm.random.randn(10, 5, 5, 5, 4)
        layer = bp.dnn.Dropout(prob=0.2,
                               mode=mode)
        output = layer(input)


class Test_function(parameterized.TestCase):
    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_Flatten(self, mode):
        bm.random.seed()
        layer = bp.dnn.Flatten(mode=mode)
        input = bm.random.randn(10, 5, 5, 5, 4)
        output = layer(input)


class Test_linear(parameterized.TestCase):

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_linear(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 9, 8, 7)
        layer = bp.dnn.Linear(num_in=7,
                              num_out=6,
                              mode=mode)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_AllToAll(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 10)
        layer = bp.dnn.AllToAll(num_pre=10,
                                num_post=20,
                                weight=0.1,
                                mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_OneToOne(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 10)
        layer = bp.dnn.OneToOne(num=10,
                                weight=0.1,
                                mode=mode)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_MaskedLinear(self, mode):
        bm.random.seed()
        input = bm.random.randn(100, 100)
        layer = bp.dnn.MaskedLinear(conn=bp.conn.FixedProb(0.1, pre=100, post=100),
                                    weight=0.1,
                                    mode=mode)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_CSRLinear(self, mode):
        bm.random.seed()
        input = bm.random.randn(100, 100)
        layer = bp.dnn.CSRLinear(conn=bp.conn.FixedProb(0.1, pre=100, post=100),
                                 weight=0.1,
                                 mode=mode)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_EventCSRLinear(self, mode):
        bm.random.seed()
        input = bm.random.randn(100, 100)
        layer = bp.dnn.EventCSRLinear(conn=bp.conn.FixedProb(0.1, pre=100, post=100),
                                      weight=0.1,
                                      mode=mode)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_JitFPHomoLinear(self, mode):
        bm.random.seed()
        layer = bp.dnn.JitFPHomoLinear(num_in=100,
                                       num_out=200,
                                       prob=0.1,
                                       weight=0.01,
                                       seed=100,
                                       mode=mode)
        input = bm.random.randn(10, 100)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_JitFPUniformLinear(self, mode):
        bm.random.seed()
        layer = bp.dnn.JitFPUniformLinear(num_in=100,
                                          num_out=200,
                                          prob=0.1,
                                          w_low=-0.01,
                                          w_high=0.01,
                                          seed=100,
                                          mode=mode)
        input = bm.random.randn(10, 100)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_JitFPNormalLinear(self, mode):
        bm.random.seed()
        layer = bp.dnn.JitFPNormalLinear(num_in=100,
                                         num_out=200,
                                         prob=0.1,
                                         w_mu=-0.01,
                                         w_sigma=0.01,
                                         seed=100,
                                         mode=mode)
        input = bm.random.randn(10, 100)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_EventJitFPHomoLinear(self, mode):
        bm.random.seed()
        layer = bp.dnn.EventJitFPHomoLinear(num_in=100,
                                            num_out=200,
                                            prob=0.1,
                                            weight=0.01,
                                            seed=100,
                                            mode=mode)
        input = bm.random.randn(10, 100)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_EventJitFPNormalLinear(self, mode):
        bm.random.seed()
        layer = bp.dnn.EventJitFPNormalLinear(num_in=100,
                                              num_out=200,
                                              prob=0.1,
                                              w_mu=-0.01,
                                              w_sigma=0.01,
                                              seed=100,
                                              mode=mode)
        input = bm.random.randn(10, 100)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_EventJitFPUniformLinear(self, mode):
        bm.random.seed()
        layer = bp.dnn.EventJitFPUniformLinear(num_in=100,
                                               num_out=200,
                                               prob=0.1,
                                               w_low=-0.01,
                                               w_high=0.01,
                                               seed=100,
                                               mode=mode)
        input = bm.random.randn(10, 100)
        output = layer(input)


class Test_Normalization(parameterized.TestCase):

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10)],
        fit=[True, False]
    )
    def test_BatchNorm1d(self, fit, mode):
        bm.random.seed()
        bp.share.save(fit=fit)
        layer = bp.dnn.BatchNorm1d(num_features=100,
                                   mode=mode,
                                   affine=False)
        input = bm.random.randn(10, 5, 100)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10)],
        fit=[True, False]
    )
    def test_BatchNorm2d(self, fit, mode):
        bm.random.seed()
        bp.share.save(fit=fit)
        layer = bp.dnn.BatchNorm2d(num_features=100,
                                   mode=mode,
                                   affine=False)
        input = bm.random.randn(10, 5, 6, 100)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10)],
        fit=[True, False]
    )
    def test_BatchNorm3d(self, fit, mode):
        bm.random.seed()
        bp.share.save(fit=fit)
        layer = bp.dnn.BatchNorm3d(num_features=100,
                                   mode=mode,
                                   affine=False)
        input = bm.random.randn(10, 5, 6, 7, 100)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()],
    )
    def test_LayerNorm(self, mode):
        bm.random.seed()
        layer = bp.dnn.LayerNorm(normalized_shape=3,
                                 mode=mode,
                                 elementwise_affine=False
                                 )
        input = bm.random.randn(10, 5, 3)
        outout = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()],
    )
    def test_GroupNorm(self, mode):
        bm.random.seed()
        layer = bp.dnn.GroupNorm(num_groups=2,
                                 num_channels=6,
                                 affine=False,
                                 mode=mode
                                 )
        input = bm.random.randn(20, 10, 10, 6)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()],
    )
    def test_InstanceNorm(self, mode):
        bm.random.seed()
        layer = bp.dnn.InstanceNorm(num_channels=6,
                                    affine=False,
                                    mode=mode
                                    )
        input = bm.random.randn(20, 10, 10, 6)
        output = layer(input)


if __name__ == '__main__':
    absltest.main()
