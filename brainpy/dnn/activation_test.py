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


class Test_Activation(parameterized.TestCase):

    @parameterized.product(
        inplace=[True, False]
    )
    def test_Threshold(self, inplace):
        bm.random.seed()
        threshold_layer = bp.dnn.Threshold(5, 20, inplace)
        input = bm.random.randn(2)
        if inplace == True:
            threshold_layer(input)
        elif inplace == False:
            output = threshold_layer(input)

    @parameterized.product(
        inplace=[True, False]
    )
    def test_ReLU(self, inplace):
        bm.random.seed()
        ReLU_layer = bp.dnn.ReLU(inplace)
        input = bm.random.randn(2)
        if inplace == True:
            ReLU_layer(input)
        elif inplace == False:
            output = ReLU_layer(input)

    @parameterized.product(
        inplace=[True, False]
    )
    def test_RReLU(self, inplace):
        bm.random.seed()
        RReLU_layer = bp.dnn.RReLU(lower=0, upper=1, inplace=inplace)
        input = bm.random.randn(2)
        if inplace == True:
            RReLU_layer(input)
        elif inplace == False:
            output = RReLU_layer(input)

    @parameterized.product(
        inplace=[True, False]
    )
    def test_Hardtanh(self, inplace):
        bm.random.seed()
        Hardtanh_layer = bp.dnn.Hardtanh(min_val=0, max_val=1, inplace=inplace)
        input = bm.random.randn(2)
        if inplace == True:
            Hardtanh_layer(input)
        elif inplace == False:
            output = Hardtanh_layer(input)

    @parameterized.product(
        inplace=[True, False]
    )
    def test_ReLU6(self, inplace):
        bm.random.seed()
        ReLU6_layer = bp.dnn.ReLU6(inplace=inplace)
        input = bm.random.randn(2)
        if inplace == True:
            ReLU6_layer(input)
        elif inplace == False:
            output = ReLU6_layer(input)

    def test_Sigmoid(self):
        bm.random.seed()
        Sigmoid_layer = bp.dnn.Sigmoid()
        input = bm.random.randn(2)
        output = Sigmoid_layer(input)

    @parameterized.product(
        inplace=[True, False]
    )
    def test_Hardsigmoid(self, inplace):
        bm.random.seed()
        Hardsigmoid_layer = bp.dnn.Hardsigmoid(inplace=inplace)
        input = bm.random.randn(2)
        if inplace == True:
            Hardsigmoid_layer(input)
        elif inplace == False:
            output = Hardsigmoid_layer(input)

    def test_Tanh(self):
        bm.random.seed()
        Tanh_layer = bp.dnn.Tanh()
        input = bm.random.randn(2)
        output = Tanh_layer(input)

    @parameterized.product(
        inplace=[True, False]
    )
    def test_SiLU(self, inplace):
        bm.random.seed()
        SiLU_layer = bp.dnn.SiLU(inplace=inplace)
        input = bm.random.randn(2)
        if inplace == True:
            SiLU_layer(input)
        elif inplace == False:
            output = SiLU_layer(input)

    @parameterized.product(
        inplace=[True, False]
    )
    def test_Mish(self, inplace):
        bm.random.seed()
        Mish_layer = bp.dnn.Mish(inplace=inplace)
        input = bm.random.randn(2)
        if inplace == True:
            Mish_layer(input)
        elif inplace == False:
            output = Mish_layer(input)

    @parameterized.product(
        inplace=[True, False]
    )
    def test_Hardswish(self, inplace):
        bm.random.seed()
        Hardswish_layer = bp.dnn.Hardswish(inplace=inplace)
        input = bm.random.randn(2)
        if inplace == True:
            Hardswish_layer(input)
        elif inplace == False:
            output = Hardswish_layer(input)

    @parameterized.product(
        inplace=[True, False]
    )
    def test_ELU(self, inplace):
        bm.random.seed()
        ELU_layer = bp.dnn.ELU(alpha=0.5, inplace=inplace)
        input = bm.random.randn(2)
        if inplace == True:
            ELU_layer(input)
        elif inplace == False:
            output = ELU_layer(input)

    @parameterized.product(
        inplace=[True, False]
    )
    def test_CELU(self, inplace):
        bm.random.seed()
        CELU_layer = bp.dnn.CELU(alpha=0.5, inplace=inplace)
        input = bm.random.randn(2)
        if inplace == True:
            CELU_layer(input)
        elif inplace == False:
            output = CELU_layer(input)

    @parameterized.product(
        inplace=[True, False]
    )
    def test_SELU(self, inplace):
        bm.random.seed()
        SELU_layer = bp.dnn.SELU(inplace=inplace)
        input = bm.random.randn(2)
        if inplace == True:
            SELU_layer(input)
        elif inplace == False:
            output = SELU_layer(input)

    def test_GLU(self):
        bm.random.seed()
        GLU_layer = bp.dnn.GLU()
        input = bm.random.randn(4, 2)
        output = GLU_layer(input)

    @parameterized.product(
        approximate=['tanh', 'none']
    )
    def test_GELU(self, approximate):
        bm.random.seed()
        GELU_layer = bp.dnn.GELU()
        input = bm.random.randn(2)
        output = GELU_layer(input)

    def test_Hardshrink(self):
        bm.random.seed()
        Hardshrink_layer = bp.dnn.Hardshrink(lambd=1)
        input = bm.random.randn(2)
        output = Hardshrink_layer(input)

    @parameterized.product(
        inplace=[True, False]
    )
    def test_LeakyReLU(self, inplace):
        bm.random.seed()
        LeakyReLU_layer = bp.dnn.LeakyReLU(inplace=inplace)
        input = bm.random.randn(2)
        if inplace == True:
            LeakyReLU_layer(input)
        elif inplace == False:
            output = LeakyReLU_layer(input)

    def test_LogSigmoid(self):
        bm.random.seed()
        LogSigmoid_layer = bp.dnn.LogSigmoid()
        input = bm.random.randn(2)
        output = LogSigmoid_layer(input)

    @parameterized.product(
        beta=[1, 2, 3],
        threshold=[20, 21, 22]
    )
    def test_Softplus(self, beta, threshold):
        bm.random.seed()
        Softplus_layer = bp.dnn.Softplus(beta=beta, threshold=threshold)
        input = bm.random.randn(2)
        output = Softplus_layer(input)

    def test_Softshrink(self):
        bm.random.seed()
        Softshrink_layer = bp.dnn.Softshrink(lambd=1)
        input = bm.random.randn(2)
        output = Softshrink_layer(input)

    def test_PReLU(self):
        bm.random.seed()
        PReLU_layer = bp.dnn.PReLU(num_parameters=2, init=0.5)
        input = bm.random.randn(2)
        output = PReLU_layer(input)

    def test_Softsign(self):
        bm.random.seed()
        Softsign_layer = bp.dnn.Softsign()
        input = bm.random.randn(2)
        output = Softsign_layer(input)

    def test_Tanhshrink(self):
        bm.random.seed()
        Tanhshrink_layer = bp.dnn.Tanhshrink()
        input = bm.random.randn(2)
        output = Tanhshrink_layer(input)

    def test_Softmin(self):
        bm.random.seed()
        Softmin_layer = bp.dnn.Softmin(dim=2)
        input = bm.random.randn(2, 3, 4)
        output = Softmin_layer(input)

    def test_Softmax(self):
        bm.random.seed()
        Softmax_layer = bp.dnn.Softmax(dim=2)
        input = bm.random.randn(2, 3, 4)
        output = Softmax_layer(input)

    def test_Softmax2d(self):
        bm.random.seed()
        Softmax2d_layer = bp.dnn.Softmax2d()
        input = bm.random.randn(2, 3, 12, 13)
        output = Softmax2d_layer(input)

    def test_LogSoftmax(self):
        bm.random.seed()
        LogSoftmax_layer = bp.dnn.LogSoftmax(dim=2)
        input = bm.random.randn(2, 3, 4)
        output = LogSoftmax_layer(input)


if __name__ == '__main__':
    absltest.main()
