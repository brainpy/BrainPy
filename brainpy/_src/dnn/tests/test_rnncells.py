import brainpy.math as bm
from absl.testing import parameterized
from brainpy.initialize import (XavierNormal,
                                ZeroInit,
                                Orthogonal,
                                parameter,
                                variable,
                                Initializer)
from absl.testing import absltest
import brainpy as bp


class Test_Rnncells(parameterized.TestCase):

    @parameterized.product(
        Wi_initializer=[XavierNormal(),
                        bm.ones([10, 64])],
        mode=[bm.TrainingMode(),
              bm.TrainingMode(20),
              bm.BatchingMode(),
              bm.BatchingMode(20),
              bm.NonBatchingMode()]
    )
    def test_RNNCell(self, Wi_initializer, mode):
        bm.random.seed()
        input = bm.random.randn(20, 10)
        layer = bp.dnn.RNNCell(num_in=10,
                               num_out=64,
                               Wi_initializer=Wi_initializer,
                               mode=mode
                               )
        if mode in [bm.TrainingMode(), bm.BatchingMode(), bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(50),
              bm.BatchingMode(),
              bm.BatchingMode(50),
              bm.NonBatchingMode()]
    )
    def test_GRUCell(self, mode):
        bm.random.seed()
        input = bm.random.randn(50, 100)
        layer = bp.dnn.GRUCell(num_in=100,
                               num_out=64,
                               mode=mode)
        if mode in [bm.TrainingMode(), bm.BatchingMode(), bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(50),
              bm.BatchingMode(),
              bm.BatchingMode(50),
              bm.NonBatchingMode()]
    )
    def test_LSTMCell(self, mode):
        bm.random.seed()
        input = bm.random.randn(50, 100)
        layer = bp.dnn.LSTMCell(num_in=100,
                                num_out=64,
                                mode=mode)
        if mode in [bm.TrainingMode(), bm.BatchingMode(), bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(4),
              bm.BatchingMode(),
              bm.BatchingMode(4), ]
    )
    def test_Conv1dLSTMCell(self, mode):
        bm.random.seed()
        input = bm.random.randn(4, 100, 3)
        layer = bp.dnn.Conv1dLSTMCell(input_shape=(100,),
                                      in_channels=3,
                                      out_channels=5,
                                      kernel_size=4,
                                      mode=mode)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(4),
              bm.BatchingMode(),
              bm.BatchingMode(4), ]
    )
    def test_Conv2dLSTMCell(self, mode):
        bm.random.seed()
        input = bm.random.randn(4, 100, 100, 3)
        layer = bp.dnn.Conv2dLSTMCell(input_shape=(100, 100),
                                      in_channels=3,
                                      out_channels=5,
                                      kernel_size=(4, 4),
                                      mode=mode)
        output = layer(input)

    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(4),
              bm.BatchingMode(),
              bm.BatchingMode(4), ]
    )
    def test_Conv3dLSTMCell(self, mode):
        bm.random.seed()
        input = bm.random.randn(4, 100, 100, 100, 3)
        layer = bp.dnn.Conv3dLSTMCell(input_shape=(100, 100, 100),
                                      in_channels=3,
                                      out_channels=5,
                                      kernel_size=(4, 4, 4),
                                      mode=mode)
        output = layer(input)


if __name__ == '__main__':
    absltest.main()
