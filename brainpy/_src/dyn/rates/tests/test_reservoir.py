import brainpy.math as bm
from absl.testing import parameterized
from absl.testing import absltest
import brainpy as bp


class Test_Reservoir(parameterized.TestCase):
    @parameterized.product(
        mode=[bm.TrainingMode(),
              bm.TrainingMode(10),
              bm.BatchingMode(),
              bm.BatchingMode(10),
              bm.NonBatchingMode()]
    )
    def test_Reservoir(self, mode):
        bm.random.seed()
        input = bm.random.randn(10, 3)
        layer = bp.dyn.Reservoir(input_shape=3,
                                 num_out=5,
                                 mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output = layer(i)
        else:
            output = layer(input)


if __name__ == '__main__':
    absltest.main()
