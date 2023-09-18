import brainpy.math as bm
from absl.testing import parameterized
from absl.testing import absltest
import brainpy as bp

class Test_NVAR(parameterized.TestCase):
    @parameterized.product(
        mode=[bm.BatchingMode(),
              bm.NonBatchingMode()]
    )
    def test_NVAR(self,mode):
        bm.random.seed()
        input=bm.random.randn(1,5)
        layer=bp.dnn.NVAR(num_in=5,
                          delay=10,
                          mode=mode)
        if mode in [bm.NonBatchingMode()]:
            for i in input:
                output=layer(i)
        else:
            output=layer(input)

if __name__ == '__main__':
    absltest.main()