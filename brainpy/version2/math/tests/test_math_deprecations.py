from absl.testing import parameterized

import brainpy.version2.math as bm

deprecated_names = list(bm.__deprecations.keys())


class Test(parameterized.TestCase):
    @parameterized.product(
        name=deprecated_names
    )
    def test(self, name):
        with self.assertWarns(DeprecationWarning):
            getattr(bm, name)
