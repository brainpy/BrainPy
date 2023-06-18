from absl.testing import parameterized

import brainpy

io_deprecated_names = list(brainpy.base.io.__deprecations.keys())


class Test(parameterized.TestCase):
  @parameterized.product(
    name=io_deprecated_names
  )
  def test_io(self, name):
    with self.assertWarns(DeprecationWarning):
      getattr(brainpy.base.io, name)
