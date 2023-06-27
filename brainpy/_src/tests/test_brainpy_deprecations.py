from absl.testing import parameterized
import brainpy

bp_deprecated_names = list(brainpy.__deprecations.keys())
mode_deprecated_names = list(brainpy.modes.__deprecations.keys())
tools_deprecated_names = list(brainpy.tools.__deprecations.keys())
train_deprecated_names = list(brainpy.train.__deprecations.keys())
# dyn_deprecated_names = list(brainpy.dyn.__deprecations.keys())
intg_deprecated_names = list(brainpy.integrators.__deprecations.keys())

io_deprecated_names = list(brainpy.base.io.__deprecations.keys())


class Test(parameterized.TestCase):
  @parameterized.product(
    name=bp_deprecated_names
  )
  def test_brainpy(self, name):
    with self.assertWarns(DeprecationWarning):
      getattr(brainpy, name)

  @parameterized.product(
    name=mode_deprecated_names
  )
  def test_brainpy_modes(self, name):
    with self.assertWarns(DeprecationWarning):
      getattr(brainpy.modes, name)

  @parameterized.product(
    name=tools_deprecated_names
  )
  def test_brainpy_tools(self, name):
    with self.assertWarns(DeprecationWarning):
      getattr(brainpy.tools, name)

  @parameterized.product(
    name=train_deprecated_names
  )
  def test_brainpy_train(self, name):
    with self.assertWarns(DeprecationWarning):
      getattr(brainpy.train, name)

  # @parameterized.product(
  #   name=dyn_deprecated_names
  # )
  # def test_brainpy_dyn(self, name):
  #   with self.assertWarns(DeprecationWarning):
  #     getattr(brainpy.dyn, name)
  #
  @parameterized.product(
    name=intg_deprecated_names
  )
  def test_brainpy_intg(self, name):
    with self.assertWarns(DeprecationWarning):
      getattr(brainpy.integrators, name)

  @parameterized.product(
    name=io_deprecated_names
  )
  def test_io(self, name):
    with self.assertWarns(DeprecationWarning):
      getattr(brainpy.base.io, name)
