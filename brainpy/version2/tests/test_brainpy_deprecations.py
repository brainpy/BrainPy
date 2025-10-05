from absl.testing import parameterized

import brainpy

bp_deprecated_names = list(brainpy.version2.__deprecations.keys())
mode_deprecated_names = list(brainpy.version2.modes.__deprecations.keys())
tools_deprecated_names = list(brainpy.version2.tools.__deprecations.keys())
train_deprecated_names = list(brainpy.version2.train.__deprecations.keys())
intg_deprecated_names = list(brainpy.version2.integrators.__deprecations.keys())


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
            getattr(brainpy.version2.modes, name)

    @parameterized.product(
        name=tools_deprecated_names
    )
    def test_brainpy_tools(self, name):
        with self.assertWarns(DeprecationWarning):
            getattr(brainpy.version2.tools, name)

    @parameterized.product(
        name=train_deprecated_names
    )
    def test_brainpy_train(self, name):
        with self.assertWarns(DeprecationWarning):
            getattr(brainpy.version2.train, name)

    # @parameterized.product(
    #   name=dyn_deprecated_names
    # )
    # def test_brainpy_dyn(self, name):
    #   with self.assertWarns(DeprecationWarning):
    #     getattr(brainpy.version2.dyn, name)
    #
    @parameterized.product(
        name=intg_deprecated_names
    )
    def test_brainpy_intg(self, name):
        with self.assertWarns(DeprecationWarning):
            getattr(brainpy.version2.integrators, name)
