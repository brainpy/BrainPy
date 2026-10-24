# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

from brainpy.state._misc import set_module_as


class TestSetModuleAs(unittest.TestCase):
    def test_sets_module_on_function(self):
        @set_module_as('brainpy.state')
        def fn():
            return 1

        self.assertEqual(fn.__module__, 'brainpy.state')
        self.assertEqual(fn(), 1)

    def test_sets_module_on_class(self):
        @set_module_as('brainpy.state')
        class Foo:
            pass

        self.assertEqual(Foo.__module__, 'brainpy.state')
        self.assertEqual(Foo.__name__, 'Foo')
        self.assertIsInstance(Foo(), Foo)

    def test_returns_same_object(self):
        def fn():
            pass

        self.assertIs(set_module_as('m')(fn), fn)

        class Bar:
            pass

        self.assertIs(set_module_as('m')(Bar), Bar)

    def test_class_behavior_preserved(self):
        @set_module_as('brainpy.state')
        class Counter:
            def __init__(self, n):
                self.n = n

            def inc(self):
                self.n += 1
                return self.n

        c = Counter(0)
        self.assertEqual(c.inc(), 1)
        self.assertEqual(c.n, 1)


if __name__ == '__main__':
    unittest.main()
