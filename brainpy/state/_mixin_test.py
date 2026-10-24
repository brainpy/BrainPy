# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import jax.numpy as jnp
import brainstate
import brainunit as u

import brainpy.state._mixin as mixin_mod
from brainpy.state._mixin import AlignPost, BindCondData


class TestModuleExports(unittest.TestCase):
    def test_all_lists_both_mixins(self):
        self.assertEqual(set(mixin_mod.__all__), {'AlignPost', 'BindCondData'})

    def test_exported_names_are_accessible(self):
        for name in mixin_mod.__all__:
            self.assertTrue(hasattr(mixin_mod, name))


class TestAlignPost(unittest.TestCase):
    def test_is_mixin_subclass(self):
        self.assertTrue(issubclass(AlignPost, brainstate.mixin.Mixin))

    def test_instantiable_without_args(self):
        # Mixins provide behavior without initialization, so the bare class
        # must construct without arguments.
        self.assertIsInstance(AlignPost(), AlignPost)

    def test_defines_no_init(self):
        # Mixin contract: mixins should not define their own __init__.
        self.assertNotIn('__init__', AlignPost.__dict__)

    def test_base_method_raises_not_implemented(self):
        obj = AlignPost()
        with self.assertRaises(NotImplementedError):
            obj.align_post_input_add()
        with self.assertRaises(NotImplementedError):
            obj.align_post_input_add(1.0)
        with self.assertRaises(NotImplementedError):
            obj.align_post_input_add(current=jnp.ones(3), scale=2.0)

    def test_subclass_override_accumulates(self):
        class Synapse(AlignPost):
            def __init__(self, weight):
                self.weight = weight
                self.post_current = 0.0

            def align_post_input_add(self, current):
                self.post_current += current * self.weight

        syn = Synapse(weight=0.5)
        syn.align_post_input_add(10.0)
        self.assertEqual(syn.post_current, 5.0)
        syn.align_post_input_add(10.0)
        self.assertEqual(syn.post_current, 10.0)

    def test_subclass_override_with_arrays(self):
        class NeuronGroup(AlignPost):
            def __init__(self, size):
                self.input_current = jnp.zeros(size)

            def align_post_input_add(self, current):
                self.input_current = self.input_current + current

        neurons = NeuronGroup(4)
        neurons.align_post_input_add(jnp.ones(4) * 0.5)
        self.assertTrue(jnp.allclose(neurons.input_current, jnp.full(4, 0.5)))


class TestBindCondData(unittest.TestCase):
    def test_is_mixin_subclass(self):
        self.assertTrue(issubclass(BindCondData, brainstate.mixin.Mixin))

    def test_instantiable_without_args(self):
        self.assertIsInstance(BindCondData(), BindCondData)

    def test_defines_no_init(self):
        self.assertNotIn('__init__', BindCondData.__dict__)

    def test_conductance_is_annotation_only(self):
        # The ``_conductance`` annotation must not create a class attribute,
        # so a fresh instance has no conductance bound yet.
        self.assertIn('_conductance', BindCondData.__annotations__)
        self.assertFalse(hasattr(BindCondData(), '_conductance'))

    def test_unset_conductance_raises_attribute_error(self):
        obj = BindCondData()
        with self.assertRaises(AttributeError):
            _ = obj._conductance

    def test_bind_sets_scalar(self):
        obj = BindCondData()
        obj.bind_cond(0.5)
        self.assertEqual(obj._conductance, 0.5)

    def test_unbind_clears(self):
        obj = BindCondData()
        obj.bind_cond(0.5)
        obj.unbind_cond()
        self.assertIsNone(obj._conductance)

    def test_unbind_before_bind_sets_none(self):
        # ``unbind_cond`` assigns None, so it is safe even before any bind.
        obj = BindCondData()
        obj.unbind_cond()
        self.assertIsNone(obj._conductance)

    def test_bind_unbind_roundtrip(self):
        obj = BindCondData()
        for value in (1.0, 2.5, -3.0):
            obj.bind_cond(value)
            self.assertEqual(obj._conductance, value)
            obj.unbind_cond()
            self.assertIsNone(obj._conductance)

    def test_rebind_overwrites(self):
        obj = BindCondData()
        obj.bind_cond(0.1)
        obj.bind_cond(0.9)
        self.assertEqual(obj._conductance, 0.9)

    def test_bind_array(self):
        obj = BindCondData()
        g = jnp.array([0.1, 0.2, 0.3])
        obj.bind_cond(g)
        self.assertTrue(jnp.allclose(obj._conductance, g))

    def test_bind_quantity_preserves_units(self):
        obj = BindCondData()
        g = 0.5 * u.nS
        obj.bind_cond(g)
        self.assertTrue(u.get_unit(obj._conductance).has_same_dim(u.nS))

    def test_bind_none_explicitly(self):
        obj = BindCondData()
        obj.bind_cond(None)
        self.assertIsNone(obj._conductance)

    def test_instances_are_independent(self):
        a, b = BindCondData(), BindCondData()
        a.bind_cond(1.0)
        self.assertEqual(a._conductance, 1.0)
        self.assertFalse(hasattr(b, '_conductance'))


class TestCombinedMixins(unittest.TestCase):
    """Both mixins composed onto a single conductance-based synapse."""

    def _make_synapse(self):
        class CondSynapse(AlignPost, BindCondData):
            def __init__(self, reversal):
                self.reversal = reversal
                self.post_current = 0.0

            def align_post_input_add(self, voltage):
                # Use the temporarily bound conductance, then release it.
                current = self._conductance * (self.reversal - voltage)
                self.post_current += current
                self.unbind_cond()

        return CondSynapse(reversal=0.0)

    def test_is_both_mixins(self):
        syn = self._make_synapse()
        self.assertIsInstance(syn, AlignPost)
        self.assertIsInstance(syn, BindCondData)
        self.assertIsInstance(syn, brainstate.mixin.Mixin)

    def test_bind_then_align_flow(self):
        syn = self._make_synapse()
        syn.bind_cond(2.0)
        syn.align_post_input_add(voltage=-1.0)
        # 2.0 * (0.0 - (-1.0)) == 2.0
        self.assertEqual(syn.post_current, 2.0)
        # conductance released after use
        self.assertIsNone(syn._conductance)


if __name__ == '__main__':
    unittest.main()
