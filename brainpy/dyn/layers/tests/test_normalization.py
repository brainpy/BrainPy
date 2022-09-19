# -*- coding: utf-8 -*-


from unittest import TestCase

import brainpy as bp


class TestBatchNorm1d(TestCase):
	def test_batchnorm1d1(self):
		class BatchNormNet(bp.dyn.DynamicalSystem):
			def __init__(self):
				super(BatchNormNet, self).__init__()
				self.norm = bp.dyn.layers.BatchNorm1d(axis=(0, 1, 2))

			def update(self, shared, x):
				x = self.norm(shared, x)
				return x

		inputs = bp.math.ones((2, 3, 4))
		inputs[0, 0, :] = 2.
		inputs[0, 1, 0] = 5.
		print(inputs)
		model = BatchNormNet()
		shared = {'fit': False}
		print(model(shared, inputs))

	def test_batchnorm1d2(self):
		class BatchNormNet(bp.dyn.DynamicalSystem):
			def __init__(self):
				super(BatchNormNet, self).__init__()
				self.norm = bp.dyn.layers.BatchNorm1d()
				self.dense = bp.dyn.layers.Dense(num_in=4, num_out=4)

			def update(self, shared, x):
				x = self.norm(shared, x)
				x = self.dense(shared, x)
				return x

		inputs = bp.math.ones((2, 4))
		inputs[0, :] = 2.
		print(inputs)
		model = BatchNormNet()
		shared = {'fit': False}
		print(model(shared, inputs))


class TestBatchNorm2d(TestCase):
	def test_batchnorm2d(self):
		class BatchNormNet(bp.dyn.DynamicalSystem):
			def __init__(self):
				super(BatchNormNet, self).__init__()
				self.norm = bp.dyn.layers.BatchNorm2d()

			def update(self, shared, x):
				x = self.norm(shared, x)
				return x

		inputs = bp.math.ones((10, 32, 32, 3))
		inputs[0, 1, :, :] = 2.
		print(inputs)
		model = BatchNormNet()
		shared = {'fit': False}
		print(model(shared, inputs))


class TestBatchNorm3d(TestCase):
	def test_batchnorm3d(self):
		class BatchNormNet(bp.dyn.DynamicalSystem):
			def __init__(self):
				super(BatchNormNet, self).__init__()
				self.norm = bp.dyn.layers.BatchNorm3d()

			def update(self, shared, x):
				x = self.norm(shared, x)
				return x

		inputs = bp.math.ones((10, 32, 32, 16, 3))
		print(inputs)
		model = BatchNormNet()
		shared = {'fit': False}
		print(model(shared, inputs))


class TestBatchNorm(TestCase):
	def test_batchnorm1(self):
		class BatchNormNet(bp.dyn.DynamicalSystem):
			def __init__(self):
				super(BatchNormNet, self).__init__()
				self.norm = bp.dyn.layers.BatchNorm(axis=(0, 2), use_bias=False)	# channel axis: 1

			def update(self, shared, x):
				x = self.norm(shared, x)
				return x

		inputs = bp.math.ones((2, 3, 4))
		inputs[0, 0, :] = 2.
		inputs[0, 1, 0] = 5.
		print(inputs)
		model = BatchNormNet()
		shared = {'fit': False}
		print(model(shared, inputs))

	def test_batchnorm2(self):
		class BatchNormNet(bp.dyn.DynamicalSystem):
			def __init__(self):
				super(BatchNormNet, self).__init__()
				self.norm = bp.dyn.layers.BatchNorm(axis=(0, 2))			# channel axis: 1
				self.dense = bp.dyn.layers.Dense(num_in=12, num_out=2)

			def update(self, shared, x):
				x = self.norm(shared, x)
				x = x.reshape(-1, 12)
				x = self.dense(shared, x)
				return x

		inputs = bp.math.ones((2, 3, 4))
		inputs[0, 0, :] = 2.
		inputs[0, 1, 0] = 5.
		# print(inputs)
		model = BatchNormNet()
		shared = {'fit': False}
		print(model(shared, inputs))


class TestLayerNorm(TestCase):
	def test_layernorm1(self):
		class LayerNormNet(bp.dyn.DynamicalSystem):
			def __init__(self):
				super(LayerNormNet, self).__init__()
				self.norm = bp.dyn.layers.LayerNorm()

			def update(self, shared, x):
				x = self.norm(shared, x)
				return x

		inputs = bp.math.ones((2, 3, 4))
		inputs[0, 0, :] = 2.
		inputs[0, 1, 0] = 5.
		print(inputs)
		model = LayerNormNet()
		shared = {'fit': False}
		print(model(shared, inputs))

	def test_layernorm2(self):
		class LayerNormNet(bp.dyn.DynamicalSystem):
			def __init__(self):
				super(LayerNormNet, self).__init__()
				self.norm = bp.dyn.layers.LayerNorm(axis=2)

			def update(self, shared, x):
				x = self.norm(shared, x)
				return x

		inputs = bp.math.ones((2, 3, 4))
		inputs[0, 0, :] = 2.
		inputs[0, 1, 0] = 5.
		print(inputs)
		model = LayerNormNet()
		shared = {'fit': False}
		print(model(shared, inputs))


class TestInstanceNorm(TestCase):
	def test_instancenorm(self):
		class InstanceNormNet(bp.dyn.DynamicalSystem):
			def __init__(self):
				super(InstanceNormNet, self).__init__()
				self.norm = bp.dyn.layers.InstanceNorm()

			def update(self, shared, x):
				x = self.norm(shared, x)
				return x

		inputs = bp.math.ones((2, 3, 4))
		inputs[0, 0, :] = 2.
		inputs[0, 1, 0] = 5.
		print(inputs)
		model = InstanceNormNet()
		shared = {'fit': False}
		print(model(shared, inputs))


class TestGroupNorm(TestCase):
	def test_groupnorm1(self):
		class GroupNormNet(bp.dyn.DynamicalSystem):
			def __init__(self):
				super(GroupNormNet, self).__init__()
				self.norm = bp.dyn.layers.GroupNorm(num_groups=2)

			def update(self, shared, x):
				x = self.norm(shared, x)
				return x

		inputs = bp.math.ones((2, 3, 4))
		inputs[0, 0, :] = 2.
		inputs[0, 1, 0] = 5.
		print(inputs)
		model = GroupNormNet()
		shared = {'fit': False}
		print(model(shared, inputs))