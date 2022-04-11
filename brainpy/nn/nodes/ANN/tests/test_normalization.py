# -*- coding: utf-8 -*-


from unittest import TestCase

import brainpy as bp


class TestBatchNorm1d(TestCase):
	def test_batchnorm1d1(self):
		i = bp.nn.Input((3, 4))
		b = bp.nn.BatchNorm1d()
		model = i >> b
		model.initialize(num_batch=2)
		# model.plot_node_graph(fig_size=(5, 5), node_size=500)

		inputs = bp.math.ones((2, 3, 4))
		inputs[0, 0, :] = 2.
		inputs[0, 1, 0] = 5.
		print(inputs)

		print(model(inputs))

	def test_batchnorm1d2(self):
		i = bp.nn.Input(4)
		b = bp.nn.BatchNorm1d()
		o = bp.nn.GeneralDense(4)
		model = i >> b >> o
		model.initialize(num_batch=2)

		inputs = bp.math.ones((2, 4))
		inputs[0, :] = 2.
		print(inputs)

		print(model(inputs))


class TestBatchNorm2d(TestCase):
	def test_batchnorm2d(self):
		i = bp.nn.Input((32, 32, 3))
		b = bp.nn.BatchNorm2d()
		model = i >> b
		model.initialize(num_batch=10)

		inputs = bp.math.ones((10, 32, 32, 3))
		inputs[0, 1, :, :] = 2.
		print(inputs.shape)

		print(model(inputs).shape)


class TestBatchNorm3d(TestCase):
	def test_batchnorm3d(self):
		i = bp.nn.Input((32, 32, 16, 3))
		b = bp.nn.BatchNorm3d()
		model = i >> b
		model.initialize(num_batch=10)

		inputs = bp.math.ones((10, 32, 32, 16, 3))
		print(inputs.shape)

		print(model(inputs).shape)


class TestBatchNorm(TestCase):
	def test_batchnorm1(self):
		i = bp.nn.Input((3, 4))
		b = bp.nn.BatchNorm(axis=(0, 2), use_bias=False)  # channel axis: 1
		model = i >> b
		model.initialize(num_batch=2)

		inputs = bp.math.ones((2, 3, 4))
		inputs[0, 0, :] = 2.
		inputs[0, 1, 0] = 5.
		print(inputs)

		print(model(inputs))

	def test_batchnorm2(self):
		i = bp.nn.Input((3, 4))
		b = bp.nn.BatchNorm(axis=(0, 2))  # channel axis: 1
		f = bp.nn.Reshape((-1, 12))
		o = bp.nn.GeneralDense(2)
		model = i >> b >> f >> o
		model.initialize(num_batch=2)

		inputs = bp.math.ones((2, 3, 4))
		inputs[0, 0, :] = 2.
		inputs[0, 1, 0] = 5.
		# print(inputs)
		print(model(inputs))

		# training
		X = bp.math.random.random((1000, 10, 3, 4))
		Y = bp.math.random.randint(0, 2, (1000, 10,  2))
		trainer = bp.nn.BPTT(model,
		                     loss=bp.losses.cross_entropy_loss,
		                     optimizer=bp.optim.Adam(lr=1e-3))
		trainer.fit([X, Y])


class TestLayerNorm(TestCase):
	def test_layernorm1(self):
		i = bp.nn.Input((3, 4))
		l = bp.nn.LayerNorm()
		model = i >> l
		model.initialize(num_batch=2)

		inputs = bp.math.ones((2, 3, 4))
		inputs[0, 0, :] = 2.
		inputs[0, 1, 0] = 5.
		print(inputs)

		print(model(inputs))

	def test_layernorm2(self):
		i = bp.nn.Input((3, 4))
		l = bp.nn.LayerNorm(axis=2)
		model = i >> l
		model.initialize(num_batch=2)

		inputs = bp.math.ones((2, 3, 4))
		inputs[0, 0, :] = 2.
		inputs[0, 1, 0] = 5.
		print(inputs)

		print(model(inputs))


class TestInstanceNorm(TestCase):
	def test_instancenorm(self):
		i = bp.nn.Input((3, 4))
		l = bp.nn.InstanceNorm()
		model = i >> l
		model.initialize(num_batch=2)

		inputs = bp.math.ones((2, 3, 4))
		inputs[0, 0, :] = 2.
		inputs[0, 1, 0] = 5.
		print(inputs)

		print(model(inputs))


class TestGroupNorm(TestCase):
	def test_groupnorm1(self):
		i = bp.nn.Input((3, 4))
		l = bp.nn.GroupNorm(num_groups=2)
		model = i >> l
		model.initialize(num_batch=2)

		inputs = bp.math.ones((2, 3, 4))
		inputs[0, 0, :] = 2.
		inputs[0, 1, 0] = 5.
		print(inputs)

		print(model(inputs))
