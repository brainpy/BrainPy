# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
import unittest


class TestIO1(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(TestIO1, self).__init__(*args, **kwargs)

    rng = bm.random.RandomState()

    class IO1(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(IO1, self).__init__()

        self.a = bm.Variable(bm.zeros(1))
        self.b = bm.Variable(bm.ones(3))
        self.c = bm.Variable(bm.ones((3, 4)))
        self.d = bm.Variable(bm.ones((2, 3, 4)))

    class IO2(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(IO2, self).__init__()

        self.a = bm.Variable(rng.rand(3))
        self.b = bm.Variable(rng.randn(10))

    io1 = IO1()
    io2 = IO2()
    io1.a2 = io2.a
    io1.b2 = io2.b
    io2.a2 = io1.a
    io2.b2 = io2.b

    self.net = bp.dyn.Container(io1, io2)

    print(self.net.vars().keys())
    print(self.net.vars().unique().keys())

  def test_h5(self):
    bp.base.save_as_h5('io_test_tmp.h5', self.net.vars())
    bp.base.load_by_h5('io_test_tmp.h5', self.net, verbose=True)

    bp.base.save_as_h5('io_test_tmp.hdf5', self.net.vars())
    bp.base.load_by_h5('io_test_tmp.hdf5', self.net, verbose=True)

  def test_h5_postfix(self):
    with self.assertRaises(ValueError):
      bp.base.save_as_h5('io_test_tmp.h52', self.net.vars())
    with self.assertRaises(ValueError):
      bp.base.load_by_h5('io_test_tmp.h52', self.net, verbose=True)

  def test_npz(self):
    bp.base.save_as_npz('io_test_tmp.npz', self.net.vars())
    bp.base.load_by_npz('io_test_tmp.npz', self.net, verbose=True)

    bp.base.save_as_npz('io_test_tmp_compressed.npz', self.net.vars(), compressed=True)
    bp.base.load_by_npz('io_test_tmp_compressed.npz', self.net, verbose=True)

  def test_npz_postfix(self):
    with self.assertRaises(ValueError):
      bp.base.save_as_npz('io_test_tmp.npz2', self.net.vars())
    with self.assertRaises(ValueError):
      bp.base.load_by_npz('io_test_tmp.npz2', self.net, verbose=True)

  def test_pkl(self):
    bp.base.save_as_pkl('io_test_tmp.pkl', self.net.vars())
    bp.base.load_by_pkl('io_test_tmp.pkl', self.net, verbose=True)

    bp.base.save_as_pkl('io_test_tmp.pickle', self.net.vars())
    bp.base.load_by_pkl('io_test_tmp.pickle', self.net, verbose=True)

  def test_pkl_postfix(self):
    with self.assertRaises(ValueError):
      bp.base.save_as_pkl('io_test_tmp.pkl2', self.net.vars())
    with self.assertRaises(ValueError):
      bp.base.load_by_pkl('io_test_tmp.pkl2', self.net, verbose=True)

  def test_mat(self):
    bp.base.save_as_mat('io_test_tmp.mat', self.net.vars())
    bp.base.load_by_mat('io_test_tmp.mat', self.net, verbose=True)

  def test_mat_postfix(self):
    with self.assertRaises(ValueError):
      bp.base.save_as_mat('io_test_tmp.mat2', self.net.vars())
    with self.assertRaises(ValueError):
      bp.base.load_by_mat('io_test_tmp.mat2', self.net, verbose=True)


class TestIO2(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(TestIO2, self).__init__(*args, **kwargs)

    rng = bm.random.RandomState()

    class IO1(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(IO1, self).__init__()

        self.a = bm.Variable(bm.zeros(1))
        self.b = bm.Variable(bm.ones(3))
        self.c = bm.Variable(bm.ones((3, 4)))
        self.d = bm.Variable(bm.ones((2, 3, 4)))

    class IO2(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(IO2, self).__init__()

        self.a = bm.Variable(rng.rand(3))
        self.b = bm.Variable(rng.randn(10))

    io1 = IO1()
    io2 = IO2()

    self.net = bp.dyn.Container(io1, io2)

    print(self.net.vars().keys())
    print(self.net.vars().unique().keys())

  def test_h5(self):
    bp.base.save_as_h5('io_test_tmp.h5', self.net.vars())
    bp.base.load_by_h5('io_test_tmp.h5', self.net, verbose=True)

    bp.base.save_as_h5('io_test_tmp.hdf5', self.net.vars())
    bp.base.load_by_h5('io_test_tmp.hdf5', self.net, verbose=True)

  def test_h5_postfix(self):
    with self.assertRaises(ValueError):
      bp.base.save_as_h5('io_test_tmp.h52', self.net.vars())
    with self.assertRaises(ValueError):
      bp.base.load_by_h5('io_test_tmp.h52', self.net, verbose=True)

  def test_npz(self):
    bp.base.save_as_npz('io_test_tmp.npz', self.net.vars())
    bp.base.load_by_npz('io_test_tmp.npz', self.net, verbose=True)

    bp.base.save_as_npz('io_test_tmp_compressed.npz', self.net.vars(), compressed=True)
    bp.base.load_by_npz('io_test_tmp_compressed.npz', self.net, verbose=True)

  def test_npz_postfix(self):
    with self.assertRaises(ValueError):
      bp.base.save_as_npz('io_test_tmp.npz2', self.net.vars())
    with self.assertRaises(ValueError):
      bp.base.load_by_npz('io_test_tmp.npz2', self.net, verbose=True)

  def test_pkl(self):
    bp.base.save_as_pkl('io_test_tmp.pkl', self.net.vars())
    bp.base.load_by_pkl('io_test_tmp.pkl', self.net, verbose=True)

    bp.base.save_as_pkl('io_test_tmp.pickle', self.net.vars())
    bp.base.load_by_pkl('io_test_tmp.pickle', self.net, verbose=True)

  def test_pkl_postfix(self):
    with self.assertRaises(ValueError):
      bp.base.save_as_pkl('io_test_tmp.pkl2', self.net.vars())
    with self.assertRaises(ValueError):
      bp.base.load_by_pkl('io_test_tmp.pkl2', self.net, verbose=True)

  def test_mat(self):
    bp.base.save_as_mat('io_test_tmp.mat', self.net.vars())
    bp.base.load_by_mat('io_test_tmp.mat', self.net, verbose=True)

  def test_mat_postfix(self):
    with self.assertRaises(ValueError):
      bp.base.save_as_mat('io_test_tmp.mat2', self.net.vars())
    with self.assertRaises(ValueError):
      bp.base.load_by_mat('io_test_tmp.mat2', self.net, verbose=True)
