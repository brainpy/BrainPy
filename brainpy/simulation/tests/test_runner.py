# -*- coding: utf-8 -*-


from unittest import TestCase

import brainpy as bp
import brainpy.math as bm


class ExampleDS(bp.DynamicalSystem):
  def __init__(self):
    super(ExampleDS, self).__init__()
    self.i = bm.Variable(bm.zeros(1))
    self.o = bm.Variable(bm.zeros(2))

  def update(self, _t, _dt):
    self.i += 1


class TestInputs(TestCase):
  def test_fix_type(self):
    duration = 10.
    dt = 0.1
    for jit in [True, False]:
      for run_method in [bp.ReportRunner, bp.StructRunner]:
        ds = ExampleDS()
        runner = run_method(ds, inputs=('o', 1.), monitors=['o'], dyn_vars=ds.vars(), jit=jit, dt=dt)
        runner(duration)
        length = int(duration / dt)
        assert bm.array_equal(runner.mon.o,
                              bm.repeat(bm.arange(length) + 1, 2).reshape((length, 2)))

  def test_iter_type_array(self):
    duration = 10.
    dt = 0.1
    for jit in [True, False]:
      for run_method in [bp.ReportRunner, bp.StructRunner]:
        ds = ExampleDS()
        length = int(duration / dt)
        runner = run_method(ds, inputs=('o', bm.ones(length), 'iter'), monitors=['o'],
                            dyn_vars=ds.vars(), jit=jit, dt=dt)
        runner(duration)
        assert bm.array_equal(runner.mon.o,
                              bm.repeat(bm.arange(length) + 1, 2).reshape((length, 2)))

  def test_iter_type_func(self):
    duration = 10.
    dt = 0.1
    for jit in [True, False]:
      for run_method in [bp.ReportRunner, bp.StructRunner]:
        ds = ExampleDS()

        def f_int():
          while True: yield 1.

        runner = run_method(ds, inputs=('o', f_int(), 'iter'), monitors=['o'],
                            dyn_vars=ds.vars(), jit=jit, dt=dt, )
        runner(duration)
        length = int(duration / dt)
        assert bm.array_equal(runner.mon.o,
                              bm.repeat(bm.arange(length) + 1, 2).reshape((length, 2)))

  def test_func_type(self):
    duration = 10.
    dt = 0.1
    for jit in [True, False]:
      for run_method in [bp.ReportRunner, bp.StructRunner]:
        ds = ExampleDS()

        def f_int(_t, _dt):
          return 1.

        runner = run_method(ds, inputs=('o', f_int, 'func'), monitors=['o'],
                            dyn_vars=ds.vars(), jit=jit, dt=dt, )
        runner(duration)
        length = int(duration / dt)
        assert bm.array_equal(runner.mon.o,
                              bm.repeat(bm.arange(length) + 1, 2).reshape((length, 2)))


class TestMonitor(TestCase):
  def test_1d_array(self):
    try1 = TryGroup(monitors=['a'])
    try1.a = np.ones(1)
    try1.run(100.)

    assert np.ndim(try1.mon.a) == 2 and np.shape(try1.mon.a)[1] == 1
    assert np.allclose(np.arange(2, 1002).reshape((-1, 1)), try1.mon.a)


  def test_2d_array():
    set(dt=0.1)
    try1 = TryGroup(monitors=['a'])
    try1.a = np.ones((2, 2))
    try1.run(100.)

    assert np.ndim(try1.mon.a) == 2 and np.shape(try1.mon.a)[1] == 4
    series = np.arange(2, 1002).reshape((-1, 1))
    series = np.repeat(series, 4, axis=1)
    assert np.allclose(series, try1.mon.a)


  def test_monitor_with_every():
    set(dt=0.1)

    # try1: 2d array
    try1 = TryGroup(monitors=Monitor(variables=['a'], every=[1.]))
    try1.run(100.)
    assert np.ndim(try1.mon.a) == 2 and np.shape(try1.mon.a)[1] == 4
    series = np.arange(2, 1002, 1. / 0.1).reshape((-1, 1))
    series = np.repeat(series, 4, axis=1)
    assert np.allclose(series, try1.mon.a)

    # try2: 1d array
    try2 = TryGroup(monitors=Monitor(variables=['a'], every=[1.]))
    try2.a = np.array([1., 1.])
    try2.run(100.)
    assert np.ndim(try2.mon.a) == 2 and np.shape(try2.mon.a)[1] == 2
    series = np.arange(2, 1002, 1. / 0.1).reshape((-1, 1))
    series = np.repeat(series, 2, axis=1)
    assert np.allclose(series, try2.mon.a)

    # try2: scalar
    try3 = TryGroup(monitors=Monitor(variables=['a'], every=[1.]))
    try3.a = 1.
    try3.run(100.)
    assert np.ndim(try3.mon.a) == 2 and np.shape(try3.mon.a)[1] == 1
    series = np.arange(2, 1002, 1. / 0.1).reshape((-1, 1))
    assert np.allclose(series, try3.mon.a)
