# -*- coding: utf-8 -*-


import unittest

import brainpy as bp
import brainpy.math as bm


class TestExplicitRKStateDelay(unittest.TestCase):
  def test_euler(self):
    xdelay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')

    @bp.ddeint(method='euler', state_delays={'x': xdelay})
    def equation(x, t, ):
      return -xdelay(t - 1)

    runner = bp.integrators.IntegratorRunner(equation, monitors=['x'])
    runner.run(20.)

    bp.visualize.line_plot(runner.mon.ts, runner.mon['x'], show=True)

  def test_midpoint(self):
    xdelay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')

    @bp.ddeint(method='midpoint', state_delays={'x': xdelay})
    def equation(x, t, ):
      return -xdelay(t - 1)

    runner = bp.integrators.IntegratorRunner(equation, monitors=['x'])
    runner.run(20.)

    bp.visualize.line_plot(runner.mon.ts, runner.mon['x'], show=True)

  def test_heun2(self):
    xdelay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')

    @bp.ddeint(method='heun2', state_delays={'x': xdelay})
    def equation(x, t, ):
      return -xdelay(t - 1)

    runner = bp.integrators.IntegratorRunner(equation, monitors=['x'])
    runner.run(20.)

    bp.visualize.line_plot(runner.mon.ts, runner.mon['x'], show=True)

  def test_ralston2(self):
    xdelay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')

    @bp.ddeint(method='ralston2', state_delays={'x': xdelay})
    def equation(x, t, ):
      return -xdelay(t - 1)

    runner = bp.integrators.IntegratorRunner(equation, monitors=['x'])
    runner.run(20.)

    bp.visualize.line_plot(runner.mon.ts, runner.mon['x'], show=True)

  def test_rk2(self):
    xdelay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')

    @bp.ddeint(method='rk2',
               state_delays={'x': xdelay})
    def equation(x, t, ):
      return -xdelay(t - 1)

    runner = bp.integrators.IntegratorRunner(equation, monitors=['x'])
    runner.run(20.)

    bp.visualize.line_plot(runner.mon.ts, runner.mon['x'], show=True)

  def test_rk3(self):
    xdelay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')

    @bp.ddeint(method='rk3', state_delays={'x': xdelay})
    def equation(x, t, ):
      return -xdelay(t - 1)

    runner = bp.integrators.IntegratorRunner(equation, monitors=['x'])
    runner.run(20.)

    bp.visualize.line_plot(runner.mon.ts, runner.mon['x'], show=True)

  def test_heun3(self):
    xdelay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')

    @bp.ddeint(method='heun3', state_delays={'x': xdelay})
    def equation(x, t, ):
      return -xdelay(t - 1)

    runner = bp.integrators.IntegratorRunner(equation, monitors=['x'])
    runner.run(20.)

    bp.visualize.line_plot(runner.mon.ts, runner.mon['x'], show=True)

  def test_ralston3(self):
    xdelay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')

    @bp.ddeint(method='ralston3', state_delays={'x': xdelay})
    def equation(x, t, ):
      return -xdelay(t - 1)

    runner = bp.integrators.IntegratorRunner(equation, monitors=['x'])
    runner.run(20.)

    bp.visualize.line_plot(runner.mon.ts, runner.mon['x'], show=True)

  def test_ssprk3(self):
    xdelay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')

    @bp.ddeint(method='ssprk3', state_delays={'x': xdelay})
    def equation(x, t, ):
      return -xdelay(t - 1)

    runner = bp.integrators.IntegratorRunner(equation, monitors=['x'])
    runner.run(20.)

    bp.visualize.line_plot(runner.mon.ts, runner.mon['x'], show=True)

  def test_rk4(self):
    xdelay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')

    @bp.ddeint(method='rk4', state_delays={'x': xdelay})
    def equation(x, t, ):
      return -xdelay(t - 1)

    runner = bp.integrators.IntegratorRunner(equation, monitors=['x'])
    runner.run(20.)

    bp.visualize.line_plot(runner.mon.ts, runner.mon['x'], show=True)

  def test_ralston4(self):
    xdelay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')

    @bp.ddeint(method='ralston4', state_delays={'x': xdelay})
    def equation(x, t, ):
      return -xdelay(t - 1)

    runner = bp.integrators.IntegratorRunner(equation, monitors=['x'])
    runner.run(20.)

    bp.visualize.line_plot(runner.mon.ts, runner.mon['x'], show=True)

  def test_rk4_38rule(self):
    xdelay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')

    @bp.ddeint(method='rk4_38rule', state_delays={'x': xdelay})
    def equation(x, t, ):
      return -xdelay(t - 1)

    runner = bp.integrators.IntegratorRunner(equation, monitors=['x'])
    runner.run(20.)

    bp.visualize.line_plot(runner.mon.ts, runner.mon['x'], show=True)

