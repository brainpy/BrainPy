# -*- coding: utf-8 -*-

import brainpy as bp

bp.math.enable_x64()
bp.math.set_platform('cpu')


def quadratic_system1():
  int_x = bp.odeint(lambda x, t: -x ** 2)
  analyzer = bp.analysis.PhasePlane1D(model=int_x,
                                      target_vars={'x': [-2, 2]},
                                      resolutions=0.001)
  analyzer.plot_vector_field()
  analyzer.plot_fixed_point(show=True)

  int_x = bp.odeint(lambda x, t: x ** 2)
  analyzer = bp.analysis.PhasePlane1D(model=int_x,
                                      target_vars={'x': [-2, 2]},
                                      resolutions=0.001)
  analyzer.plot_vector_field()
  analyzer.plot_fixed_point(show=True)


def cubic_system1():
  int_x = bp.odeint(lambda x, t: -x ** 3)
  analyzer = bp.analysis.PhasePlane1D(model=int_x,
                                      target_vars={'x': [-2, 2]},
                                      resolutions=0.001)
  analyzer.plot_vector_field()
  analyzer.plot_fixed_point(show=True)

  int_x = bp.odeint(lambda x, t: x ** 3)
  analyzer = bp.analysis.PhasePlane1D(model=int_x,
                                      target_vars={'x': [-2, 2]},
                                      resolutions=0.001)
  analyzer.plot_vector_field()
  analyzer.plot_fixed_point(show=True)


def cubic_system_2():
  @bp.odeint
  def int_x(x, t, Iext):
    return x ** 3 - x + Iext

  analyzer = bp.analysis.PhasePlane1D(model=int_x,
                                      target_vars={'x': [-2, 2]},
                                      pars_update={'Iext': 0.},
                                      resolutions=0.001)
  analyzer.plot_vector_field()
  analyzer.plot_fixed_point(show=True)


def sin_1d():
  @bp.odeint
  def int_x(x, t, Iext):
    return bp.math.sin(x) + Iext

  pp = bp.analysis.PhasePlane1D(model=int_x,
                                target_vars={'x': [-5, 5]},
                                pars_update={'Iext': 0.9},
                                resolutions=0.001)
  pp.plot_vector_field()
  pp.plot_fixed_point(show=True)

  bf = bp.analysis.Bifurcation1D(model=int_x,
                                 target_vars={'x': [-5, 5]},
                                 target_pars={'Iext': [0., 1.5]},
                                 resolutions=0.001)
  bf.plot_bifurcation(show=True, tol_aux=1e-7)


def sincos_1d():
  @bp.odeint
  def int_x(x, t, a=1., b=1.):
    return bp.math.sin(a * x) + bp.math.cos(b * x)

  pp = bp.analysis.PhasePlane1D(
    model=int_x,
    target_vars={'x': [-bp.math.pi, bp.math.pi]},
    resolutions=0.001
  )
  pp.plot_vector_field()
  pp.plot_fixed_point(show=True)

  bf = bp.analysis.Bifurcation1D(
    model=int_x,
    target_vars={'x': [-bp.math.pi, bp.math.pi]},
    target_pars={'a': [0.5, 1.5], 'b': [0.5, 1.5]},
    resolutions={'a': 0.01, 'b': 0.01}
  )
  bf.plot_bifurcation(show=True)


if __name__ == '__main__':
    sin_1d()
