# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

import brainpy as bp
import brainpy.math as bm

bp.math.enable_x64()


class GJCoupledFHN(bp.dyn.DynamicalSystem):
  def __init__(self, num=2, method='exp_auto'):
    super(GJCoupledFHN, self).__init__()

    # parameters
    self.num = num
    self.a = 0.7
    self.b = 0.8
    self.tau = 12.5
    self.gjw = 0.0001

    # variables
    self.V = bm.Variable(bm.random.uniform(-2, 2, num))
    self.w = bm.Variable(bm.random.uniform(-2, 2, num))
    self.Iext = bm.Variable(bm.zeros(num))

    # functions
    self.int_V = bp.odeint(self.dV, method=method)
    self.int_w = bp.odeint(self.dw, method=method)

  def dV(self, V, t, w, Iext=0.):
    gj = (V.reshape((-1, 1)) - V).sum(axis=0) * self.gjw
    dV = V - V * V * V / 3 - w + Iext + gj
    return dV

  def dw(self, w, t, V):
    dw = (V + self.a - self.b * w) / self.tau
    return dw

  def update(self, t, dt):
    self.V.value = self.int_V(self.V, t, self.w, self.Iext, dt)
    self.w.value = self.int_w(self.w, t, self.V, dt)
    self.Iext[:] = 0.


def d4_system():
  model = GJCoupledFHN(2)
  model.gjw = 0.01
  # Iext = bm.asarray([0., 0.1])
  Iext = bm.asarray([0., 0.6])

  # simulation
  runner = bp.dyn.DSRunner(model, monitors=['V'], inputs=['Iext', Iext])
  runner.run(300.)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V',
                         plot_ids=list(range(model.num)), show=True)

  # analysis
  finder = bp.analysis.SlowPointFinder(f_cell=model,
                                       included_vars={'V': model.V, 'w': model.w},
                                       inputs=['Iext', Iext])
  # finder.find_fps_with_gd_method(
  #   candidates={'V': bm.random.normal(0., 2., (1000, model.num)),
  #               'w': bm.random.normal(0., 2., (1000, model.num))},
  #   tolerance=1e-7,
  #   num_batch=200,
  #   optimizer=bp.optim.Adam(lr=bp.optim.ExponentialDecay(0.05, 1, 0.9999))
  # )
  finder.find_fps_with_opt_solver(candidates={'V': bm.random.normal(0., 2., (1000, model.num)),
                                              'w': bm.random.normal(0., 2., (1000, model.num))})
  finder.filter_loss(1e-7)
  finder.keep_unique()

  print('fixed_points: ', finder.fixed_points)
  print('losses:', finder.losses)
  if len(finder.fixed_points):
    jac = finder.compute_jacobians(finder.fixed_points)
    for i in range(len(finder.selected_ids)):
      eigval, eigvec = np.linalg.eig(np.asarray(jac[i]))
      plt.figure()
      plt.scatter(np.real(eigval), np.imag(eigval))
      plt.plot([1, 1], [-1, 1], '--')
      plt.xlabel('Real')
      plt.ylabel('Imaginary')
      plt.title(f'FP {i}')
      plt.show()


def d8_system():
  model = GJCoupledFHN(4)
  model.gjw = 0.1
  Iext = bm.asarray([0., 0., 0., 0.6])

  # simulation
  runner = bp.dyn.DSRunner(model, monitors=['V'], inputs=['Iext', Iext])
  runner.run(300.)

  bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V',
                         plot_ids=list(range(model.num)),
                         show=True)

  finder = bp.analysis.SlowPointFinder(f_cell=model,
                                       included_vars={'V': model.V, 'w': model.w},
                                       inputs=[model.Iext, Iext])
  finder.find_fps_with_gd_method(
    candidates={'V': bm.random.normal(0., 2., (1000, model.num)),
                'w': bm.random.normal(0., 2., (1000, model.num))},
    tolerance=1e-6,
    num_batch=200,
    optimizer=bp.optim.Adam(lr=bp.optim.ExponentialDecay(0.05, 1, 0.9999)),
  )
  finder.filter_loss(1e-7)
  finder.keep_unique()

  print('fixed_points: ', finder.fixed_points)
  print('losses:', finder.losses)
  if len(finder.fixed_points):
    jac = finder.compute_jacobians(finder.fixed_points)
    for i in range(finder.num_fps):
      eigval, eigvec = np.linalg.eig(np.asarray(jac[i]))
      plt.figure()
      plt.scatter(np.real(eigval), np.imag(eigval))
      plt.plot([1, 1], [-1, 1], '--')
      plt.xlabel('Real')
      plt.ylabel('Imaginary')
      plt.title(f'FP {i}')
      plt.show()


if __name__ == '__main__':
  d4_system()
  d8_system()
