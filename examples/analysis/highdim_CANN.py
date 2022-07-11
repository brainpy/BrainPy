# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import brainpy as bp
import brainpy.math as bm

bm.set_platform('cpu')


class CANN1D(bp.dyn.NeuGroup):
  def __init__(self, num, tau=1., k=8.1, a=0.5, A=10., J0=4.,
               z_min=-bm.pi, z_max=bm.pi, **kwargs):
    super(CANN1D, self).__init__(size=num, **kwargs)

    # parameters
    self.tau = tau  # The synaptic time constant
    self.k = k  # Degree of the rescaled inhibition
    self.a = a  # Half-width of the range of excitatory connections
    self.A = A  # Magnitude of the external input
    self.J0 = J0  # maximum connection value

    # feature space
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = bm.linspace(z_min, z_max, num)  # The encoded feature values
    self.rho = num / self.z_range  # The neural density
    self.dx = self.z_range / num  # The stimulus density

    # variables
    self.u = bm.Variable(bm.zeros(num))
    self.input = bm.Variable(bm.zeros(num))

    # The connection matrix
    self.conn_mat = self.make_conn(self.x)

    # function
    self.integral = bp.odeint(self.derivative)

  def derivative(self, u, t, Iext):
    r1 = bm.square(u)
    r2 = 1.0 + self.k * bm.sum(r1)
    r = r1 / r2
    Irec = bm.dot(self.conn_mat, r)
    du = (-u + Irec + Iext) / self.tau
    return du

  def dist(self, d):
    d = bm.remainder(d, self.z_range)
    d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
    return d

  def make_conn(self, x):
    assert bm.ndim(x) == 1
    x_left = bm.reshape(x, (-1, 1))
    x_right = bm.repeat(x.reshape((1, -1)), len(x), axis=0)
    d = self.dist(x_left - x_right)
    Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
    return Jxx

  def get_stimulus_by_pos(self, pos):
    return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))

  def update(self, tdi):
    t, dt = tdi.get('t'), tdi.get('dt')
    self.u.value = self.integral(self.u, t, self.input, dt)
    self.input[:] = 0.


def find_fixed_points(pars=None, verbose=False, opt_method='gd', cand_method='random', tolerance=1e-6):
  if pars is None: pars = dict()
  cann = CANN1D(num=512, **pars)

  if cand_method == 'random':
    candidates = bm.random.uniform(0, 20., (1000, cann.num))
  elif cand_method == 'bump':
    candidates = cann.get_stimulus_by_pos(bm.arange(-bm.pi, bm.pi, 0.01).reshape((-1, 1)))
    candidates += bm.random.normal(0., 0.01, candidates.shape)
  else:
    raise ValueError

  finder = bp.analysis.SlowPointFinder(f_cell=cann, included_vars={'u': cann.u}, dt=1.)
  if opt_method == 'gd':
    finder.find_fps_with_gd_method(
      candidates={'u': candidates}, tolerance=tolerance, num_batch=200,
      optimizer=bp.optim.Adam(lr=bp.optim.ExponentialDecay(0.2, 1, 0.999)),
    )
  elif opt_method == 'BFGS':
    finder.find_fps_with_opt_solver({'u': candidates})
  else:
    raise ValueError
  finder.filter_loss(tolerance)
  finder.keep_unique(5e-3)

  if verbose:
    print(finder.fixed_points)
    print(finder.losses)
    print(finder.selected_ids)

  return finder.fixed_points, finder


def visualize_fixed_points(fixed_points):
  bp.visualize.animate_1D(
    dynamical_vars={'ys': fixed_points['u'],
                    'xs': bm.linspace(-bm.pi, bm.pi, fixed_points['u'].shape[1]),
                    'legend': 'fixed point'},
    frame_step=1,
    frame_delay=100,
    show=True,
    # save_path='cann_fps.gif'
  )


def verify_fixed_points_through_simulation(fixed_points, pars=None, num=3):
  if pars is None: pars = dict()
  cann = CANN1D(num=512, **pars)

  for i in range(num):
    cann.u[:] = fixed_points['u'][i]
    runner = bp.dyn.DSRunner(cann,
                             monitors=['u'],
                             dyn_vars=cann.vars())
    runner.run(100.)
    plt.plot(runner.mon.ts, runner.mon.u.max(axis=1))
    plt.ylim(0, runner.mon.u.max() + 1)
    plt.show()


def pca_reduction(fixed_points):
  pca = PCA(2)
  pca.fit(fixed_points['u'])
  fixedpoints_pc = pca.transform(fixed_points['u'])
  plt.plot(fixedpoints_pc[:, 0], fixedpoints_pc[:, 1], 'x', label='fixed points')

  plt.xlabel('PC 1')
  plt.ylabel('PC 2')
  plt.legend()
  plt.show()


if __name__ == '__main__':
  params = dict(k=0.1, a=0.5, A=20)
  fps, finder = find_fixed_points(params, cand_method='bump', tolerance=1e-7)
  # fps, finder = find_fixed_points(params, cand_method='random', opt_method='gd', tolerance=1e-7)
  # fps, finder = find_fixed_points(params, cand_method='random', opt_method='BFGS', tolerance=1e-5)
  visualize_fixed_points(fps)
  verify_fixed_points_through_simulation(fps, params)
  finder.compute_jacobians(fps['u'][:6], plot=True)
  pca_reduction(fps)

