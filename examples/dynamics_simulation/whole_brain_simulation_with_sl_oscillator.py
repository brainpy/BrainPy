# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import brainpy as bp
import brainpy.math as bm

bp.check.turn_off()


def bifurcation_analysis():
  model = bp.rates.StuartLandauOscillator(1, method='exp_auto')
  pp = bp.analysis.Bifurcation2D(
    model,
    target_vars={'x': [-2, 2], 'y': [-2, 2]},
    pars_update={'x_ext': 0., 'y_ext': 0., 'w': 0.2},
    target_pars={'a': [-2, 2]},
    resolutions={'a': 0.01}
  )
  pp.plot_bifurcation()
  pp.show_figure()


class Network(bp.Network):
  def __init__(self, noise=0.14):
    super(Network, self).__init__()

    # Please download the processed data "hcp.npz" of the
    # ConnectomeDB of the Human Connectome Project (HCP)
    # from the following link:
    # - https://share.weiyun.com/wkPpARKy
    hcp = np.load('data/hcp.npz')
    conn_mat = bm.asarray(hcp['Cmat'])
    bm.fill_diagonal(conn_mat, 0)
    gc = 0.6  # global coupling strength

    self.sl = bp.rates.StuartLandauOscillator(80, x_ou_sigma=noise, y_ou_sigma=noise)
    self.coupling = bp.synapses.DiffusiveCoupling(
      self.sl.x, self.sl.x,
      var_to_output=self.sl.input,
      conn_mat=conn_mat * gc
    )


def simulation():
  net = Network()
  runner = bp.DSRunner(net, monitors=['sl.x'], jit=True)
  runner.run(6e3)

  plt.rcParams['image.cmap'] = 'plasma'
  fig, axs = plt.subplots(1, 2, figsize=(12, 4))
  fc = bp.measure.functional_connectivity(runner.mon['sl.x'])
  ax = axs[0].imshow(fc)
  plt.colorbar(ax, ax=axs[0])
  axs[1].plot(runner.mon['ts'], runner.mon['sl.x'][:, ::5], alpha=0.8)
  plt.tight_layout()
  plt.show()


def net_analysis():
  import matplotlib
  matplotlib.use('WebAgg')
  bp.math.enable_x64()

  # get candidate points
  net = Network()
  runner = bp.DSRunner(
    net,
    monitors={'x': net.sl.x, 'y': net.sl.y},
    numpy_mon_after_run=False
  )
  runner.run(1e3)
  candidates = dict(x=runner.mon.x, y=runner.mon.y)

  # analysis
  net = Network(noise=0.)
  finder = bp.analysis.SlowPointFinder(
    net, target_vars={'x': net.sl.x, 'y': net.sl.y}
  )
  finder.find_fps_with_opt_solver(candidates=candidates)
  finder.filter_loss(1e-5)
  finder.keep_unique(1e-3)
  finder.compute_jacobians({'x': finder._fixed_points['x'][:10],
                            'y': finder._fixed_points['y'][:10]},
                           plot=True)


if __name__ == '__main__':
  bifurcation_analysis()
  simulation()
  net_analysis()
