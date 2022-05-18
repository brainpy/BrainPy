# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.dyn import rates

bp.check.turn_off()


def bifurcation_analysis():
  model = rates.FHN(1, method='exp_auto')
  pp = bp.analysis.Bifurcation2D(
    model,
    target_vars={'x': [-2, 2], 'y': [-2, 2]},
    target_pars={'x_ext': [0, 2]},
    resolutions={'x_ext': 0.01}
  )
  pp.plot_bifurcation()
  pp.plot_limit_cycle_by_sim(duration=500)
  pp.show_figure()


class Network(bp.dyn.Network):
  def __init__(self, signal_speed=20.):
    super(Network, self).__init__()

    # Please download the processed data "hcp.npz" of the
    # ConnectomeDB of the Human Connectome Project (HCP)
    # from the following link:
    # - https://share.weiyun.com/wkPpARKy
    hcp = np.load('data/hcp.npz')
    conn_mat = bm.asarray(hcp['Cmat'])
    bm.fill_diagonal(conn_mat, 0)
    delay_mat = bm.round(hcp['Dmat'] / signal_speed / bm.get_dt())
    bm.fill_diagonal(delay_mat, 0)

    self.fhn = rates.FHN(80, x_ou_sigma=0.01, y_ou_sigma=0.01, name='fhn')
    self.coupling = rates.DiffusiveCoupling(self.fhn.x, self.fhn.x, self.fhn.input,
                                            conn_mat=conn_mat,
                                            delay_steps=delay_mat.astype(bm.int_),
                                            initial_delay_data=bp.init.Uniform(0, 0.05))


def brain_simulation():
  net = Network()
  runner = bp.dyn.DSRunner(net, monitors=['fhn.x'], inputs=['fhn.input', 0.72])
  runner.run(6e3)

  plt.rcParams['image.cmap'] = 'plasma'
  fig, axs = plt.subplots(1, 2, figsize=(12, 4))
  fc = bp.measure.functional_connectivity(runner.mon['fhn.x'])
  ax = axs[0].imshow(fc)
  plt.colorbar(ax, ax=axs[0])
  axs[1].plot(runner.mon.ts, runner.mon['fhn.x'][:, ::5], alpha=0.8)
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  bifurcation_analysis()
  brain_simulation()
