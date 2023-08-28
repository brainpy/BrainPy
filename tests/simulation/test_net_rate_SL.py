import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
import unittest
import os

show = False


class Network(bp.Network):
  def __init__(self, noise=0.14):
    super(Network, self).__init__()

    hcp = np.load(os.path.join(os.path.dirname(__file__), 'data/hcp.npz'))
    conn_mat = bm.asarray(hcp['Cmat'])
    bm.fill_diagonal(conn_mat, 0)
    gc = 0.6  # global coupling strength
    self.sl = bp.rates.StuartLandauOscillator(80, x_ou_sigma=noise, y_ou_sigma=noise)
    self.coupling = bp.synapses.DiffusiveCoupling(
      self.sl.x, self.sl.x,
      var_to_output=self.sl.input,
      conn_mat=conn_mat * gc
    )


class TestSL(unittest.TestCase):
  def test1(self):
    bm.random.seed()
    net = Network()
    runner = bp.DSRunner(net, monitors=['sl.x'])
    runner.run(6e3 if show else 1e2)

    if show:
      plt.rcParams['image.cmap'] = 'plasma'
      fig, axs = plt.subplots(1, 2, figsize=(12, 4))
      fc = bp.measure.functional_connectivity(runner.mon['sl.x'])
      ax = axs[0].imshow(fc)
      plt.colorbar(ax, ax=axs[0])
      axs[1].plot(runner.mon['ts'], runner.mon['sl.x'][:, ::5], alpha=0.8)
      plt.tight_layout()
      plt.show()

