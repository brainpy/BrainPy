import brainpy as bp
import unittest
import numpy as np
import brainpy.math as bm
import matplotlib.pyplot as plt
import os


show = False
bm.set_platform('cpu')

class Network(bp.Network):
  def __init__(self, signal_speed=20.):
    super(Network, self).__init__()

    hcp = np.load(os.path.join(os.path.dirname(__file__), 'data/hcp.npz'))
    conn_mat = bm.asarray(hcp['Cmat'])
    bm.fill_diagonal(conn_mat, 0)
    delay_mat = bm.round(hcp['Dmat'] / signal_speed / bm.dt).astype(bm.int_)
    delay_mat = bm.asarray(delay_mat)
    bm.fill_diagonal(delay_mat, 0)

    self.fhn = bp.rates.FHN(
      80,
      x_ou_sigma=0.01,
      y_ou_sigma=0.01,
    )
    self.coupling = bp.synapses.DiffusiveCoupling(
      self.fhn.x,
      self.fhn.x,
      var_to_output=self.fhn.input,
      conn_mat=conn_mat,
      delay_steps=delay_mat,
      initial_delay_data=bp.init.Uniform(0, 0.05)
    )


class TestFHN(unittest.TestCase):
  def test1(self):
    net = Network()
    runner = bp.DSRunner(net, monitors=['fhn.x'], inputs=['fhn.input', 0.72], jit=True)
    runner.run(6e3 if show else 1e2)

    if show:
      plt.rcParams['image.cmap'] = 'plasma'
      fig, axs = plt.subplots(1, 2, figsize=(12, 4))
      fc = bp.measure.functional_connectivity(runner.mon['fhn.x'])
      ax = axs[0].imshow(fc)
      plt.colorbar(ax, ax=axs[0])
      axs[1].plot(runner.mon['ts'], runner.mon['fhn.x'][:, ::5], alpha=0.8)
      plt.tight_layout()
      plt.show()

    bm.clear_buffer_memory()
