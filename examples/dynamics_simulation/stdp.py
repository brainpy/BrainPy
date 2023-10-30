"""
Reproduce the following STDP paper:

- Song, S., Miller, K. & Abbott, L. Competitive Hebbian learning through spike-timing-dependent
  synaptic plasticity. Nat Neurosci 3, 919–926 (2000). https://doi.org/10.1038/78829
"""

import matplotlib.pyplot as plt
import numpy as np

import brainpy as bp
import brainpy.math as bm


class STDPNet(bp.DynSysGroup):
  def __init__(self, num_poisson, num_lif=1, g_max=0.01):
    super().__init__()

    self.g_max = g_max

    # neuron groups
    self.noise = bp.dyn.PoissonGroup(num_poisson, freqs=15.)
    self.group = bp.dyn.Lif(num_lif, V_reset=-60., V_rest=-74, V_th=-54, tau=10.,
                            V_initializer=bp.init.Normal(-60., 1.))

    # synapses
    syn = bp.dyn.Expon.desc(num_lif, tau=5.)
    out = bp.dyn.COBA.desc(E=0.)
    comm = bp.dnn.AllToAll(num_poisson, num_lif, bp.init.Uniform(0., g_max))
    self.syn = bp.dyn.STDP_Song2000(self.noise, None, syn, comm, out, self.group,
                                    tau_s=20, tau_t=20, W_max=g_max, W_min=0.,
                                    A1=0.01 * g_max, A2=0.0105 * g_max)

  def update(self, *args, **kwargs):
    self.noise()
    self.syn()
    self.group()
    return self.syn.comm.weight.flatten()[:10]


def run_model():
  net = STDPNet(1000, 1)
  indices = np.arange(int(100.0e3 / bm.dt))  # 100 s
  ws = bm.for_loop(net.step_run, indices, progress_bar=True)
  weight = bm.as_numpy(net.syn.comm.weight.flatten())

  fig, gs = bp.visualize.get_figure(3, 1, 3, 10)
  fig.add_subplot(gs[0, 0])
  plt.plot(weight / net.g_max, '.k')
  plt.xlabel('Weight / gmax')

  fig.add_subplot(gs[1, 0])
  plt.hist(weight / net.g_max, 20)
  plt.xlabel('Weight / gmax')

  fig.add_subplot(gs[2, 0])
  plt.plot(indices * bm.dt, bm.as_numpy(ws) / net.g_max)
  plt.xlabel('Time (s)')
  plt.ylabel('Weight / gmax')
  plt.show()


if __name__ == '__main__':
  run_model()
