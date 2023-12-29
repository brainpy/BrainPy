import matplotlib.pyplot as plt

import brainpy as bp
import brainpy.math as bm


class NetForHalfProj(bp.DynamicalSystem):
  def __init__(self):
    super().__init__()

    self.pre = bp.dyn.PoissonGroup(10, 100.)
    self.post = bp.dyn.LifRef(1)
    self.syn = bp.dyn.HalfProjDelta(bp.dnn.Linear(10, 1, bp.init.OneInit(2.)), self.post)

  def update(self):
    self.syn(self.pre())
    self.post()
    return self.post.V.value


def test1():
  net = NetForHalfProj()
  indices = bm.arange(1000).to_numpy()
  vs = bm.for_loop(net.step_run, indices, progress_bar=True)
  bp.visualize.line_plot(indices, vs, show=False)
  plt.close('all')


class NetForFullProj(bp.DynamicalSystem):
  def __init__(self):
    super().__init__()

    self.pre = bp.dyn.PoissonGroup(10, 100.)
    self.post = bp.dyn.LifRef(1)
    self.syn = bp.dyn.FullProjDelta(self.pre, 0., bp.dnn.Linear(10, 1, bp.init.OneInit(2.)), self.post)

  def update(self):
    self.syn()
    self.pre()
    self.post()
    return self.post.V.value


def test2():
  net = NetForFullProj()
  indices = bm.arange(1000).to_numpy()
  vs = bm.for_loop(net.step_run, indices, progress_bar=True)
  bp.visualize.line_plot(indices, vs, show=False)
  plt.close('all')


