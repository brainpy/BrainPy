# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import brainpy as bp

bp.math.use_backend('numpy')
bp.math.set_dt(0.005)


class LorenzSystem(bp.DynamicSystem):
  target_backend = 'general'

  def __init__(self, size=0, sigma=10, beta=8 / 3, rho=28, p=0.1, **kwargs):
    self.sigma = sigma
    self.beta = beta
    self.rho = rho
    self.p = p

    self.x = bp.math.ones(size)
    self.y = bp.math.ones(size)
    self.z = bp.math.ones(size)

    self.lorenz = bp.sdeint(f=self.lorenz_f, g=self.loren_g)

    super(LorenzSystem, self).__init__(**kwargs)

  def loren_g(self, x, y, z, t):
    return self.p * x, self.p * y, self.p * z

  def lorenz_f(self, x, y, z, t):
    dx = self.sigma * (y - x)
    dy = x * (self.rho - z) - y
    dz = x * y - self.beta * z
    return dx, dy, dz

  def update(self, _t, _i):
    self.x[:], self.y[:], self.z[:] = self.lorenz(self.x, self.y, self.z, _t)


sys = LorenzSystem(1, monitors=['x', 'y', 'z'])
sys.run(100.)

fig = plt.figure()
ax = fig.gca(projection='3d')
plt.plot(sys.mon.x[:, 0], sys.mon.y[:, 0], sys.mon.z[:, 0])
ax.set_xlabel('x')
ax.set_xlabel('y')
ax.set_xlabel('z')
plt.show()

if __name__ == '__main__':
  Axes3D
