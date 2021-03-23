# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import brainpy as bp

bp.backend.set('numpy', dt=0.005)


class LorenzSystem(bp.DynamicSystem):
    target_backend = 'general'

    def __init__(self, size=0, sigma=10, beta=8 / 3, rho=28, p=0.1, **kwargs):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.p = p

        self.x = bp.backend.ones(size)
        self.y = bp.backend.ones(size)
        self.z = bp.backend.ones(size)

        def lorenz_g(x, y, z, t, sigma, rho, beta, p):
            return p * x, p * y, p * z

        @bp.sdeint(g=lorenz_g)
        def lorenz_f(x, y, z, t, sigma, rho, beta, p):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return dx, dy, dz

        self.lorenz = lorenz_f

        super(LorenzSystem, self).__init__(steps=[self.update], **kwargs)

    def update(self, _t):
        self.x, self.y, self.z = self.lorenz(self.x, self.y, self.z, _t,
                                             self.sigma, self.rho, self.beta, self.p)


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
