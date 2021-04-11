# -*- coding: utf-8 -*-

import numpy as np
import numba
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from brainpy.integrators import ode
import brainpy as bp


sigma = 10
beta = 8 / 3
rho = 28


@numba.njit
def lorenz_f(x, y, z, t):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz


def lorenz_system(method, dt=0.01, tol=0.1):
    bp.backend.set('numba')

    integral = numba.njit(method(lorenz_f, show_code=True, tol=tol, adaptive=True))

    times = np.arange(0, 100, 0.01)
    mon1 = []
    mon2 = []
    mon3 = []
    mon4 = []
    x, y, z = 1, 1, 1
    for t in times:
        x, y, z, dt = integral(x, y, z, t, dt)
        mon1.append(x)
        mon2.append(y)
        mon3.append(z)
        mon4.append(dt)
    mon1 = np.array(mon1)
    mon2 = np.array(mon2)
    mon3 = np.array(mon3)
    mon4 = np.array(mon4)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.plot(mon1, mon2, mon3)
    ax.set_xlabel('x')
    ax.set_xlabel('y')
    ax.set_xlabel('z')

    fig = plt.figure()
    plt.plot(mon4)

    plt.show()


lorenz_system(ode.rkf45, dt=0.1, tol=0.001)


if __name__ == '__main__':
    Axes3D
