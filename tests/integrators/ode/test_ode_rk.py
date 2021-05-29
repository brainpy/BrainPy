# -*- coding: utf-8 -*-

import numpy as np
from brainpy.integrators import ode
import numba
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


sigma = 10
beta = 8 / 3
rho = 28


@numba.njit
def lorenz_f(x, y, z, t):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz


def lorenz_system(method):
    integral = numba.njit(method(lorenz_f, show_code=True, dt=0.005))

    times = np.arange(0, 100, 0.01)
    mon1 = []
    mon2 = []
    mon3 = []
    x, y, z = 1, 1, 1
    for t in times:
        x, y, z = integral(x, y, z, t)
        mon1.append(x)
        mon2.append(y)
        mon3.append(z)
    mon1 = np.array(mon1)
    mon2 = np.array(mon2)
    mon3 = np.array(mon3)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # plt.plot(mon1, mon2, mon3)
    # ax.set_xlabel('x')
    # ax.set_xlabel('y')
    # ax.set_xlabel('z')
    # plt.show()


def test():
    lorenz_system(ode.rk4)
    lorenz_system(ode.ralston4)
    lorenz_system(ode.rk4_38rule)
    lorenz_system(ode.rk3)
    lorenz_system(ode.heun3)
    lorenz_system(ode.ralston3)
    lorenz_system(ode.rk2)
    lorenz_system(ode.ralston2)
    lorenz_system(ode.heun2)

