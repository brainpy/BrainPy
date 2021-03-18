# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numba
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


import brainpy as bp
bp.backend.set('numba')
from brainpy.integrators import sde


sigma = 10
beta = 8 / 3
rho = 28
p = 0.1


@numba.njit
def lorenz_f(x, y, z, t):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz


@numba.njit
def lorenz_g(x, y, z, t):
    return p * x, p * y, p * z


def lorenz_system(method, **kwargs):
    integral = numba.njit(method(f=lorenz_f, g=lorenz_g, show_code=True, dt=0.005,
                                 **kwargs))

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

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.plot(mon1, mon2, mon3)
    ax.set_xlabel('x')
    ax.set_xlabel('y')
    ax.set_xlabel('z')
    plt.show()


# lorenz_system(sde.srk1w1_scalar)
# lorenz_system(sde.srk2w1_scalar)
# lorenz_system(sde.euler, sde_type=bp.integrators.ITO_SDE)
# lorenz_system(sde.euler, sde_type=bp.integrators.STRA_SDE)
# lorenz_system(sde.milstein, sde_type=bp.integrators.ITO_SDE)
# lorenz_system(sde.milstein, sde_type=bp.integrators.STRA_SDE)
lorenz_system(sde.srk1_strong,
              sde_type=bp.integrators.STRA_SDE,
              wiener_type=bp.integrators.SCALAR_WIENER,
              var_type=bp.integrators.POPU_VAR)


if __name__ == '__main__':
    Axes3D
