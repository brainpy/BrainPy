# -*- coding: utf-8 -*-

import brainpy as bp
from brainpy.integrators.sde.euler_and_milstein import exponential_euler


def test1():
    p = 0.1

    def lorenz_g(x, y, z, t, sigma=10, beta=8 / 3, rho=28):
        return p * x, p * y, p * z

    def lorenz_f(x, y, z, t, sigma=10, beta=8 / 3, rho=28):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return dx, dy, dz

    exponential_euler(f=lorenz_f, g=lorenz_g, dt=0.01,
                      sde_type=bp.ITO_SDE,
                      wiener_type=bp.SCALAR_WIENER,
                      var_type=bp.POPU_VAR,
                      show_code=True)


if __name__ == '__main__':
    test1()
