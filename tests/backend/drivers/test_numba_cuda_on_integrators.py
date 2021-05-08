# -*- coding: utf-8 -*-


import pytest
import brainpy as bp
from brainpy.integrators.ode import general_rk_methods
from brainpy.integrators.ode import exp_euler_method
from brainpy.integrators.ode import adaptive_rk_methods
from brainpy.integrators.sde import euler_and_milstein

bp.backend.set('numba-cuda')


def test_ode():
    def lorenz_f(x, y, z, t, sigma=10, beta=8 / 3, rho=28):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return dx, dy, dz

    for method in general_rk_methods.__all__:
        print(f'{method} method:')
        bp.odeint(f=lorenz_f, show_code=True, method=method, var_type=bp.SCALAR_VAR)
        print()

        bp.odeint(f=lorenz_f, show_code=True, method=method, var_type=bp.POPU_VAR)

        bp.odeint(f=lorenz_f, show_code=True, method=method, var_type=bp.SYSTEM_VAR)

    for method in exp_euler_method.__all__:
        print(f'{method} method:')
        bp.odeint(f=lorenz_f, show_code=True, method=method, var_type=bp.SCALAR_VAR)
        print()

        bp.odeint(f=lorenz_f, show_code=True, method=method, var_type=bp.POPU_VAR)

        with pytest.raises(bp.errors.IntegratorError):
            bp.odeint(f=lorenz_f, show_code=True, method=method, var_type=bp.SYSTEM_VAR)

    for method in adaptive_rk_methods.__all__:
        print(f'{method} method:')
        bp.odeint(f=lorenz_f, show_code=True, method=method, var_type=bp.SCALAR_VAR, adaptive=True)
        print()

        bp.odeint(f=lorenz_f, show_code=True, method=method, var_type=bp.POPU_VAR, adaptive=True)
        bp.odeint(f=lorenz_f, show_code=True, method=method, var_type=bp.SYSTEM_VAR, adaptive=True)


def test_sde():
    def lorenz_g(x, y, z, t, sigma=10, beta=8 / 3, rho=28, p=0.1):
        return p * x, p * y, p * z

    def lorenz_f(x, y, z, t, sigma=10, beta=8 / 3, rho=28, p=0.1):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return dx, dy, dz

    # for method in euler_and_milstein.__all__:
    for method in ['milstein', ]:
        for sde_type in [bp.ITO_SDE, bp.STRA_SDE]:
            print(f'{sde_type} type, {method} method:')

            bp.sdeint(f=lorenz_f, g=lorenz_g, show_code=True, method=method,
                      var_type=bp.SCALAR_VAR, sde_type=sde_type)
            print()

    for method in ['euler', 'exponential_euler']:
        sde_type = bp.ITO_SDE
        print(f'{sde_type} type, {method} method:')

        bp.sdeint(f=lorenz_f, g=lorenz_g, show_code=True, method=method,
                  var_type=bp.SCALAR_VAR, sde_type=sde_type)
        print()

    for method in ['heun']:
        sde_type = bp.STRA_SDE
        print(f'{sde_type} type, {method} method:')

        bp.sdeint(f=lorenz_f, g=lorenz_g, show_code=True, method=method,
                  var_type=bp.SCALAR_VAR, sde_type=sde_type)
        print()


# test_sde()
