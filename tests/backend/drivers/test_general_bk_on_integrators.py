# -*- coding: utf-8 -*-


import pytest

import brainpy as bp
from brainpy.integrators.ode import adaptive_rk_methods
from brainpy.integrators.ode import exp_euler_method
from brainpy.integrators.ode import general_rk_methods

bp.backend.set('numpy')


def test_ode():
    def lorenz_f(x, y, z, t, sigma=10, beta=8 / 3, rho=28):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return dx, dy, dz

    for method in general_rk_methods.__all__ + exp_euler_method.__all__:
        for var_type in bp.SUPPORTED_VAR_TYPE:
            print(f'"{method}" method, "{var_type}" var type:')
            if method == 'exponential_euler' and var_type == bp.SYSTEM_VAR:
                with pytest.raises(bp.errors.IntegratorError):
                    bp.odeint(f=lorenz_f, show_code=True, method=method, var_type=var_type)
            else:
                bp.odeint(f=lorenz_f, show_code=True, method=method, var_type=var_type)
            print()

    for method in adaptive_rk_methods.__all__:
        for var_type in bp.SUPPORTED_VAR_TYPE:
            for adaptive in [True, False]:
                print(f'"{method}" method, "{var_type}" var type, adaptive = {adaptive}:')
                bp.odeint(f=lorenz_f, show_code=True, method=method, var_type=var_type, adaptive=adaptive)
                print()


def test_sde():
    def lorenz_g(x, y, z, t, sigma=10, beta=8 / 3, rho=28, p=0.1):
        return p * x, p * y, p * z

    def lorenz_f(x, y, z, t, sigma=10, beta=8 / 3, rho=28, p=0.1):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return dx, dy, dz

    # for method in euler_and_milstein.__all__:
    for method in ['euler', 'milstein']:
        for sde_type in bp.SUPPORTED_SDE_TYPE:
            if method == 'heun' and sde_type == bp.ITO_SDE:
                continue
            for var_type in bp.SUPPORTED_VAR_TYPE:
                for wiener_type in bp.SUPPORTED_WIENER_TYPE:
                    print(f'"{method}" method, "{sde_type}" sde type, '
                          f'"{var_type}" var type, "{wiener_type}" wiener type:')
                    bp.sdeint(f=lorenz_f, g=lorenz_g, show_code=True, method=method,
                              var_type=var_type,
                              sde_type=sde_type,
                              wiener_type=wiener_type)
                    print()

    for method in ['exponential_euler', 'srk1w1_scalar', 'srk2w1_scalar', 'KlPl_scalar']:
        for var_type in bp.SUPPORTED_VAR_TYPE:
            for wiener_type in bp.SUPPORTED_WIENER_TYPE:
                with pytest.raises(bp.errors.IntegratorError):
                    bp.sdeint(f=lorenz_f, g=lorenz_g, show_code=True, method=method,
                              var_type=var_type,
                              sde_type=bp.STRA_SDE,
                              wiener_type=wiener_type)

    # for method in ['exponential_euler', 'srk1w1_scalar', 'srk2w1_scalar', 'KlPl_scalar']:



test_sde()
