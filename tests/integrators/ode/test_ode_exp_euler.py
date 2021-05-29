# -*- coding: utf-8 -*-

import pytest
import brainpy as bp
from brainpy.integrators.ode.wrapper import exp_euler_wrapper


def test1():

    import numpy as np

    def drivative(V, m, h, n, t, Iext, gNa, ENa, gK, EK, gL, EL, C):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m

        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h

        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n

        I_Na = (gNa * m ** 3.0 * h) * (V - ENa)
        I_K = (gK * n ** 4.0) * (V - EK)
        I_leak = gL * (V - EL)
        dVdt = (- I_Na - I_K - I_leak + Iext) / C

        return dVdt, dmdt, dhdt, dndt

    exp_euler_wrapper(f=drivative, show_code=True, dt=0.01, var_type='SCALAR', im_return=())


def test2():

    def derivative(s, t, tau):
        return -s / tau

    with pytest.raises(bp.errors.DiffEqError):
        exp_euler_wrapper(f=derivative, show_code=True, dt=0.01, var_type='SCALAR', im_return=())


def test3():

    def derivative(s, v, t, tau):
        dv = -v + 1
        return -s / tau, dv

    with pytest.raises(bp.errors.DiffEqError):
        exp_euler_wrapper(f=derivative, show_code=True, dt=0.01, var_type='SCALAR', im_return=())


def test4():
    f = lambda s, t, tau: -s / tau
    with pytest.raises(bp.errors.AnalyzerError) as excinfo:
        exp_euler_wrapper(f=f, show_code=True, dt=0.01, var_type='SCALAR', im_return=())
