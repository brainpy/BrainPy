# -*- coding: utf-8 -*-

import numpy as np

from npbrain.helper import autojit
from npbrain import profile

__all__ = [
    'ode_generator',
    'forward_Euler', 'explicit_Euler',
    'exp_euler_o2',
    'rk2', 'RK2',
    'modified_Euler', 'explicit_midpoint_Euler',
    'rk3', 'RK3',
    'rk4', 'RK4',
    'RK4_alternative', 'rk4_alternative',
    'backward_Euler', 'implicit_Euler',
    'trapezoidal_rule',
]


def ode_generator(f=None, method=None):
    """Generate the one-step ODE integrator function.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    method : None, str, callable
        Method of numerical integrator.

    Returns
    -------
    func : callable
        If `f` is provided, then the one-step numerical integrator function will be returned.
        if not, the wrapper will be provided.
    """

    wrapper = _get_generator(method)
    if f is not None:
        dt = profile.get_dt()
        return wrapper(f, dt)
    else:
        return wrapper


def _get_generator(method=None):
    """Generate the one-step ODE integrator function.

    Parameters
    ----------
    method : None, str, callable
        Method of numerical integrator.

    Returns
    -------
    wrapper : callable
        The wrapper function.
    """

    if method in ['euler', 'forward_Euler', 'explicit_Euler']:
        return forward_Euler
    if method in ['rk2', 'RK2']:
        return rk2
    if method in ['explicit_midpoint_Euler', 'modified_Euler']:
        return modified_Euler
    if method in ['rk3', 'RK3']:
        return rk3
    if method in ['rk4', 'RK4']:
        return rk4
    if method in ['RK4_alternative', 'rk4_alternative']:
        return rk4_alternative

    if method in ['backward_Euler', 'implicit_Euler']:
        return implicit_Euler
    if method in ['trapezoidal_rule']:
        return trapezoidal_rule

    raise ValueError('Unknown method: ', method)


# euler-start
def forward_Euler(f, dt=None):
    """Forward Euler method. Also named as ``explicit_Euler``.

    The most unstable integrator known. Requires a very small timestep.
    Accuracy is O(dt).

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integrator.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.
    """
    f = autojit(f)
    if dt is None:
        dt = profile.get_dt()

    def int_f(y0, t, *args):
        return y0 + dt * f(y0, t, *args)

    return autojit(int_f)


# euler-end


explicit_Euler = forward_Euler


# rk2-start
def rk2(f, dt=None, beta=2 / 3):
    """Parametric second-order Runge-Kutta (RK2).
    Also named as ``RK2``.

    Popular choices for 'beta':
        1/2 :
            explicit midpoint method
        2/3 :
            Ralston's method
        1 :
            Heun's method, also known as the explicit trapezoid rule

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integrator.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.
    """
    f = autojit(f)
    if dt is None:
        dt = profile.get_dt()

    def int_f(y0, t, *args):
        k1 = f(y0, t, *args)
        k2 = f(y0 + beta * dt * k1, t + beta * dt, *args)
        return y0 + dt * ((1 - 1 / (2 * beta)) * k1 + 1 / (2 * beta) * k2)

    return autojit(int_f)


# rk2-end


RK2 = rk2


# modified_Euler-start
def explicit_midpoint_Euler(f, dt=None):
    """Explicit midpoint Euler method. Also named as ``modified_Euler``.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integrator.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.
    """
    return rk2(f, dt, beta=0.5)


# modified_Euler-end


modified_Euler = explicit_midpoint_Euler


# rk3-start
def rk3(f, dt=None):
    """Kutta's third-order method (commonly known as RK3).
    Also named as ``RK3``.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integrator.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.
    """
    f = autojit(f)
    if dt is None:
        dt = profile.get_dt()

    def int_f(y0, t, *args):
        k1 = f(y0, t, *args)
        k2 = f(y0 + dt / 2 * k1, t + dt / 2, *args)
        k3 = f(y0 - dt * k1 + 2 * dt * k2, t + dt, *args)
        return y0 + dt / 6 * (k1 + 4 * k2 + k3)

    return autojit(int_f)


# rk3-end


RK3 = rk3


# rk4-start
def rk4(f, dt=None):
    """Fourth-order Runge-Kutta (RK4). Also named as ``RK4``.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integrator.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.
    """
    f = autojit(f)
    if dt is None:
        dt = profile.get_dt()

    def int_f(y0, t, *args):
        k1 = f(y0, t, *args)
        k2 = f(y0 + dt / 2 * k1, t + dt / 2, *args)
        k3 = f(y0 + dt / 2 * k2, t + dt / 2, *args)
        k4 = f(y0 + dt * k3, t + dt, *args)
        return y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return autojit(int_f)


# rk4-end


RK4 = rk4


# rk4_alternative-start
def rk4_alternative(f, dt=None):
    """An alternative of fourth-order Runge-Kutta method.
    Also named as ``RK4_alternative``.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integrator.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.
    """
    f = autojit(f)
    if dt is None:
        dt = profile.get_dt()

    def int_f(y0, t, *args):
        k1 = f(y0, t, *args)
        k2 = f(y0 + dt / 3 * k1, t + dt / 3, *args)
        k3 = f(y0 - dt / 3 * k1 + dt * k2, t + 2 * dt / 3, *args)
        k4 = f(y0 + dt * k1 - dt * k2 + dt * k3, t + dt, *args)
        return y0 + dt / 8 * (k1 + 3 * k2 + 3 * k3 + k4)

    return autojit(int_f)


# rk4_alternative-end


RK4_alternative = rk4_alternative


# backward_Euler-start
def backward_Euler(f, dt=None, epsilon=1e-12):
    """Backward Euler method. Also named as ``implicit_Euler``.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integrator.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.
    """
    f = autojit(f)
    if dt is None:
        dt = profile.get_dt()

    def int_f(y0, t, *args):
        y1 = y0 + dt * f(y0, t, *args)
        y2 = y0 + dt * f(y1, t, *args)
        while not np.all(np.abs(y1 - y2) < epsilon):
            y1 = y2
            y2 = y0 + dt * f(y1, t, *args)
        return y2

    return autojit(int_f)


# backward_Euler-end


implicit_Euler = backward_Euler


# trapezoidal_rule-start
def trapezoidal_rule(f, dt=None, epsilon=1e-12):
    """Trapezoidal rule.

    The trapezoidal rule is an implicit second-order method, which can
    be considered as both a Runge–Kutta method and a linear multistep method.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integrator.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.
    """
    f = autojit(f)
    if dt is None:
        dt = profile.get_dt()

    def int_f(y0, t, *args):
        dy0 = f(y0, t, *args)
        y1 = y0 + dt * dy0
        y2 = y0 + dt / 2 * (dy0 + f(y1, t + dt, *args))
        while not np.all(np.abs(y1 - y2) < epsilon):
            y1 = y2
            y2 = y0 + dt / 2 * (dy0 + f(y1, t + dt, *args))
        return y2

    return autojit(int_f)


# trapezoidal_rule-end


def exp_euler_o2(f, dt, factor_zero_order, factor_one_order):
    """Order 2 Exponential Euler method.

    For an equation of the form

    .. math:

        y^{\\prime}=f(y), \quad y(0)=y_{0}

    its schema is given by

    .. math:

        y_{n+1}=y_{n}+h \\varphi(hA) f (y_{n})

    where :math::`A=f^{\prime}(y_{n})` and
    :math::`\\varphi(z)=\\frac{e^{z}-1}{z}`.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
    factor_zero_order : int, float
        The factor of the zero order function in the equation.
    factor_one_order : int, float
        The factor of the one order function in the equation.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.
    """

    a = np.exp(-factor_one_order * dt)
    b = factor_zero_order / factor_one_order * (1 - a)

    def int_f(y0, t, *args):
        y0 = f(y0, t, *args)
        return y0 * a + b

    return autojit(int_f)
