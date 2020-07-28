# -*- coding: utf-8 -*-

import numpy as np

from npbrain.utils import profile
from npbrain.utils.helper import autojit

__all__ = [
    'integrate',
    'forward_Euler',
    'rk2',
    'midpoint',
    'rk3',
    'rk4',
    'rk4_alternative',
    'backward_Euler',
    'trapezoidal_rule',
    'Euler_method',
    'Milstein_dfree_Ito',
    'Heun_method',
    'Milstein_dfree_Stra',
]


def integrate(func=None, noise=None, method=None, signature=None):
    """Generate the one-step integration function for differential equations.

    Using this method, the users only need to define the right side of the equation.
    For example, for the `m` channel in the Hodgkin–Huxley neuron model

    .. math::

        \\alpha = {0.1 * (V + 40 \\over 1 - \\exp(-(V + 40) / 10)}

        \\beta = 4.0 * \\exp(-(V + 65) / 18)

        {dm \\over dt} = \\alpha * (1 - m) - \\beta * m

    Using ``NumpyBrain``, this ODE function can be written as

    >>> @integrate(method='rk4')
    >>> def int_m(m, t, V):
    >>>     alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    >>>     beta = 4.0 * np.exp(-(V + 65) / 18)
    >>>     return alpha * (1 - m) - beta * m



    Parameters
    ----------
    func : callable
        The function at the right hand of the differential equation.
        If a stochastic equation (SDE) is defined, then `func` is the drift coefficient
        (the deterministic part) of the SDE.
    noise : callable, float
        The diffusion coefficient (the stochastic part) of the SDE. `noise` can be a float
        number, or a function.
    method : None, str, callable
        The method of numerical integration.
    signature : list, str, None
        The numba compilation signature.

    Returns
    -------
    func : callable
        If `f` is provided, then the one-step numerical integration function will be returned.
        if not, the wrapper will be provided.
    """

    try:
        has_noise = not (noise is None or noise == 0.)
    except Exception:
        has_noise = True
    method = method if method is not None else profile.get_method()
    wrapper = _get_integrator(method=method, has_noise=has_noise)

    if func is None:
        if not has_noise:  # ODE
            return wrapper
        else:  # SDE
            return lambda f: wrapper(f, g=noise)
    else:
        if not has_noise:  # ODE
            return wrapper(func)
        else:  # SDE
            return wrapper(func, noise)


def _get_integrator(method=None, has_noise=False):
    """Generate the one-step ODE integration function.

    Parameters
    ----------
    method : None, str
        Method of numerical integration.
    has_noise : bool
        if `has_noise=True`, the equation is a SDE; otherwise, it is a ODE.

    Returns
    -------
    wrapper : callable
        The wrapper function.
    """

    if has_noise:
        if method == 'euler':
            return Euler_method
        if method == 'Ito_milstein':
            return Milstein_dfree_Ito

        if method == 'heun':
            return Heun_method
        if method == 'Stra_milstein':
            return Milstein_dfree_Stra

        raise ValueError('Do not support SDE updater: ', method)
    else:
        if method == 'euler':
            return forward_Euler
        if method == midpoint:
            return midpoint
        if method == 'rk2':
            return rk2
        if method == 'rk3':
            return rk3
        if method == 'rk4':
            return rk4
        if method == 'rk4_alternative':
            return rk4_alternative

        if method in ['backward_Euler', 'implicit_Euler']:
            return backward_Euler
        if method in ['trapezoidal_rule']:
            return trapezoidal_rule

        raise ValueError('Do not support ODE updater: ', method)


##################################
# Numerical integration of ODE
##################################


def forward_Euler(f, dt=None):
    """Forward Euler method. Also named as ``explicit_Euler``.

    The most unstable integrator known. Requires a very small timestep.
    Accuracy is O(dt).

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integration.

    Returns
    -------
    func : callable
        The one-step numerical integration function.
    """
    f = autojit(f)
    if dt is None:
        dt = profile.get_dt()

    def int_f(y0, t, *args):
        return y0 + dt * f(y0, t, *args)

    return autojit(int_f)


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
        Precision of numerical integration.

    Returns
    -------
    func : callable
        The one-step numerical integration function.
    """
    f = autojit(f)
    if dt is None:
        dt = profile.get_dt()

    def int_f(y0, t, *args):
        k1 = f(y0, t, *args)
        k2 = f(y0 + beta * dt * k1, t + beta * dt, *args)
        return y0 + dt * ((1 - 1 / (2 * beta)) * k1 + 1 / (2 * beta) * k2)

    return autojit(int_f)


def midpoint(f, dt=None):
    """Explicit midpoint Euler method. Also named as ``modified_Euler``.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integration.

    Returns
    -------
    func : callable
        The one-step numerical integration function.
    """
    return rk2(f, dt, beta=0.5)


def rk3(f, dt=None):
    """Kutta's third-order method (commonly known as RK3).
    Also named as ``RK3``.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integration.

    Returns
    -------
    func : callable
        The one-step numerical integration function.
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


def rk4(f, dt=None):
    """Fourth-order Runge-Kutta (RK4). Also named as ``RK4``.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integration.

    Returns
    -------
    func : callable
        The one-step numerical integration function.
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


def rk4_alternative(f, dt=None):
    """An alternative of fourth-order Runge-Kutta method.
    Also named as ``RK4_alternative``.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integration.

    Returns
    -------
    func : callable
        The one-step numerical integration function.
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


def backward_Euler(f, dt=None, epsilon=1e-12):
    """Backward Euler method. Also named as ``implicit_Euler``.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integration.

    Returns
    -------
    func : callable
        The one-step numerical integration function.
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


def trapezoidal_rule(f, dt=None, epsilon=1e-12):
    """Trapezoidal rule.

    The trapezoidal rule is an implicit second-order method, which can
    be considered as both a Runge–Kutta method and a linear multistep method.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integration.

    Returns
    -------
    func : callable
        The one-step numerical integration function.
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


def exponential_euler(f, factor_zero_order, factor_one_order, dt=None):
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
        The one-step numerical integration function.
    """

    a = np.exp(-factor_one_order * dt)
    b = factor_zero_order / factor_one_order * (1 - a)

    def int_f(y0, t, *args):
        y0 = f(y0, t, *args)
        return y0 * a + b

    return autojit(int_f)


##################################
# Numerical integration of SDE
##################################


def Euler_method(f, g, dt=None):
    """Itô stochastic integral. The simplest stochastic numerical approximation
        is the Euler-Maruyama method. Its is an order 0.5 strong Taylor schema.
        Also named as ``EM``, ``EM_method``, ``Euler``, ``Euler_Maruyama_method``.

        Parameters
        ----------
        f : callable
            The drift coefficient, the deterministic part of the SDE.
        g : callable, float
            The diffusion coefficient, the stochastic part.
        dt : None, float
            Precision of numerical integration.

        Returns
        -------
        func : callable
            The one-step numerical integration function.
        """
    dt = profile.get_dt() if dt is None else dt
    dt_sqrt = np.sqrt(dt)
    f = autojit(f)

    if callable(g):
        g = autojit(g)

        def int_fg(y0, t, *args):
            dW = np.random.normal(0.0, 1.0, y0.shape)
            df = f(y0, t, *args) * dt
            dg = dt_sqrt * g(y0, t, *args) * dW
            return y0 + df + dg
    else:
        assert isinstance(g, (int, float, np.ndarray))

        def int_fg(y0, t, *args):
            dW = np.random.normal(0.0, 1.0, y0.shape)
            df = f(y0, t, *args) * dt
            dg = dt_sqrt * g * dW
            return y0 + df + dg

    return autojit(int_fg)


def Milstein_dfree_Ito(f, g, dt=None):
    """Itô stochastic integral. The derivative-free Milstein method is
    an order 1.0 strong Taylor schema.

    Parameters
    ----------
    f : callable
        The drift coefficient, the deterministic part of the SDE.
    g : callable, float
        The diffusion coefficient, the stochastic part.
    dt : None, float
        Precision of numerical integration.

    Returns
    -------
    func : callable
        The one-step numerical integration function.
    """
    dt = profile.get_dt() if dt is None else dt
    dt_sqrt = np.sqrt(dt)
    f = autojit(f)

    if callable(g):
        g = autojit(g)

        def int_fg(y0, t, *args):
            dW = np.random.normal(0.0, 1.0, y0.shape)
            df = f(y0, t - dt, *args) * dt
            g_n = g(y0, t - dt, *args)
            dg = g_n * dW * dt_sqrt
            y_n_bar = y0 + df + g_n * dt_sqrt
            g_n_bar = g(y_n_bar, t, *args)
            y1 = y0 + df + dg + 0.5 * (g_n_bar - g_n) * (dW * dW * dt_sqrt - dt_sqrt)
            return y1
    else:
        assert isinstance(g, (int, float, np.ndarray))

        def int_fg(y0, t, *args):
            dW = np.random.normal(0.0, 1.0, y0.shape)
            df = f(y0, t - dt, *args) * dt
            dg = g * dW * dt_sqrt
            y1 = y0 + df + dg
            return y1

    return autojit(int_fg)


def Heun_method(f, g, dt=None):
    """Stratonovich stochastic integral.

    Use the Stratonovich Heun algorithm
    to integrate Stratonovich equation,
    according to paper [2]_, [3]_.

    Parameters
    ----------
    f : callable
        The drift coefficient, the deterministic part of the SDE.
    g : callable, float
        The diffusion coefficient, the stochastic part.
    dt : None, float
        Precision of numerical integration.

    Returns
    -------
    func : callable
        The one-step numerical integration function.

    References
    ----------

    .. [2] H. Gilsing and T. Shardlow, SDELab: A package for solving stochastic differential
         equations in MATLAB, Journal of Computational and Applied Mathematics 205 (2007),
         no. 2, 1002{1018.
    .. [3] P.E. Kloeden, E. Platen, and H. Schurz, Numerical solution of SDE through computer
         experiments, Springer, 1994.
    """
    dt = profile.get_dt() if dt is None else dt
    dt_sqrt = np.sqrt(dt)
    f = autojit(f)

    if callable(g):
        g = autojit(g)

        def int_fg(y0, t, *args):
            dW = np.random.normal(0.0, 1.0, y0.shape)
            df = f(y0, t - dt, *args) * dt
            gn = g(y0, t - dt, *args)
            y_bar = y0 + gn * dW * dt_sqrt
            gn_bar = g(y_bar, t, *args)
            dg = 0.5 * (gn + gn_bar) * dW * dt_sqrt
            y1 = y0 + df + dg
            return y1
    else:
        assert isinstance(g, (int, float, np.ndarray))

        def int_fg(y0, t, *args):
            dW = np.random.normal(0.0, 1.0, y0.shape)
            df = f(y0, t - dt, *args) * dt
            dg = g * dW * dt_sqrt
            y1 = y0 + df + dg
            return y1

    return autojit(int_fg)


def Milstein_dfree_Stra(f, g, dt=None):
    """Stratonovich stochastic integral. The derivative-free Milstein
    method is an order 1.0 strong Taylor schema.

    Parameters
    ----------
    f : callable
        The drift coefficient, the deterministic part of the SDE.
    g : callable, float
        The diffusion coefficient, the stochastic part.
    dt : None, float
        Precision of numerical integration.

    Returns
    -------
    func : callable
        The one-step numerical integration function.
    """
    dt = profile.get_dt() if dt is None else dt
    dt_sqrt = np.sqrt(dt)
    f = autojit(f)

    if callable(g):
        g = autojit(g)

        def int_fg(y0, t, *args):
            dW = np.random.normal(0.0, 1.0, y0.shape)
            df = f(y0, t - dt, *args) * dt
            g_n = g(y0, t - dt, *args)
            dg = g_n * dW * dt_sqrt
            y_n_bar = y0 + df + g_n * dt_sqrt
            g_n_bar = g(y_n_bar, t, *args)
            extra_term = 0.5 * (g_n_bar - g_n) * (dW * dW * dt_sqrt)
            y1 = y0 + df + dg + extra_term
            return y1
    else:
        assert isinstance(g, (int, float, np.ndarray))

        def int_fg(y0, t, *args):
            dW = np.random.normal(0.0, 1.0, y0.shape)
            df = f(y0, t - dt, *args) * dt
            dg = g * dW * dt_sqrt
            y1 = y0 + df + dg
            return y1

    return autojit(int_fg)
