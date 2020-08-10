# -*- coding: utf-8 -*-

import numpy as np
import numba as nb

from ..utils import profile
from ..utils.helper import autojit

__all__ = [
    'integrate',

    # ODE methods
    'ode_euler',
    'ode_rk2',
    'midpoint',
    'ode_heun',
    'ode_rk3',
    'ode_rk4',
    'ode_rk4_alternative',
    'ode_backward_euler',
    'trapezoidal_rule',

    'ode_exponential_euler',

    # SDE methods
    'sde_euler',
    'Milstein_dfree_Ito',
    'sde_heun',
    'Milstein_dfree_Stra',

    'sde_exponential_euler',
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
            return lambda f: wrapper(f, signature=signature)
        else:  # SDE
            return lambda f: wrapper(f, g=noise, signature=signature)
    else:
        if not has_noise:  # ODE
            return wrapper(func, signature=signature)
        else:  # SDE
            return wrapper(func, noise, signature=signature)


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
            return sde_euler
        if method == 'Ito_milstein':
            return Milstein_dfree_Ito

        if method == 'heun':
            return sde_heun
        if method == 'Stra_milstein':
            return Milstein_dfree_Stra

        if method in ['exp', 'exponential']:
            return sde_exponential_euler

        raise ValueError('Do not support SDE updater: ', method)

    else:
        if method == 'euler':
            return ode_euler

        if method in ['exp', 'exponential']:
            return ode_exponential_euler

        if method == 'heun':
            return ode_heun
        if method == 'midpoint':
            return midpoint
        if method == 'rk2':
            return ode_rk2

        if method == 'rk3':
            return ode_rk3

        if method == 'rk4':
            return ode_rk4
        if method == 'rk4_alternative':
            return ode_rk4_alternative

        if method in ['backward_Euler', 'implicit_Euler']:
            return ode_backward_euler
        if method in ['trapezoidal_rule']:
            return trapezoidal_rule

        raise ValueError('Do not support ODE updater: ', method)


def _jit(f, signature):
    # if signature and profile.jit_diff_eq == 'cfunc' and profile.is_numba_bk():
    #     f = nb.cfunc(signature, nopython=True)(f)
    # else:
    #     f = autojit(signature)(f)
    f = autojit(signature)(f)
    return f


##################################
# Numerical integration of ODE
##################################


def ode_euler(f, dt=None, signature=None):
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
    f = _jit(f, signature)

    if dt is None:
        dt = profile.get_dt()

    def int_f(y0, t, *args):
        return y0 + dt * f(y0, t, *args)

    return autojit(int_f)


def ode_rk2(f, dt=None, beta=2 / 3, signature=None):
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
    f = _jit(f, signature)
    if dt is None:
        dt = profile.get_dt()

    def int_f(y0, t, *args):
        k1 = f(y0, t, *args)
        k2 = f(y0 + beta * dt * k1, t + beta * dt, *args)
        return y0 + dt * ((1 - 1 / (2 * beta)) * k1 + 1 / (2 * beta) * k2)

    return autojit(int_f)


def ode_heun(f, dt=None, signature=None):
    """Two-stage method for ODE numerical integration.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
    dt : None, float
        Precision of numerical integration.
    signature : str, list, tuple
        The signature for numba compilation.

    Returns
    -------
    func : callable
        The one-step numerical integration function.
    """
    return ode_rk2(f, dt, beta=1., signature=signature)


def midpoint(f, dt=None, signature=None):
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
    return ode_rk2(f, dt, beta=0.5, signature=signature)


def ode_rk3(f, dt=None, signature=None):
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
    f = _jit(f, signature)
    if dt is None:
        dt = profile.get_dt()

    def int_f(y0, t, *args):
        k1 = f(y0, t, *args)
        k2 = f(y0 + dt / 2 * k1, t + dt / 2, *args)
        k3 = f(y0 - dt * k1 + 2 * dt * k2, t + dt, *args)
        return y0 + dt / 6 * (k1 + 4 * k2 + k3)

    return autojit(int_f)


def ode_rk4(f, dt=None, signature=None):
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
    f = _jit(f, signature)
    if dt is None:
        dt = profile.get_dt()

    def int_f(y0, t, *args):
        k1 = f(y0, t, *args)
        k2 = f(y0 + dt / 2 * k1, t + dt / 2, *args)
        k3 = f(y0 + dt / 2 * k2, t + dt / 2, *args)
        k4 = f(y0 + dt * k3, t + dt, *args)
        return y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return autojit(int_f)


def ode_rk4_alternative(f, dt=None, signature=None):
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
    f = _jit(f, signature)
    if dt is None:
        dt = profile.get_dt()

    def int_f(y0, t, *args):
        k1 = f(y0, t, *args)
        k2 = f(y0 + dt / 3 * k1, t + dt / 3, *args)
        k3 = f(y0 - dt / 3 * k1 + dt * k2, t + 2 * dt / 3, *args)
        k4 = f(y0 + dt * k1 - dt * k2 + dt * k3, t + dt, *args)
        return y0 + dt / 8 * (k1 + 3 * k2 + 3 * k3 + k4)

    return autojit(int_f)


def ode_backward_euler(f, dt=None, epsilon=1e-12, signature=None):
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
    f = _jit(f, signature)
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


def trapezoidal_rule(f, dt=None, epsilon=1e-12, signature=None):
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
    f = _jit(f, signature)
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


def ode_exponential_euler(f, dt=None, signature=None):
    """First order, explicit exponential Euler method for ODE integration.

    For an equation of the form

    .. math::

        y^{\\prime}=f(y), \quad y(0)=y_{0}

    its schema is given by

    .. math::

        y_{n+1}=y_{n}+h \\varphi(hA) f (y_{n})

    where :math:`A=f^{\prime}(y_{n})` and
    :math:`\\varphi(z)=\\frac{e^{z}-1}{z}`.

    Parameters
    ----------
    f : callable
        The function at the right hand of the differential equation.
        Note, the `dydt` (i.e., :math:`f`) and linear coefficient `A` (i.e.,
        :math:`f'(y0)`) must be returned in the customized function.
    dt : None, float
        Precision of numerical integration.
    signature : str, list, tuple
        The signature for Numba compilation.

    Returns
    -------
    func : callable
        The one-step numerical integration function.
    """

    f = _jit(f, signature)
    if dt is None:
        dt = profile.get_dt()

    def int_f(y0, t, *args):
        y0, A = f(y0, t, *args)
        return y0 + (np.exp(A * dt) - 1) / A * y0

    return autojit(int_f)


##################################
# Numerical integration of SDE
##################################


def sde_euler(f, g, dt=None, signature=None):
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
    dt_sqrt = np.sqrt(dt).astype(profile.ftype)
    f = _jit(f, signature)

    if callable(g):
        g = _jit(g, signature)

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


def sde_exponential_euler(f, g, dt=None, signature=None):
    """First order, explicit exponential Euler method for SDE integration.

    For an equation of the form

    .. math::

        d y=(A y+ F(y)) dt + g(y) dW(t), \\quad y(0)=y_{0}

    its schema is given by [1]_

    .. math::

        y_{n+1}=e^{\\Delta t A}(y_{n}+ g(t)\\Delta W_{n})+\\varphi(\\Delta t A) F(y_{n}) \\Delta t

    where :math:`\\varphi(z)=\\frac{e^{z}-1}{z}`.

    Parameters
    ----------
    f : callable
        The drift coefficient, the deterministic part of the SDE.
        Note, the `dydt` (i.e., :math:`f`) and linear coefficient `A` (i.e.,
        :math:`f'(y0)`) must be returned in the customized function.
    g : callable, float
        The diffusion coefficient, the stochastic part.
    dt : None, float
        Precision of numerical integration.
    signature : str, list, tuple
        The signature for Numba compilation.

    Returns
    -------
    func : callable
        The one-step numerical integration function.

    References
    ----------

    .. [1] Erdoğan, Utku, and Gabriel J. Lord. "A new class of exponential integrators for stochastic
           differential equations with multiplicative noise." arXiv preprint arXiv:1608.07096 (2016).

    """

    f = _jit(f, signature)
    if dt is None:
        dt = profile.get_dt()
    dt_sqrt = np.sqrt(dt).astype(profile.ftype)

    if callable(g):
        g = _jit(g, signature)

        def int_fg(y0, t, *args):
            y0, A = f(y0, t, *args)
            dW = np.random.normal(0.0, 1.0, y0.shape)
            dg = dt_sqrt * g(y0, t, *args) * dW
            exp = np.exp(A * dt)
            y1 = y0 + (exp - 1) / A * y0
            return y1 + exp * dg

    else:
        assert isinstance(g, (int, float, np.ndarray))

        def int_fg(y0, t, *args):
            y0, A = f(y0, t, *args)
            dW = np.random.normal(0.0, 1.0, y0.shape)
            dg = dt_sqrt * g * dW
            exp = np.exp(A * dt)
            y1 = y0 + (exp - 1) / A * y0
            return y1 + exp * dg

    return autojit(int_fg)


def Milstein_dfree_Ito(f, g, dt=None, signature=None):
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
    dt_sqrt = np.sqrt(dt).astype(profile.ftype)
    f = _jit(f, signature)

    if callable(g):
        g = _jit(g, signature)

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


def sde_heun(f, g, dt=None, signature=None):
    """Heun two-stage stochastic numerical method for Stratonovich integral.

    Use the Stratonovich Heun algorithm to integrate Stratonovich equation,
    according to paper [1]_ [2]_.

    Parameters
    ----------
    f : callable
        The drift coefficient, the deterministic part of the SDE.
    g : callable, float
        The diffusion coefficient, the stochastic part.
    dt : None, float
        Precision of numerical integration.
    signature : str, list
        The signature for Numba compilation.

    Returns
    -------
    func : callable
        The one-step numerical integration function.

    References
    ----------

    .. [1] H. Gilsing and T. Shardlow, SDELab: A package for solving stochastic differential
         equations in MATLAB, Journal of Computational and Applied Mathematics 205 (2007),
         no. 2, 1002-1018.
    .. [2] P.E. Kloeden, E. Platen, and H. Schurz, Numerical solution of SDE through computer
         experiments, Springer, 1994.
    """
    dt = profile.get_dt() if dt is None else dt
    dt_sqrt = np.sqrt(dt).astype(profile.ftype)
    f = _jit(f, signature)

    if callable(g):
        g = _jit(g, signature)

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


def Milstein_dfree_Stra(f, g, dt=None, signature=None):
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
    dt_sqrt = np.sqrt(dt).astype(profile.ftype)
    f = _jit(f, signature)

    if callable(g):
        g = _jit(g, signature)

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



def sde_rk2():
    pass


def sde_rk3():
    pass


def sde_rk4():
    pass



