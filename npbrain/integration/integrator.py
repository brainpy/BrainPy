# -*- coding: utf-8 -*-

import sympy
from abc import ABCMeta
from abc import abstractmethod

from .diff_equation import DiffEquation
from .diff_equation import _ODE_TYPE
from .diff_equation import _SDE_TYPE
from .sympy_tools import str_to_sympy
from .sympy_tools import sympy_to_str
from .. import _numpy as bnp
from .. tools import word_substitute
from .. import profile

__all__ = [
    'get_integrator',
    'Integrator',
]


def get_integrator(method):
    if method.lower() == 'euler':
        return Euler
    elif method.lower() == 'midpoint':
        return MidPoint
    elif method.lower() == 'heun':
        return Heun
    elif method.lower() == 'rk2':
        return RK2
    elif method.lower() == 'rk3':
        return RK3
    elif method.lower() == 'rk4':
        return RK4
    elif method.lower() == 'exponential':
        return ExponentialEuler
    elif method.lower() == 'milstein':
        return MilsteinIto
    elif method.lower() == 'milstein_ito':
        return MilsteinIto
    elif method.lower() == 'milstein_stra':
        return MilsteinStra
    else:
        raise ValueError(f'Unknown method: {method}.')


class IntegrationError(Exception):
    pass


class Integrator(metaclass=ABCMeta):
    def __init__(self, diff_eqs):
        self.diff_eqs = diff_eqs
        self._np_step = None
        self._update_code = None

    def __call__(self, y0, t, *args):
        """Function in Numpy mode."""
        return self._np_step(y0, t, *args)

    @abstractmethod
    def get_np_step(self):
        pass

    @abstractmethod
    def get_nb_step(self):
        pass

    def argument_substitute(self, code):
        subs_dict = {arg: f'_{self.py_func_name}_{arg}' for arg in self.diff_eqs.func_args}
        code = word_substitute(code, subs_dict)
        return code

    @property
    def py_func(self):
        return self.diff_eqs.f

    @property
    def py_func_name(self):
        return self.diff_eqs.f.__name__

    @property
    def update_code(self):
        return self._update_code

    @property
    def code_scope(self):
        return self.diff_eqs.func_scope


class Exact(Integrator):
    def __init__(self, diff_eqs):
        super(Exact, self).__init__(diff_eqs)

    def get_np_step(self):
        raise IntegrationError(f'{self.__class__.__name__} does not support numpy backend.')

    def get_nb_step(self):
        pass


class Euler(Integrator):
    """Forward Euler method. Also named as ``explicit_Euler``.

    The simplest way for solving ordinary differential equations is "the
    Euler method" by Press et al. (1992) [1]_ :

    .. math::

        y_{n+1} = y_n + f(y_n, t_n) \\Delta t

    This formula advances a solution from :math:`y_n` to :math:`y_{n+1}=y_n+h`.
    Note that the method increments a solution through an interval :math:`h`
    while using derivative information from only the beginning of the interval.
    As a result, the step's error is :math:`O(h^2)`.

    For SDE equations, this approximation is a continuous time stochastic process that
    satisfy the iterative scheme [1]_.

    .. math::

        Y_{n+1} = Y_n + f(Y_n)h_n + g(Y_n)\\Delta W_n

    where :math:`n=0,1, \\cdots , N-1`, :math:`Y_0=x_0`, :math:`Y_n = Y(t_n)`,
    :math:`h_n = t_{n+1} - t_n` is the step size,
    :math:`\\Delta W_n = [W(t_{n+1}) - W(t_n)] \\sim N(0, h_n)=\\sqrt{h}N(0, 1)`
    with :math:`W(t_0) = 0`.

    For simplicity, we rewrite the above equation into

    .. math::

        Y_{n+1} = Y_n + f_n h + g_n \\Delta W_n

    As the order of convergence for the Euler-Maruyama method is low (strong order of
    convergence 0.5, weak order of convergence 1), the numerical results are inaccurate
    unless a small step size is used. By adding one more term from the stochastic
    Taylor expansion, one obtains a 1.0 strong order of convergence scheme known
    as *Milstein scheme* [2]_.

    Parameters
    ----------
    diff_eqs : DiffEquation
        The differential equation.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.

    References
    ----------
    .. [1] W. H.; Flannery, B. P.; Teukolsky, S. A.; and Vetterling,
            W. T. Numerical Recipes in FORTRAN: The Art of Scientific
            Computing, 2nd ed. Cambridge, England: Cambridge University
            Press, p. 710, 1992.
    .. [2] U. Picchini, Sde toolbox: Simulation and estimation of stochastic
           differential equations with matlab.
    """

    def __init__(self, diff_eqs):
        super(Euler, self).__init__(diff_eqs)

        if profile.is_numpy_bk():
            self.get_np_step()
        elif profile.is_numba_bk():
            self.get_nb_step()
        else:
            raise ValueError

    def get_np_step(self):
        # numpy mode
        # -----------
        dt = profile.get_dt()
        f = self.py_func

        # ODE
        if self.diff_eqs.type == _ODE_TYPE:
            if self.diff_eqs.is_multi_return:
                def int_f(y0, t, *args):
                    val = f(y0, t, *args)
                    y = y0 + dt * val[0]
                    return (y,) + tuple(val[1:])
            else:
                def int_f(y0, t, *args):
                    return y0 + dt * f(y0, t, *args)

        # SDE
        else:
            g = self.diff_eqs.g
            dt_sqrt = bnp.sqrt(dt)

            if callable(g):
                if self.diff_eqs.is_multi_return:
                    def int_f(y0, t, *args):
                        dW = bnp.random.normal(0.0, 1.0, y0.shape)
                        val = f(y0, t, *args)
                        df = val[0] * dt
                        dg = dt_sqrt * g(y0, t, *args) * dW
                        y = y0 + df + dg
                        return (y,) + tuple(val[1:])

                else:
                    def int_f(y0, t, *args):
                        dW = bnp.random.normal(0.0, 1.0, y0.shape)
                        df = f(y0, t, *args) * dt
                        dg = dt_sqrt * g(y0, t, *args) * dW
                        return y0 + df + dg
            else:
                assert isinstance(g, (int, float, bnp.ndarray))

                if self.diff_eqs.is_multi_return:
                    def int_f(y0, t, *args):
                        val = f(y0, t, *args)
                        df = val[0] * dt
                        dW = bnp.random.normal(0.0, 1.0, y0.shape)
                        dg = dt_sqrt * g * dW
                        y = y0 + df + dg
                        return (y,) + val[1:]

                else:
                    def int_f(y0, t, *args):
                        df = f(y0, t, *args) * dt
                        dW = bnp.random.normal(0.0, 1.0, y0.shape)
                        dg = dt_sqrt * g * dW
                        return y0 + df + dg

        self._np_step = int_f

    def get_nb_step(self):
        pass


class RK2(Integrator):
    """Parametric second-order Runge-Kutta (RK2).
    Also named as ``RK2``.

    It is given in parametric form by [1]_ .

    .. math::

        k_1	&=	f(y_n, t_n)  \\\\
        k_2	&=	f(y_n + \\beta \\Delta t k_1, t_n + \\beta \\Delta t) \\\\
        y_{n+1} &= y_n + \\Delta t [(1-\\frac{1}{2\\beta})k_1+\\frac{1}{2\\beta}k_2]

    Parameters
    ----------
    diff_eqs : DiffEquation
        The differential equation.
    beta : float
        Popular choices for 'beta':
        1/2 :
            explicit midpoint method
        2/3 :
            Ralston's method
        1 :
            Heun's method, also known as the explicit trapezoid rule

    Returns
    -------
    func : callable
        The one-step numerical integrator function.

    References
    ----------
    .. [1] https://lpsa.swarthmore.edu/NumInt/NumIntSecond.html

    See Also
    --------
    Heun, MidPoint
    """

    def __init__(self, diff_eqs, beta=2 / 3):
        super(RK2, self).__init__(diff_eqs)
        self.beta = beta

        if profile.is_numpy_bk():
            self.get_np_step()

    def get_np_step(self):

        # numpy mode
        # -----------
        dt = profile.get_dt()
        f = self.diff_eqs.f

        if self.diff_eqs.type == _ODE_TYPE:
            if self.diff_eqs.is_multi_return:
                def int_f(y0, t, *args):
                    val = f(y0, t, *args)
                    k1 = val[0]
                    k2 = f(y0 + self.beta * dt * k1, t + self.beta * dt, *args)[0]
                    y = y0 + dt * ((1 - 1 / (2 * self.beta)) * k1 + 1 / (2 * self.beta) * k2)
                    return (y,) + tuple(val[1:])
            else:
                def int_f(y0, t, *args):
                    k1 = f(y0, t, *args)
                    k2 = f(y0 + self.beta * dt * k1, t + self.beta * dt, *args)
                    y = y0 + dt * ((1 - 1 / (2 * self.beta)) * k1 + 1 / (2 * self.beta) * k2)
                    return y
        else:
            raise NotImplementedError

        self._np_step = int_f


class Heun(Integrator):
    """Two-stage method for numerical integrator.

    For ODE, please see "RK2".

    For stochastic Stratonovich integral, the Heun algorithm is given by,
    according to paper [1]_ [2]_.

    .. math::
        Y_{n+1} &= Y_n + f_n h + {1 \\over 2}[g_n + g(\\overline{Y}_n)] \\Delta W_n

        \\overline{Y}_n &= Y_n + g_n \\Delta W_n


    Or, it is written as [22]_

    .. math::

        Y_1 &= y_n + f(y_n)h + g_n \\Delta W_n

        y_{n+1} &= y_n + {1 \over 2}[f(y_n) + f(Y_1)]h + {1 \\over 2} [g(y_n) + g(Y_1)] \\Delta W_n

    Parameters
    ----------
    diff_eqs : DiffEquation
        The differential equation.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.

    References
    ----------
    .. [1] H. Gilsing and T. Shardlow, SDELab: A package for solving stochastic differential
         equations in MATLAB, Journal of Computational and Applied Mathematics 205 (2007),
         no. 2, 1002-1018.
    .. [2] P.E. Kloeden, E. Platen, and H. Schurz, Numerical solution of SDE through computer
         experiments, Springer, 1994.

    See Also
    --------
    RK2, MidPoint, MilsteinStra
    """

    def __init__(self, diff_eqs):
        super(Heun, self).__init__(diff_eqs)

        if profile.is_numpy_bk():
            self.get_np_step()

    def get_np_step(self):

        # numpy mode
        # -----------
        dt = profile.get_dt()
        f = self.diff_eqs.f

        if self.diff_eqs.type == _ODE_TYPE:
            if self.diff_eqs.is_multi_return:
                def int_f(y0, t, *args):
                    val = f(y0, t, *args)
                    k1 = val[0]
                    k2 = f(y0 + dt * k1, t + dt, *args)[0]
                    y = y0 + dt * (0.5 * k1 + 0.5 * k2)
                    return (y,) + tuple(val[1:])
            else:
                def int_f(y0, t, *args):
                    k1 = f(y0, t, *args)
                    k2 = f(y0 + dt * k1, t + dt, *args)
                    y = y0 + dt * (0.5 * k1 + 0.5 * k2)
                    return y
        else:
            g = self.diff_eqs.g
            dt_sqrt = bnp.sqrt(dt)

            if callable(g):
                if self.diff_eqs.is_multi_return:
                    def int_f(y0, t, *args):
                        dW = bnp.random.normal(0.0, 1.0, y0.shape)
                        val = f(y0, t, *args)
                        df = val[0] * dt
                        gn = g(y0, t, *args)
                        y_bar = y0 + gn * dW * dt_sqrt
                        gn_bar = g(y_bar, t, *args)
                        dg = 0.5 * (gn + gn_bar) * dW * dt_sqrt
                        y1 = y0 + df + dg
                        return (y1,) + tuple(val[1:])

                else:
                    def int_f(y0, t, *args):
                        dW = bnp.random.normal(0.0, 1.0, y0.shape)
                        df = f(y0, t, *args) * dt
                        gn = g(y0, t, *args)
                        y_bar = y0 + gn * dW * dt_sqrt
                        gn_bar = g(y_bar, t, *args)
                        dg = 0.5 * (gn + gn_bar) * dW * dt_sqrt
                        y1 = y0 + df + dg
                        return y1
            else:

                if self.diff_eqs.is_multi_return:
                    def int_f(y0, t, *args):
                        dW = bnp.random.normal(0.0, 1.0, y0.shape)
                        val = f(y0, t, *args)
                        df = val[0] * dt
                        dg = g * dW * dt_sqrt
                        y1 = y0 + df + dg
                        return (y1,) + tuple(val[1:])

                else:
                    def int_f(y0, t, *args):
                        dW = bnp.random.normal(0.0, 1.0, y0.shape)
                        df = f(y0, t, *args) * dt
                        dg = g * dW * dt_sqrt
                        y1 = y0 + df + dg
                        return y1

        self._np_step = int_f


class MidPoint(Integrator):
    """Explicit midpoint Euler method. Also named as ``modified_Euler``.

    Parameters
    ----------
    diff_eqs : DiffEquation
        The differential equation.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.

    See Also
    --------
    RK2, Heun
    """

    def __init__(self, diff_eqs):
        super(MidPoint, self).__init__(diff_eqs)

        if profile.is_numpy_bk():
            self.get_np_step()

    def get_np_step(self):

        # numpy mode
        # -----------
        dt = profile.get_dt()
        f = self.diff_eqs.f

        if self.diff_eqs.type == _ODE_TYPE:
            if self.diff_eqs.is_multi_return:
                def int_f(y0, t, *args):
                    val = f(y0, t, *args)
                    k1 = val[0]
                    k2 = f(y0 + 0.5 * dt * k1, t + 0.5 * dt, *args)[0]
                    y = y0 + dt * k2
                    return (y,) + tuple(val[1:])
            else:
                def int_f(y0, t, *args):
                    k1 = f(y0, t, *args)
                    k2 = f(y0 + 0.5 * dt * k1, t + 0.5 * dt, *args)
                    y = y0 + dt * k2
                    return y
        else:
            raise NotImplementedError

        self._np_step = int_f


class RK3(Integrator):
    """Kutta's third-order method (commonly known as RK3).
    Also named as ``RK3`` [1]_ [2]_ [3]_ .

    .. math::

        k_1 &= f(y_n, t_n) \\\\
        k_2 &= f(y_n + \\frac{\\Delta t}{2}k_1, tn+\\frac{\\Delta t}{2}) \\\\
        k_3 &= f(y_n -\\Delta t k_1 + 2\\Delta t k_2, t_n + \\Delta t) \\\\
        y_{n+1} &= y_{n} + \\frac{\\Delta t}{6}(k_1 + 4k_2+k_3)

    Parameters
    ----------
    diff_eqs : DiffEquation
        The differential equation.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.

    References
    ----------
    .. [1] http://mathworld.wolfram.com/Runge-KuttaMethod.html
    .. [2] https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    .. [3] https://zh.wikipedia.org/wiki/龙格－库塔法

    """

    def __init__(self, diff_eqs):
        super(RK3, self).__init__(diff_eqs)

        if profile.is_numpy_bk():
            self.get_np_step()

    def get_np_step(self):

        # numpy mode
        # -----------
        dt = profile.get_dt()
        f = self.diff_eqs.f

        if self.diff_eqs.type == _ODE_TYPE:
            if self.diff_eqs.is_multi_return:
                def int_f(y0, t, *args):
                    val = f(y0, t, *args)
                    k1 = val[0]
                    k2 = f(y0 + dt / 2 * k1, t + dt / 2, *args)[0]
                    k3 = f(y0 - dt * k1 + 2 * dt * k2, t + dt, *args)[0]
                    y = y0 + dt / 6 * (k1 + 4 * k2 + k3)
                    return (y,) + tuple(val[1:])

            else:
                def int_f(y0, t, *args):
                    k1 = f(y0, t, *args)
                    k2 = f(y0 + dt / 2 * k1, t + dt / 2, *args)
                    k3 = f(y0 - dt * k1 + 2 * dt * k2, t + dt, *args)
                    return y0 + dt / 6 * (k1 + 4 * k2 + k3)
        else:
            raise NotImplementedError

        self._np_step = int_f


class RK4(Integrator):
    """Fourth-order Runge-Kutta (RK4) [1]_ [2]_ [3]_ .

    .. math::

        k_1 &= f(y_n, t_n) \\\\
        k_2 &= f(y_n + \\frac{\\Delta t}{2}k_1, t_n + \\frac{\\Delta t}{2}) \\\\
        k_3 &= f(y_n + \\frac{\\Delta t}{2}k_2, t_n + \\frac{\\Delta t}{2}) \\\\
        k_4 &= f(y_n + \\Delta t k_3, t_n + \\Delta t) \\\\
        y_{n+1} &= y_n + \\frac{\\Delta t}{6}(k_1 + 2*k_2 + 2* k_3 + k_4)

    Parameters
    ----------
    diff_eqs : DiffEquation
        The differential equation.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.

    References
    ----------
    .. [1] http://mathworld.wolfram.com/Runge-KuttaMethod.html
    .. [2] https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    .. [3] https://zh.wikipedia.org/wiki/龙格－库塔法

    """

    def __init__(self, diff_eqs):
        super(RK4, self).__init__(diff_eqs)

        if profile.is_numpy_bk():
            self.get_np_step()

    def get_np_step(self):

        # numpy mode
        # -----------
        dt = profile.get_dt()
        f = self.diff_eqs.f

        if self.diff_eqs.type == _ODE_TYPE:
            if self.diff_eqs.is_multi_return:
                def int_f(y0, t, *args):
                    val = f(y0, t, *args)
                    k1 = val[0]
                    k2 = f(y0 + dt / 2 * k1, t + dt / 2, *args)[0]
                    k3 = f(y0 + dt / 2 * k2, t + dt / 2, *args)[0]
                    k4 = f(y0 + dt * k3, t + dt, *args)[0]
                    y = y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                    return (y,) + tuple(val[1:])

            else:
                def int_f(y0, t, *args):
                    k1 = f(y0, t, *args)
                    k2 = f(y0 + dt / 2 * k1, t + dt / 2, *args)
                    k3 = f(y0 + dt / 2 * k2, t + dt / 2, *args)
                    k4 = f(y0 + dt * k3, t + dt, *args)
                    return y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise NotImplementedError

        self._np_step = int_f


class RK4Alternative(Integrator):
    """An alternative of fourth-order Runge-Kutta method.
    Also named as ``RK4_alternative`` ("3/8" rule).

    It is a less often used fourth-order
    explicit RK method, and was also proposed by Kutta [1]_:

    .. math::

        k_1 &= f(y_n, t_n) \\\\
        k_2 &= f(y_n + \\frac{\\Delta t}{3}k_1, t_n + \\frac{\\Delta t}{3}) \\\\
        k_3 &= f(y_n - \\frac{\\Delta t}{3}k_1 + \\Delta t k_2, t_n + \\frac{2 \\Delta t}{3}) \\\\
        k_4 &= f(y_n + \\Delta t k_1 - \\Delta t k_2 + \\Delta t k_3, t_n + \\Delta t) \\\\
        y_{n+1} &= y_n + \\frac{\\Delta t}{8}(k_1 + 3*k_2 + 3* k_3 + k_4)

    Parameters
    ----------
    diff_eqs : DiffEquation
        The differential equation.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    """

    def __init__(self, diff_eqs):
        super(RK4Alternative, self).__init__(diff_eqs)

        if profile.is_numpy_bk():
            self.get_np_step()

    def get_np_step(self):

        # numpy mode
        # -----------
        dt = profile.get_dt()
        f = self.diff_eqs.f

        if self.diff_eqs.type == _ODE_TYPE:
            if self.diff_eqs.is_multi_return:
                def int_f(y0, t, *args):
                    val = f(y0, t, *args)
                    k1 = val[0]
                    k2 = f(y0 + dt / 3 * k1, t + dt / 3, *args)[0]
                    k3 = f(y0 - dt / 3 * k1 + dt * k2, t + 2 * dt / 3, *args)[0]
                    k4 = f(y0 + dt * k1 - dt * k2 + dt * k3, t + dt, *args)[0]
                    y = y0 + dt / 8 * (k1 + 3 * k2 + 3 * k3 + k4)
                    return (y,) + tuple(val[1:])

            else:
                def int_f(y0, t, *args):
                    k1 = f(y0, t, *args)
                    k2 = f(y0 + dt / 3 * k1, t + dt / 3, *args)
                    k3 = f(y0 - dt / 3 * k1 + dt * k2, t + 2 * dt / 3, *args)
                    k4 = f(y0 + dt * k1 - dt * k2 + dt * k3, t + dt, *args)
                    return y0 + dt / 8 * (k1 + 3 * k2 + 3 * k3 + k4)
        else:
            raise NotImplementedError

        self._np_step = int_f


class ExponentialEuler(Integrator):
    """First order, explicit exponential Euler method.

    For an ODE equation of the form

    .. math::

        y^{\\prime}=f(y), \quad y(0)=y_{0}

    its schema is given by

    .. math::

        y_{n+1}= y_{n}+h \\varphi(hA) f (y_{n})

    where :math:`A=f^{\prime}(y_{n})` and :math:`\\varphi(z)=\\frac{e^{z}-1}{z}`.

    For linear ODE system: :math:`y^{\\prime} = Ay + B`,
    the above equation is equal to

    .. math::

        y_{n+1}= y_{n}e^{hA}-B/A(1-e^{hA})

    For a SDE equation of the form

    .. math::

        d y=(Ay+ F(y))dt + g(y)dW(t) = f(y)dt + g(y)dW(t), \\quad y(0)=y_{0}

    its schema is given by [1]_

    .. math::

        y_{n+1} & =e^{\\Delta t A}(y_{n}+ g(y_n)\\Delta W_{n})+\\varphi(\\Delta t A) F(y_{n}) \\Delta t \\\\
         &= y_n + \\Delta t \\varphi(\\Delta t A) f(y) + e^{\\Delta t A}g(y_n)\\Delta W_{n}

    where :math:`\\varphi(z)=\\frac{e^{z}-1}{z}`.

    Parameters
    ----------
    diff_eqs : DiffEquation
        The differential equation.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.
    """

    def __init__(self, diff_eqs):
        super(ExponentialEuler, self).__init__(diff_eqs)

        if profile.is_numpy_bk():
            self.get_np_step()
        elif profile.is_numba_bk():
            self.get_nb_step()
        else:
            raise NotImplementedError

    def get_np_step(self):

        # numpy mode
        # -----------
        dt = profile.get_dt()
        f = self.diff_eqs.f

        if self.diff_eqs.type == _ODE_TYPE:
            if self.diff_eqs.is_multi_return:
                def int_f(y0, t, *args):
                    val = f(y0, t, *args)
                    df, linear_part = val[0], val[1]
                    y = y0 + (bnp.exp(linear_part * dt) - 1) / linear_part * df
                    return (y,) + tuple(val[2:])

            else:
                def int_f(y0, t, *args):
                    df, linear_part = f(y0, t, *args)
                    y = y0 + (bnp.exp(linear_part * dt) - 1) / linear_part * df
                    return y
        else:
            g = self.diff_eqs.g
            dt_sqrt = bnp.sqrt(dt)

            if callable(g):
                if self.diff_eqs.is_multi_return:
                    def int_f(y0, t, *args):
                        val = f(y0, t, *args)
                        dydt, linear_part = val[0], val[1]
                        dW = bnp.random.normal(0.0, 1.0, y0.shape)
                        dg = dt_sqrt * g(y0, t, *args) * dW
                        exp = bnp.exp(linear_part * dt)
                        y1 = y0 + (exp - 1) / linear_part * dydt + exp * dg
                        return (y1,) + val[2:]

                else:
                    def int_f(y0, t, *args):
                        dydt, linear_part = f(y0, t, *args)
                        dW = bnp.random.normal(0.0, 1.0, y0.shape)
                        dg = dt_sqrt * g(y0, t, *args) * dW
                        exp = bnp.exp(linear_part * dt)
                        y1 = y0 + (exp - 1) / linear_part * dydt + exp * dg
                        return y1

            else:
                assert isinstance(g, (int, float, bnp.ndarray))

                if self.diff_eqs.is_multi_return:
                    def int_f(y0, t, *args):
                        val = f(y0, t, *args)
                        dydt, linear_part = val[0], val[1]
                        dW = bnp.random.normal(0.0, 1.0, y0.shape)
                        dg = dt_sqrt * g * dW
                        exp = bnp.exp(linear_part * dt)
                        y1 = y0 + (exp - 1) / linear_part * dydt + exp * dg
                        return (y1,) + val[1:]

                else:
                    def int_f(y0, t, *args):
                        dydt, linear_part = f(y0, t, *args)
                        dW = bnp.random.normal(0.0, 1.0, y0.shape)
                        dg = dt_sqrt * g * dW
                        exp = bnp.exp(linear_part * dt)
                        y1 = y0 + (exp - 1) / linear_part * dydt + exp * dg
                        return y1

        self._np_step = int_f

    def get_nb_step(self):
        dt = profile.get_dt()
        sub_exprs = self.diff_eqs.substitute()
        code_lines = []
        for k, v in list(sub_exprs.items())[:-1]:
            code_lines.append(f'{k} = {v.code}')

        # get the linear system using sympy
        var = sympy.Symbol(self.diff_eqs.var, real=True)
        diff_eqs_expr = sub_exprs[f'd{self.diff_eqs.var}dt']
        df_expr = str_to_sympy(diff_eqs_expr.code).expand()
        s_df = sympy.Symbol(f'_d{self.diff_eqs.var}dt')
        code_lines.append(f'{str(s_df)} = {sympy_to_str(df_expr)}')
        if df_expr.has(var):
            linear = sympy.collect(df_expr, var, evaluate=False)[var]
            s_lin_name = f'_{self.py_func_name}_linear'
            s_lin = sympy.Symbol(s_lin_name)
            code_lines.append(f'{s_lin_name} = {sympy_to_str(linear)}')
            update = var + (sympy.exp(s_lin * dt) - 1) / s_lin * s_df
        else:
            update = var + dt * s_df

        # The actual update step
        code_lines.append(f'{self.diff_eqs.var} = {sympy_to_str(update)}')
        code_lines.append(f'_{self.py_func_name}_res = {self.diff_eqs.return_expressions.code_line}')

        #
        code = '\n'.join(code_lines)
        self._update_code = self.argument_substitute(code)


class MilsteinIto(Integrator):
    """Itô stochastic integral. The derivative-free Milstein method is
    an order 1.0 strong Taylor schema.

    The following implementation approximates this derivative thanks to a
    Runge-Kutta approach [1]_.

    In Itô scheme, it is expressed as

    .. math::

        Y_{n+1} = Y_n + f_n h + g_n \\Delta W_n + {1 \\over 2\\sqrt{h}}
        [g(\\overline{Y_n}) - g_n] [(\\Delta W_n)^2-h]

    where :math:`\\overline{Y_n} = Y_n + f_n h + g_n \\sqrt{h}`.

    Parameters
    ----------
    diff_eqs : DiffEquation
        The differential equation.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.

    References
    ----------
    .. [1] P.E. Kloeden, E. Platen, and H. Schurz, Numerical solution of SDE
           through computer experiments, Springer, 1994.

    """

    def __init__(self, diff_eqs):
        super(MilsteinIto, self).__init__(diff_eqs)

        if profile.is_numpy_bk():
            self.get_np_step()

    def get_np_step(self):

        assert self.diff_eqs.type == _SDE_TYPE
        f = self.diff_eqs.f
        g = self.diff_eqs.g
        dt = profile.get_dt()
        dt_sqrt = bnp.sqrt(dt)

        if callable(g):
            if self.diff_eqs.is_multi_return:
                def int_fg(y0, t, *args):
                    dW = bnp.random.normal(0.0, 1.0, y0.shape)
                    val = f(y0, t, *args)
                    df = val[0] * dt
                    g_n = g(y0, t, *args)
                    dg = g_n * dW * dt_sqrt
                    y_n_bar = y0 + df + g_n * dt_sqrt
                    g_n_bar = g(y_n_bar, t, *args)
                    y1 = y0 + df + dg + 0.5 * (g_n_bar - g_n) * (dW * dW * dt_sqrt - dt_sqrt)
                    return (y1,) + val[1:]

            else:
                def int_fg(y0, t, *args):
                    dW = bnp.random.normal(0.0, 1.0, y0.shape)
                    df = f(y0, t, *args) * dt
                    g_n = g(y0, t, *args)
                    dg = g_n * dW * dt_sqrt
                    y_n_bar = y0 + df + g_n * dt_sqrt
                    g_n_bar = g(y_n_bar, t, *args)
                    y1 = y0 + df + dg + 0.5 * (g_n_bar - g_n) * (dW * dW * dt_sqrt - dt_sqrt)
                    return y1

        else:
            if self.diff_eqs.is_multi_return:
                def int_fg(y0, t, *args):
                    dW = bnp.random.normal(0.0, 1.0, y0.shape)
                    val = f(y0, t, *args)
                    df = val[0] * dt
                    dg = g * dW * dt_sqrt
                    y1 = y0 + df + dg
                    return (y1,) + tuple(val[1:])

            else:
                def int_fg(y0, t, *args):
                    dW = bnp.random.normal(0.0, 1.0, y0.shape)
                    df = f(y0, t, *args) * dt
                    dg = g * dW * dt_sqrt
                    y1 = y0 + df + dg
                    return y1

        self._np_step = int_fg


class MilsteinStra(Integrator):
    """Heun two-stage stochastic numerical method for Stratonovich integral.

    Use the Stratonovich Heun algorithm to integrate Stratonovich equation,
    according to paper [1]_ [2]_.

    .. math::
        Y_{n+1} &= Y_n + f_n h + {1 \\over 2}[g_n + g(\\overline{Y}_n)] \\Delta W_n

        \\overline{Y}_n &= Y_n + g_n \\Delta W_n


    Or, it is written as [22]_

    .. math::

        Y_1 &= y_n + f(y_n)h + g_n \\Delta W_n

        y_{n+1} &= y_n + {1 \over 2}[f(y_n) + f(Y_1)]h + {1 \\over 2} [g(y_n) + g(Y_1)] \\Delta W_n


    Parameters
    ----------
    diff_eqs : DiffEquation
        The differential equation.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.

    References
    ----------

    .. [1] H. Gilsing and T. Shardlow, SDELab: A package for solving stochastic differential
         equations in MATLAB, Journal of Computational and Applied Mathematics 205 (2007),
         no. 2, 1002-1018.
    .. [2] P.E. Kloeden, E. Platen, and H. Schurz, Numerical solution of SDE through computer
         experiments, Springer, 1994.

    See Also
    --------
    MilsteinIto

    """

    def __init__(self, diff_eqs):
        super(MilsteinStra, self).__init__(diff_eqs)

        if profile.is_numpy_bk():
            self.get_np_step()

    def get_np_step(self):

        assert self.diff_eqs.type == _SDE_TYPE
        f = self.diff_eqs.f
        g = self.diff_eqs.g
        dt = profile.get_dt()
        dt_sqrt = bnp.sqrt(dt)

        if callable(g):
            if self.diff_eqs.is_multi_return:
                def int_fg(y0, t, *args):
                    dW = bnp.random.normal(0.0, 1.0, y0.shape)
                    val = f(y0, t, *args)
                    df = val[0] * dt
                    g_n = g(y0, t, *args)
                    dg = g_n * dW * dt_sqrt
                    y_n_bar = y0 + df + g_n * dt_sqrt
                    g_n_bar = g(y_n_bar, t, *args)
                    extra_term = 0.5 * (g_n_bar - g_n) * (dW * dW * dt_sqrt)
                    y1 = y0 + df + dg + extra_term
                    return (y1,) + tuple(val[1:])

            else:
                def int_fg(y0, t, *args):
                    dW = bnp.random.normal(0.0, 1.0, y0.shape)
                    df = f(y0, t, *args) * dt
                    g_n = g(y0, t, *args)
                    dg = g_n * dW * dt_sqrt
                    y_n_bar = y0 + df + g_n * dt_sqrt
                    g_n_bar = g(y_n_bar, t, *args)
                    extra_term = 0.5 * (g_n_bar - g_n) * (dW * dW * dt_sqrt)
                    y1 = y0 + df + dg + extra_term
                    return y1

        else:
            if self.diff_eqs.is_multi_return:
                def int_fg(y0, t, *args):
                    dW = bnp.random.normal(0.0, 1.0, y0.shape)
                    val = f(y0, t, *args)
                    df = val[0] * dt
                    dg = g * dW * dt_sqrt
                    y1 = y0 + df + dg
                    return (y1,) + tuple(val[1:])

            else:
                def int_fg(y0, t, *args):
                    dW = bnp.random.normal(0.0, 1.0, y0.shape)
                    df = f(y0, t, *args) * dt
                    dg = g * dW * dt_sqrt
                    y1 = y0 + df + dg
                    return y1

        self._np_step = int_fg
