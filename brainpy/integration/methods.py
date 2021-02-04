# -*- coding: utf-8 -*-

import numpy as np

from . import diff_equation
from .. import errors
from .. import profile
from ..backend import normal_like

__all__ = [
    'euler',
    'heun',
    'rk2',
    'rk3',
    'rk4',
    'rk4_alternative',
    'exponential_euler',
    'milstein_Ito',
    'milstein_Stra',
]


def euler(diff_eqs):
    __f = diff_eqs.func
    __dt = profile.get_dt()

    # SDE
    if diff_eqs.is_stochastic:
        __dt_sqrt = np.sqrt(profile.get_dt())

        if diff_eqs.is_multi_return:
            if diff_eqs.return_type == '(x,x),':
                def int_func(y0, t, *args):
                    val = __f(y0, t, *args)
                    dfdg = val[0]
                    df = dfdg[0] * __dt
                    dW = normal_like(y0)
                    dg = __dt_sqrt * dfdg[1] * dW
                    y = y0 + df + dg
                    return (y,) + tuple(val[1:])
            else:
                raise ValueError
        else:

            if diff_eqs.return_type == '(x,x),':
                def int_func(y0, t, *args):
                    val = __f(y0, t, *args)[0]
                    df = val[0] * __dt
                    dW = normal_like(y0)
                    dg = __dt_sqrt * val[1] * dW
                    return y0 + df + dg

            elif diff_eqs.return_type == 'x,x':
                def int_func(y0, t, *args):
                    val = __f(y0, t, *args)
                    df = val[0] * __dt
                    dW = normal_like(y0)
                    dg = __dt_sqrt * val[1] * dW
                    return y0 + df + dg
            else:
                raise errors.IntegratorError

    # ODE
    else:
        if diff_eqs.is_multi_return:
            if diff_eqs.return_type in ['(x,),', '(x,x),']:
                def int_func(y0, t, *args):
                    val = __f(y0, t, *args)
                    y = y0 + __dt * val[0][0]
                    return (y,) + tuple(val[1:])
            else:
                raise errors.IntegratorError(f'Unrecognized differential return type: '
                                             f'{diff_eqs.return_type}')

        else:
            if diff_eqs.return_type == 'x':
                def int_func(y0, t, *args):
                    return y0 + __dt * __f(y0, t, *args)
            elif diff_eqs.return_type == 'x,x':
                def int_func(y0, t, *args):
                    return y0 + __dt * __f(y0, t, *args)[0]
            elif diff_eqs.return_type in ['(x,),', '(x,x),']:
                def int_func(y0, t, *args):
                    return y0 + __dt * __f(y0, t, *args)[0][0]
            else:
                raise errors.IntegratorError(f'Unrecognized differential return type: '
                                             f'{diff_eqs.return_type}')
    return int_func


def rk2(diff_eqs, __beta=2 / 3):
    __f = diff_eqs.func
    __dt = profile.get_dt()

    if diff_eqs.is_stochastic:
        raise NotImplementedError
    else:

        if diff_eqs.is_multi_return:
            if diff_eqs.return_type in ['(x,),', '(x,x),']:
                def int_func(y0, t, *args):
                    val = __f(y0, t, *args)
                    k1 = val[0][0]
                    k2 = __f(y0 + __beta * __dt * k1, t + __beta * __dt, *args)[0][0]
                    y = y0 + __dt * ((1 - 1 / (2 * __beta)) * k1 + 1 / (2 * __beta) * k2)
                    return (y,) + tuple(val[1:])
            else:
                raise errors.IntegratorError(f'Unrecognized differential return type: '
                                             f'{diff_eqs.return_type}')

        else:
            if diff_eqs.return_type == 'x':
                def int_func(y0, t, *args):
                    k1 = __f(y0, t, *args)
                    k2 = __f(y0 + __beta * __dt * k1, t + __beta * __dt, *args)
                    y = y0 + __dt * ((1 - 1 / (2 * __beta)) * k1 + 1 / (2 * __beta) * k2)
                    return y
            elif diff_eqs.return_type == 'x,x':
                def int_func(y0, t, *args):
                    k1 = __f(y0, t, *args)[0]
                    k2 = __f(y0 + __beta * __dt * k1, t + __beta * __dt, *args)[0]
                    y = y0 + __dt * ((1 - 1 / (2 * __beta)) * k1 + 1 / (2 * __beta) * k2)
                    return y
            elif diff_eqs.return_type in ['(x,),', '(x,x),']:
                def int_func(y0, t, *args):
                    k1 = __f(y0, t, *args)[0][0]
                    k2 = __f(y0 + __beta * __dt * k1, t + __beta * __dt, *args)[0][0]
                    y = y0 + __dt * ((1 - 1 / (2 * __beta)) * k1 + 1 / (2 * __beta) * k2)
                    return y
            else:
                raise errors.IntegratorError(f'Unrecognized differential return type: '
                                             f'{diff_eqs.return_type}')

    return int_func


def heun(diff_eqs):
    __f = diff_eqs.func
    __dt = profile.get_dt()

    if diff_eqs.is_stochastic:
        __dt_sqrt = np.sqrt(profile.get_dt())

        if diff_eqs.is_functional_noise:
            if diff_eqs.is_multi_return:
                if diff_eqs.return_type == '(x,x),':
                    def int_func(y0, t, *args):
                        val = __f(y0, t, *args)
                        dfdg = val[0]
                        dg = dfdg[1]
                        df = dfdg[0] * __dt
                        dW = normal_like(y0)
                        y_bar = y0 + dg * dW * __dt_sqrt
                        dg_bar = __f(y_bar, t, *args)[0][1]
                        dg = 0.5 * (dg + dg_bar) * dW * __dt_sqrt
                        y = y0 + df + dg
                        return (y,) + tuple(val[1:])
                else:
                    raise ValueError

            else:
                if diff_eqs.return_type == '(x,x),':
                    def int_func(y0, t, *args):
                        val = __f(y0, t, *args)[0]
                        df = val[0] * __dt
                        dg = val[1]
                        dW = normal_like(y0)
                        y_bar = y0 + dg * dW * __dt_sqrt
                        dg_bar = __f(y_bar, t, *args)[0][1]
                        dg = 0.5 * (dg + dg_bar) * dW * __dt_sqrt
                        y = y0 + df + dg
                        return y
                elif diff_eqs.return_type == 'x,x':
                    def int_func(y0, t, *args):
                        val = __f(y0, t, *args)
                        df = val[0] * __dt
                        dg = val[1]
                        dW = normal_like(y0)
                        y_bar = y0 + dg * dW * __dt_sqrt
                        dg_bar = __f(y_bar, t, *args)[1]
                        dg = 0.5 * (dg + dg_bar) * dW * __dt_sqrt
                        y = y0 + df + dg
                        return y
                else:
                    raise errors.IntegratorError

            return int_func

        else:
            return euler(diff_eqs)
    else:
        return rk2(diff_eqs, __beta=1.)


def rk3(diff_eqs):
    __f = diff_eqs.func
    __dt = profile.get_dt()

    if diff_eqs.is_stochastic:
        raise NotImplementedError

    else:
        if diff_eqs.is_multi_return:

            if diff_eqs.return_type in ['(x,),', '(x,x),']:
                def int_func(y0, t, *args):
                    val = __f(y0, t, *args)
                    k1 = val[0][0]
                    k2 = __f(y0 + __dt / 2 * k1, t + __dt / 2, *args)[0][0]
                    k3 = __f(y0 - __dt * k1 + 2 * __dt * k2, t + __dt, *args)[0][0]
                    y = y0 + __dt / 6 * (k1 + 4 * k2 + k3)
                    return (y,) + tuple(val[1:])
            else:
                raise errors.IntegratorError(f'Unrecognized differential return type: '
                                             f'{diff_eqs.return_type}')

        else:
            if diff_eqs.return_type == 'x':
                def int_func(y0, t, *args):
                    k1 = __f(y0, t, *args)
                    k2 = __f(y0 + __dt / 2 * k1, t + __dt / 2, *args)
                    k3 = __f(y0 - __dt * k1 + 2 * __dt * k2, t + __dt, *args)
                    return y0 + __dt / 6 * (k1 + 4 * k2 + k3)

            elif diff_eqs.return_type == 'x,x':
                def int_func(y0, t, *args):
                    k1 = __f(y0, t, *args)[0]
                    k2 = __f(y0 + __dt / 2 * k1, t + __dt / 2, *args)[0]
                    k3 = __f(y0 - __dt * k1 + 2 * __dt * k2, t + __dt, *args)[0]
                    return y0 + __dt / 6 * (k1 + 4 * k2 + k3)

            elif diff_eqs.return_type in ['(x,),', '(x,x),']:
                def int_func(y0, t, *args):
                    k1 = __f(y0, t, *args)[0][0]
                    k2 = __f(y0 + __dt / 2 * k1, t + __dt / 2, *args)[0][0]
                    k3 = __f(y0 - __dt * k1 + 2 * __dt * k2, t + __dt, *args)[0][0]
                    return y0 + __dt / 6 * (k1 + 4 * k2 + k3)
            else:
                raise errors.IntegratorError(f'Unrecognized differential return type: '
                                             f'{diff_eqs.return_type}')

    return int_func


def rk4(diff_eqs):
    __f = diff_eqs.func
    __dt = profile.get_dt()

    if diff_eqs.is_stochastic:
        raise NotImplementedError

    else:
        if diff_eqs.is_multi_return:
            if diff_eqs.return_type in ['(x,),', '(x,x),']:
                def int_func(y0, t, *args):
                    val = __f(y0, t, *args)
                    k1 = val[0][0]
                    k2 = __f(y0 + __dt / 2 * k1, t + __dt / 2, *args)[0][0]
                    k3 = __f(y0 + __dt / 2 * k2, t + __dt / 2, *args)[0][0]
                    k4 = __f(y0 + __dt * k3, t + __dt, *args)[0][0]
                    y = y0 + __dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                    return (y,) + tuple(val[1:])

            else:
                raise errors.IntegratorError(f'Unrecognized differential return type: '
                                             f'{diff_eqs.return_type}')

        else:
            if diff_eqs.return_type == 'x':
                def int_func(y0, t, *args):
                    k1 = __f(y0, t, *args)
                    k2 = __f(y0 + __dt / 2 * k1, t + __dt / 2, *args)
                    k3 = __f(y0 + __dt / 2 * k2, t + __dt / 2, *args)
                    k4 = __f(y0 + __dt * k3, t + __dt, *args)
                    return y0 + __dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            elif diff_eqs.return_type == 'x,x':
                def int_func(y0, t, *args):
                    k1 = __f(y0, t, *args)[0]
                    k2 = __f(y0 + __dt / 2 * k1, t + __dt / 2, *args)[0]
                    k3 = __f(y0 + __dt / 2 * k2, t + __dt / 2, *args)[0]
                    k4 = __f(y0 + __dt * k3, t + __dt, *args)[0]
                    return y0 + __dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            elif diff_eqs.return_type in ['(x,),', '(x,x),']:
                def int_func(y0, t, *args):
                    k1 = __f(y0, t, *args)[0][0]
                    k2 = __f(y0 + __dt / 2 * k1, t + __dt / 2, *args)[0][0]
                    k3 = __f(y0 + __dt / 2 * k2, t + __dt / 2, *args)[0][0]
                    k4 = __f(y0 + __dt * k3, t + __dt, *args)[0][0]
                    return y0 + __dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise errors.IntegratorError(f'Unrecognized differential return type: '
                                             f'{diff_eqs.return_type}')

    return int_func


def rk4_alternative(diff_eqs):
    __f = diff_eqs.func
    __dt = profile.get_dt()

    if diff_eqs.is_stochastic:
        raise errors.IntegratorError('"RK4_alternative" method doesn\'t support stochastic differential equation.')

    else:
        if diff_eqs.is_multi_return:
            if diff_eqs.return_type in ['(x,),', '(x,x),']:
                def int_func(y0, t, *args):
                    val = __f(y0, t, *args)
                    k1 = val[0][0]
                    k2 = __f(y0 + __dt / 3 * k1, t + __dt / 3, *args)[0][0]
                    k3 = __f(y0 - __dt / 3 * k1 + __dt * k2, t + 2 * __dt / 3, *args)[0][0]
                    k4 = __f(y0 + __dt * k1 - __dt * k2 + __dt * k3, t + __dt, *args)[0][0]
                    y = y0 + __dt / 8 * (k1 + 3 * k2 + 3 * k3 + k4)
                    return (y,) + tuple(val[1:])
            else:
                raise errors.IntegratorError(f'Unrecognized differential return type: '
                                             f'{diff_eqs.return_type}')

        else:
            if diff_eqs.return_type == 'x':
                def int_func(y0, t, *args):
                    k1 = __f(y0, t, *args)
                    k2 = __f(y0 + __dt / 3 * k1, t + __dt / 3, *args)
                    k3 = __f(y0 - __dt / 3 * k1 + __dt * k2, t + 2 * __dt / 3, *args)
                    k4 = __f(y0 + __dt * k1 - __dt * k2 + __dt * k3, t + __dt, *args)
                    return y0 + __dt / 8 * (k1 + 3 * k2 + 3 * k3 + k4)
            elif diff_eqs.return_type == 'x,x':
                def int_func(y0, t, *args):
                    k1 = __f(y0, t, *args)[0]
                    k2 = __f(y0 + __dt / 3 * k1, t + __dt / 3, *args)[0]
                    k3 = __f(y0 - __dt / 3 * k1 + __dt * k2, t + 2 * __dt / 3, *args)[0]
                    k4 = __f(y0 + __dt * k1 - __dt * k2 + __dt * k3, t + __dt, *args)[0]
                    return y0 + __dt / 8 * (k1 + 3 * k2 + 3 * k3 + k4)
            elif diff_eqs.return_type in ['(x,),', '(x,x),']:
                def int_func(y0, t, *args):
                    k1 = __f(y0, t, *args)[0][0]
                    k2 = __f(y0 + __dt / 3 * k1, t + __dt / 3, *args)[0][0]
                    k3 = __f(y0 - __dt / 3 * k1 + __dt * k2, t + 2 * __dt / 3, *args)[0][0]
                    k4 = __f(y0 + __dt * k1 - __dt * k2 + __dt * k3, t + __dt, *args)[0][0]
                    return y0 + __dt / 8 * (k1 + 3 * k2 + 3 * k3 + k4)
            else:
                raise errors.IntegratorError(f'Unrecognized differential return type: '
                                             f'{diff_eqs.return_type}')

    return int_func


def exponential_euler(diff_eq):
    assert isinstance(diff_eq, diff_equation.DiffEquation)

    f = diff_eq.f
    dt = profile.get_dt()

    if diff_eq.is_stochastic:
        dt_sqrt = np.sqrt(dt)
        g = diff_eq.g

        if callable(g):

            if diff_eq.is_multi_return:

                def int_f(y0, t, *args):
                    val = f(y0, t, *args)
                    dydt, linear_part = val[0], val[1]
                    dW = normal_like(y0)
                    dg = dt_sqrt * g(y0, t, *args) * dW
                    exp = np.exp(linear_part * dt)
                    y1 = y0 + (exp - 1) / linear_part * dydt + exp * dg
                    return (y1,) + tuple(val[2:])

            else:

                def int_f(y0, t, *args):
                    dydt, linear_part = f(y0, t, *args)
                    dW = normal_like(y0)
                    dg = dt_sqrt * g(y0, t, *args) * dW
                    exp = np.exp(linear_part * dt)
                    y1 = y0 + (exp - 1) / linear_part * dydt + exp * dg
                    return y1

        else:
            assert isinstance(g, (int, float, np.ndarray))

            if diff_eq.is_multi_return:

                def int_f(y0, t, *args):
                    val = f(y0, t, *args)
                    dydt, linear_part = val[0], val[1]
                    dW = normal_like(y0)
                    dg = dt_sqrt * g * dW
                    exp = np.exp(linear_part * dt)
                    y1 = y0 + (exp - 1) / linear_part * dydt + exp * dg
                    return (y1,) + tuple(val[1:])

            else:

                def int_f(y0, t, *args):
                    dydt, linear_part = f(y0, t, *args)
                    dW = normal_like(y0)
                    dg = dt_sqrt * g * dW
                    exp = np.exp(linear_part * dt)
                    y1 = y0 + (exp - 1) / linear_part * dydt + exp * dg
                    return y1

    else:

        if diff_eq.is_multi_return:

            def int_f(y0, t, *args):
                val = f(y0, t, *args)
                df, linear_part = val[0], val[1]
                y = y0 + (np.exp(linear_part * dt) - 1) / linear_part * df
                return (y,) + tuple(val[2:])

        else:

            def int_f(y0, t, *args):
                df, linear_part = f(y0, t, *args)
                y = y0 + (np.exp(linear_part * dt) - 1) / linear_part * df
                return y

    return int_f


def milstein_Ito(diff_eq):
    __f = diff_eq.func
    __dt = profile.get_dt()
    __dt_sqrt = np.sqrt(profile.get_dt())

    if diff_eq.is_stochastic:
        if diff_eq.is_functional_noise:
            if diff_eq.return_type == '(x,x),':
                if diff_eq.is_multi_return:
                    def int_func(y0, t, *args):
                        dW = normal_like(y0)
                        val = __f(y0, t, *args)
                        dfdg = val[0][0]
                        df = dfdg[0] * __dt
                        g_n = dfdg[1]
                        dg = dfdg[1] * dW * __dt_sqrt
                        y_n_bar = y0 + df + g_n * __dt_sqrt
                        g_n_bar = __f(y_n_bar, t, *args)[0][1]
                        y1 = y0 + df + dg + 0.5 * (g_n_bar - g_n) * (dW * dW * __dt_sqrt - __dt_sqrt)
                        return (y1,) + tuple(val[1:])
                else:
                    def int_func(y0, t, *args):
                        dW = normal_like(y0)
                        val = __f(y0, t, *args)
                        dfdg = val[0][0]
                        df = dfdg[0] * __dt
                        g_n = dfdg[1]
                        dg = dfdg[1] * dW * __dt_sqrt
                        y_n_bar = y0 + df + g_n * __dt_sqrt
                        g_n_bar = __f(y_n_bar, t, *args)[0][1]
                        y1 = y0 + df + dg + 0.5 * (g_n_bar - g_n) * (dW * dW * __dt_sqrt - __dt_sqrt)
                        return y1
            elif diff_eq.return_type == 'x,x':
                def int_func(y0, t, *args):
                    dW = normal_like(y0)
                    val = __f(y0, t, *args)
                    df = val[0] * __dt
                    g_n = val[1]
                    dg = g_n * dW * __dt_sqrt
                    y_n_bar = y0 + df + g_n * __dt_sqrt
                    g_n_bar = __f(y_n_bar, t, *args)[1]
                    y1 = y0 + df + dg + 0.5 * (g_n_bar - g_n) * (dW * dW * __dt_sqrt - __dt_sqrt)
                    return y1
            else:
                raise ValueError

            return int_func

    return euler(diff_eq)


def milstein_Stra(diff_eq):
    __f = diff_eq.func
    __dt = profile.get_dt()
    __dt_sqrt = np.sqrt(profile.get_dt())

    if diff_eq.is_stochastic:
        if diff_eq.is_functional_noise:
            if diff_eq.return_type == '(x,x),':

                if diff_eq.is_multi_return:
                    def int_func(y0, t, *args):
                        dW = normal_like(y0)
                        val = __f(y0, t, *args)
                        dfdg = val[0]
                        df = dfdg[0] * __dt
                        g_n = dfdg[1]
                        dg = dfdg[1] * dW * __dt_sqrt
                        y_n_bar = y0 + df + g_n * __dt_sqrt
                        g_n_bar = __f(y_n_bar, t, *args)[0][1]
                        extra_term = 0.5 * (g_n_bar - g_n) * (dW * dW * __dt_sqrt)
                        y1 = y0 + df + dg + extra_term
                        return (y1,) + tuple(val[1:])
                else:
                    def int_func(y0, t, *args):
                        dW = normal_like(y0)
                        val = __f(y0, t, *args)
                        dfdg = val[0]
                        df = dfdg[0] * __dt
                        g_n = dfdg[1]
                        dg = dfdg[1] * dW * __dt_sqrt
                        y_n_bar = y0 + df + g_n * __dt_sqrt
                        g_n_bar = __f(y_n_bar, t, *args)[0][1]
                        extra_term = 0.5 * (g_n_bar - g_n) * (dW * dW * __dt_sqrt)
                        y1 = y0 + df + dg + extra_term
                        return y1
            elif diff_eq.return_type == 'x,x':
                def int_func(y0, t, *args):
                    dW = normal_like(y0)
                    dfdg = __f(y0, t, *args)
                    df = dfdg[0] * __dt
                    g_n = dfdg[1]
                    dg = dfdg[1] * dW * __dt_sqrt
                    y_n_bar = y0 + df + g_n * __dt_sqrt
                    g_n_bar = __f(y_n_bar, t, *args)[0][1]
                    extra_term = 0.5 * (g_n_bar - g_n) * (dW * dW * __dt_sqrt)
                    y1 = y0 + df + dg + extra_term
                    return y1
            else:
                raise ValueError

            return int_func

    return euler(diff_eq)
