# -*- coding: utf-8 -*-

import numpy as np
from brainpy.integration import DiffEquation
from brainpy import integrate


def try_analyse_func():
    import numpy as np

    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m, (alpha, beta)

    from pprint import pprint

    df = DiffEquation(int_m)
    pprint(df.expressions)
    pprint(df.return_intermediates)
    pprint(df.get_f_expressions())
    pprint(df.get_g_expressions())
    print('-' * 30)


def try_analyse_func2():
    def func(m, t):
        a = t + 2
        b = m + 6
        c = a + b
        d = m * 2 + c
        return d, c

    from pprint import pprint

    df = DiffEquation(func=func)
    pprint(df.expressions)
    pprint(df.return_intermediates)
    pprint(df.get_f_expressions())
    pprint(df.get_g_expressions())
    print('-' * 30)


def try_analyse_func3():
    def g_func(m, t):
        a = t + 2
        b = m + 6
        c = a + b
        d = m * 2 + c
        return 0., d

    from pprint import pprint

    df = DiffEquation(func=g_func)
    pprint(df.expressions)
    pprint(df.return_intermediates)
    pprint(df.get_f_expressions())
    pprint(df.get_g_expressions())
    print('-' * 30)


def try_analyse_func4():
    def g_func(m, t):
        a = t + 2
        b = m + 6
        c = a + b
        d = m * 2 + c
        return (d,), a, b

    from pprint import pprint

    df = DiffEquation(func=g_func)
    pprint(df.expressions)
    pprint(df.return_intermediates)
    pprint(df.get_f_expressions())
    pprint(df.get_g_expressions())
    print('-' * 30)


def try_analyse_func5():
    def g_func(m, t):
        a = t + 2
        b = m + 6
        c = a + b
        d = m * 2 + c
        return (c,d), a, b

    from pprint import pprint

    df = DiffEquation(func=g_func)
    pprint(df.expressions)
    pprint(df.return_intermediates)
    pprint(df.get_f_expressions())
    pprint(df.get_g_expressions())
    print('-' * 30)


def test_multi_system():
    def int_func(array, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        m = alpha * (1 - array[0]) - beta * array[0]

        alpha2 = 0.07 * np.exp(-(V + 65) / 20.)
        beta2 = 1 / (1 + np.exp(-(V + 35) / 10))
        h = alpha2 * (1 - array[1]) - beta2 * array[1]
        return np.array([m, h])

    from pprint import pprint

    df = DiffEquation(func=int_func)
    pprint(df.expressions)
    pprint(df.return_intermediates)
    pprint(df.get_f_expressions())
    pprint(df.get_g_expressions())
    print('-' * 30)


def test_assignment():
    def int_func(array, t, V):
        res = np.zeros_like(array)
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        res[0] = alpha * (1 - array[0]) - beta * array[0]

        alpha2 = 0.07 * np.exp(-(V + 65) / 20.)
        beta2 = 1 / (1 + np.exp(-(V + 35) / 10))
        res[1] = alpha2 * (1 - array[1]) - beta2 * array[1]
        return res

    from pprint import pprint

    df = DiffEquation(func=int_func)
    pprint(df.expressions)
    pprint(df.return_intermediates)
    pprint(df.get_f_expressions())
    pprint(df.get_g_expressions())
    print('-' * 30)


def test_stochastic():
    # Case 1
    # ------

    noise = 1.
    tau = 10.

    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m, noise / beta

    diff_eq = DiffEquation(int_m)
    assert diff_eq.is_stochastic is True

    # Case 2
    # ------

    noise = 0.
    tau = 10.

    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m, noise / beta

    diff_eq = DiffEquation(int_m)
    assert diff_eq.is_stochastic is True

    # Case 3
    # ------

    noise = 0.
    tau = 10.

    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m, noise / tau

    diff_eq = DiffEquation(int_m)
    assert diff_eq.is_stochastic is False

    # Case 4
    # ------

    noise = 1.
    tau = 10.

    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m, noise / tau

    diff_eq = DiffEquation(int_m)
    assert diff_eq.is_stochastic is True


if __name__ == '__main__':
    # try_analyse_func()
    # try_analyse_func2()
    # try_analyse_func3()
    # try_analyse_func4()
    # try_analyse_func5()

    # test_multi_system()
    test_assignment()
    # test_stochastic()

