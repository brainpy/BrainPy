import numpy as np
from npbrain.integration import DiffEquation
from npbrain import integrate
from npbrain import tools
from npbrain import profile
profile.set_backend('numba')


def try_analyse_func():
    import numpy as np

    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m, (alpha, beta)

    from pprint import pprint

    df = DiffEquation(int_m)
    pprint(df.f_expressions)
    pprint(df.f_returns)
    pprint(df.get_f_expressions())
    pprint(df.get_g_expressions())
    print('-'* 30)


def try_analyse_func2():
    def func(m, t):
        a = t + 2
        b = m + 6
        c = a + b
        d = m * 2 + c
        return d, c

    from pprint import pprint

    df = DiffEquation(f=func, g=np.zeros(10))
    pprint(df.f_expressions)
    pprint(df.f_returns)
    pprint(df.get_f_expressions())
    pprint(df.get_g_expressions())
    print('-' * 30)


def try_analyse_func3():
    def g_func(m, t):
        a = t + 2
        b = m + 6
        c = a + b
        d = m * 2 + c
        return d

    from pprint import pprint

    df = DiffEquation(f=None, g=g_func)
    pprint(df.f_expressions)
    pprint(df.f_returns)
    pprint(df.get_f_expressions())
    pprint(df.get_g_expressions())
    print('-' * 30)


def try_integrate():
    import numpy as np

    @integrate(method='exponential')
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m, alpha, beta

    print(type(int_m))
    print(int_m._update_code)


def try_diff_eq_analyser():
    code = '''
alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

beta = 4.0 * np.exp(-(V + 65) / 18)
return alpha * (1 - m) - beta * m, f(alpha, beta)
    '''

    res = tools.analyse_diff_eq(code)
    print('Return: ', res[2], '\n')
    for var, exp in zip(res[0], res[1]):
        print(var, '=', exp)




if __name__ == '__main__':
    # try_analyse_func()
    try_analyse_func2()
    try_analyse_func3()
    # try_integrate()
    # try_diff_eq_analyser()

