
from npbrain.integration import DiffEquation
from npbrain import integrate
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
    pprint(df.expressions)
    pprint(df.return_expressions)
    pprint(df.substitute(include_subexpressions=True))


def try_analyse_func2():
    def func(m, t):
        a = t + 2
        b = m + 6
        c = a + b
        d = m * 2 + c
        return d, c

    from pprint import pprint

    df = DiffEquation(func)
    pprint(df.expressions)
    pprint(df.return_expressions)
    pprint(df.substitute(include_subexpressions=False))



def try_integrate():
    import numpy as np

    @integrate(method='exponential')
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m, alpha, beta

    print(type(int_m))
    print(int_m._update_code)


if __name__ == '__main__':
    # try_analyse_func()
    # try_analyse_func2()
    try_integrate()

