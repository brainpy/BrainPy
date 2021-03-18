# -*- coding: utf-8 -*-

import numpy as np
import brainpy as bp
import sympy

# @bp.integrate
# def int_m(m, t, V):
#     alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
#     beta = 4.0 * np.exp(-(V + 65) / 18)
#     return alpha * (1 - m) - beta * m, (alpha, beta)
#
# sympy_eqs = [ bp.integrators.str2sympy(str(eq))
#     for eq in int_m.diff_eq.get_f_expressions()]


m = sympy.Symbol('m')
V = sympy.Symbol('V')
# alpha = sympy.Symbol('alpha')
# beta = sympy.Symbol('beta')

f = sympy.Function('f')
alpha = 0.1 * (V + 40) / (1 - sympy.exp(-(V + 40) / 10))
beta = 4.0 * sympy.exp(-(V + 65) / 18)
dvdt = f(alpha) * (1 - m) - beta * m
# dvdt = bp.integrators.str2sympy('alpha * (1 - m) - beta * m')

# print(sympy.Derivative(dvdt, V).doit())
diff = sympy.diff(dvdt, V)
print(diff)

print(sympy.solve(dvdt, V))

pass

