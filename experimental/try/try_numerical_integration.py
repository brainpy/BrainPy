
import numpy as np

import numba


@numba.njit
def int_f(f, dt, y0, t, *args):
    return y0 + dt * f(y0, t, *args)


f1 = numba.njit(lambda a, t, b, c: 1 + a + b + c)
f2 = numba.njit(lambda a, t, b: 1 + a + b)


y1 = np.zeros(10)
y1 = int_f(f1, 0.1, y1, 0., 1, 2)
y2 = np.zeros(5)
y2 = int_f(f1, 0.1, y2, 0., 1)






