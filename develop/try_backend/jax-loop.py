# -*- coding: utf-8 -*-
import jax
import jax.numpy as np
from jax.experimental import loops


def test_laxwhile(a, b):
    quantity = 0.
    i = a + b
    j = 2 * i + 1
    k = 2 * j + 1

    condfun_1 = lambda inp: inp[1] > 0.  # i > 0.
    condfun_2 = lambda inp: inp[2] > 0.  # j > 0.
    condfun_3 = lambda inp: inp[3] > 0.  # k > 0.

    def bodyfun_1(inp):
        quantity, i, j, k = inp
        j = 2 * i + 1

        def bodyfun_2(inp):
            quantity, i, j, k = inp
            k = 2 * j + 1

            def bodyfun_3(inp):
                quantity, i, j, k = inp
                # quantity += i + j + k  # This works!
                quantity += i + k + j  # This works!
                k -= 1.
                return (quantity, i, j, k)

            result = jax.lax.while_loop(condfun_3, bodyfun_3, (quantity, i, j, k))
            quantity = result[0]
            j -= 1.
            return (quantity, i, j, k)

        result = jax.lax.while_loop(condfun_2, bodyfun_2, (quantity, i, j, k))
        quantity = result[0]
        i -= 1.
        return (quantity, i, j, k)

    result = jax.lax.while_loop(condfun_1, bodyfun_1, (quantity, i, j, k))
    return result[0]


# print(test_laxwhile(3., 2.))


def control(a, b):
    quantity = 0
    i = a + b
    while i > 0:
        j = 2 * i + 1
        while j > 0:
            k = 2 * j + 1
            while k > 0:
                quantity += i + j + k
                k -= 1
            j -= 1
        i -= 1
    return quantity


def test(a, b):
    with loops.Scope() as s:
        s.quantity = 0.0
        s.a = a
        s.b = b
        s.i = s.a + s.b
        s.j = 2 * s.i + 1
        s.k = 2 * s.j + 1

        for _ in s.while_range(lambda: s.i > 0):
            s.j = 2 * s.i + 1
            for _ in s.while_range(lambda: s.j > 0):
                s.k = 2 * s.j + 1
                for _ in s.while_range(lambda: s.k > 0):
                    s.quantity += s.i + s.j + s.k
                    s.k -= 1
                s.j -= 1
            s.i -= 1

    return s.quantity + 1.
        # return s.quantity


# A = np.repeat(np.arange(10), 1000)
# B = np.repeat(np.arange(10), 1000)
# v_test = jax.vmap(test, (0, 0))
# print(v_test(A, B))
print(test(3., 2.))
