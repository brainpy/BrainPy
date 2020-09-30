# -*- coding: utf-8 -*-

import numba as nb



@nb.njit
def a(num, idx):
    b = [[i for i in range(n)] for n in range(num)]
    return b[idx]
# print(a(10, 2))


@nb.njit
def b(data, idx):
    return data[idx]

# l1 = nb.typed.List()
data = (nb.typed.List([1, 2, 3, 4]),
        nb.typed.List.empty_list(nb.types.int64),
        nb.typed.List([2, 3, 4, 5]))

print(b(data, 2))
