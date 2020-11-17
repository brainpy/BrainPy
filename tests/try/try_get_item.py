# -*- coding: utf-8 -*-


class A():
    def __init__(self):
        pass

    def __getitem__(self, item):
        print(item)
        print(item[0].indices(10))

# A()[1]
# A()[1:5]
# A()[1:5, 2:3]
# A()[1, 2]
A()[:, 2]
