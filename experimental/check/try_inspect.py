# -*- coding: utf-8 -*-


import inspect


def f1():
    print("f1")


def f2():
    f1()
    print('f2')


def f3(a, b):
    print("a = ", a)
    print("b = ", b)
    f2()
    print('f3')


def try1():
    r1 = inspect.getclosurevars(f3)
    print(r1)

    r2 = inspect.getclosurevars(r1.globals['f2'])
    print(r2)

    r3 = inspect.getclosurevars(r2.globals['f1'])
    print(r3)


# try1()

def try2():
    def f():
        print('a')

    return f


# ff = try2()
# print(inspect.getsourcelines(ff))

def try3(func):
    def f():
        func()
        print('a')

    return f


@try3
def f4():
    print('f4')


print(inspect.getsourcelines(f4))
print(inspect.getclosurevars(f4))


