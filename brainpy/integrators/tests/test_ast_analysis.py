# -*- coding: utf-8 -*-

import ast
from pprint import pprint

import pytest

from brainpy.errors import DiffEqError
from brainpy.integrators.analysis_by_ast import DiffEqReader
from brainpy.integrators.analysis_by_ast import separate_variables


def test_reader1():
    eq_code = '''
def func():
    a, b = f()
    c = a + b
    d = a * b
    return c, d
    '''
    analyser = DiffEqReader()
    analyser.visit(ast.parse(eq_code))

    print(analyser.returns)
    assert analyser.returns == ['c', 'd']

    print(analyser.code_lines)
    assert analyser.code_lines == ['a, b = f()\n',
                                   'c = a + b\n',
                                   'd = a * b\n']

    print(analyser.rights)
    assert analyser.rights == ['f()', 'a + b', 'a * b']

    print(analyser.variables)
    assert analyser.variables == [['a', 'b'],
                                  ['c'],
                                  ['d']]


def test_reader2():
    eq_code = '''
def func():
    a, b = f()
    if a > 0:
        c = a + b
    else:
        c = a * b
    return c
        '''
    analyser = DiffEqReader()
    with pytest.raises(DiffEqError):
        analyser.visit(ast.parse(eq_code))


def test_reader3():
    eq_code = '''
def func():
    a, b = f()
    for i in range(10):
        a += b
    return a
        '''
    analyser = DiffEqReader()
    with pytest.raises(DiffEqError):
        analyser.visit(ast.parse(eq_code))


def test_separate_variables1():
    eq_code = '''
def integral(V, m, h, n, t, Iext):
    alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    beta = 4.0 * np.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m

    alpha = 0.07 * np.exp(-(V + 65) / 20.)
    beta = 1 / (1 + np.exp(-(V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h
    return dmdt, dhdt
'''
    analyser = DiffEqReader()
    analyser.visit(ast.parse(eq_code))

    print('returns: ')
    pprint(analyser.returns)

    print("code_lines: ")
    pprint(analyser.code_lines)

    print("rights: ")
    pprint(analyser.rights)

    print("variables: ")
    pprint(analyser.variables)

    r = separate_variables(eq_code)
    pprint(r)


def test_separate_variables2():
    code = '''def derivative(V, m, h, n, t, C, gNa, ENa, gK, EK, gL, EL, Iext):
    alpha = 0.1 * (V + 40) / (1 - bp.backend.exp(-(V + 40) / 10))
    beta = 4.0 * bp.backend.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m

    alpha = 0.07 * bp.backend.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bp.backend.exp(-(V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h

    alpha = 0.01 * (V + 55) / (1 - bp.backend.exp(-(V + 55) / 10))
    beta = 0.125 * bp.backend.exp(-(V + 65) / 80)
    dndt = alpha * (1 - n) - beta * n

    I_Na = (gNa * m ** 3.0 * h) * (V - ENa)
    I_K = (gK * n ** 4.0) * (V - EK)
    I_leak = gL * (V - EL)
    dVdt = (- I_Na - I_K - I_leak + Iext) / C

    return dVdt, dmdt, dhdt, dndt
    '''

    from pprint import pprint
    pprint(separate_variables(code))

# test_reader1()
# test_reader2()
# test_reader3()
# test_separate_variables1()
# test_separate_variables2()
