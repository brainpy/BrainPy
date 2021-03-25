# -*- coding: utf-8 -*-

import ast
from pprint import pprint

import pytest

from brainpy.errors import DiffEqError
from brainpy.integrators.ast_analysis import DiffEqReader
from brainpy.integrators.ast_analysis import separate_variables


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

    r = separate_variables(returns=analyser.returns,
                           variables=analyser.variables,
                           right_exprs=analyser.rights,
                           code_lines=analyser.code_lines)
    pprint(r)



# test_separate_variables1()
