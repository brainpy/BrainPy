# -*- coding: utf-8 -*-

import pytest

from brainpy.errors import DiffEquationError
from brainpy.integration import utils


def try_diff_eq_analyser():
    code = '''
alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

beta = 4.0 * np.exp(-(V + 65) / 18)
return alpha * (1 - m) - beta * m, f(alpha, beta)
    '''

    res = utils.analyse_diff_eq(code)

    assert res.return_intermediates == []
    assert res.return_type == 'x,x'

    f_expr = res.f_expr
    assert f_expr[0] == '_f_res_'
    assert f_expr[1] == 'alpha * (1 - m) - beta * m'

    g_expr = res.g_expr
    assert g_expr[0] == '_g_res_'
    assert g_expr[1] == 'f(alpha, beta)'

    for var, exp in zip(res.variables, res.expressions):
        print(var, '=', exp)


def try_diff_eq_analyser2():
    res = utils.analyse_diff_eq('return a')
    assert len(res.return_intermediates) == 0
    assert res.return_type == 'x'
    assert res.f_expr[1] == 'a'
    assert res.g_expr is None

    res = utils.analyse_diff_eq('return a, b')
    assert len(res.return_intermediates) == 0
    assert res.return_type == 'x,x'
    assert res.f_expr[1] == 'a'
    assert res.g_expr[1] == 'b'

    res = utils.analyse_diff_eq('return (a, b)')
    assert len(res.return_intermediates) == 0
    assert res.return_type == 'x,x'
    assert res.f_expr[1] == 'a'
    assert res.g_expr[1] == 'b'

    res = utils.analyse_diff_eq('return (a, ), b')
    assert len(res.return_intermediates) == 1
    assert res.return_intermediates[0] == 'b'
    assert res.return_type == '(x,),'
    assert res.f_expr[1] == 'a'
    assert res.g_expr is None

    res = utils.analyse_diff_eq('return (a, b), ')
    assert len(res.return_intermediates) == 0
    assert res.return_type == '(x,x),'
    assert res.f_expr[1] == 'a'
    assert res.g_expr[1] == 'b'

    res = utils.analyse_diff_eq('return (a, b), c, d')
    assert len(res.return_intermediates) == 2
    assert res.return_intermediates[0] == 'c'
    assert res.return_intermediates[1] == 'd'
    assert res.return_type == '(x,x),'
    assert res.f_expr[1] == 'a'
    assert res.g_expr[1] == 'b'

    res = utils.analyse_diff_eq('return ((a, b), c, d)')
    assert len(res.return_intermediates) == 2
    assert res.return_intermediates[0] == 'c'
    assert res.return_intermediates[1] == 'd'
    assert res.return_type == '(x,x),'
    assert res.f_expr[1] == 'a'
    assert res.g_expr[1] == 'b'

    res = utils.analyse_diff_eq('return (a, ), ')
    assert len(res.return_intermediates) == 0
    assert res.return_type == '(x,),'
    assert res.f_expr[1] == 'a'
    assert res.g_expr is None

    res = utils.analyse_diff_eq('return (a+b, b*2), ')
    assert len(res.return_intermediates) == 0
    assert res.return_type == '(x,x),'
    assert res.f_expr[1] == 'a + b'
    assert res.g_expr[1] == 'b * 2'

    with pytest.raises(DiffEquationError) as e:
        utils.analyse_diff_eq('return a, b, c')


