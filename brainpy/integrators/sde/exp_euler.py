# -*- coding: utf-8 -*-

import numpy as np
import sympy

from brainpy import backend
from brainpy import errors
from brainpy import tools
from brainpy.integrators import ast_analysis

__all__ = [
    'exponential_euler',
]


class Integrator(object):
    def __init__(self, diff_eq):
        if not isinstance(diff_eq, ast_analysis.DiffEquation):
            if diff_eq.__class__.__name__ != 'function':
                raise errors.IntegratorError('"diff_eq" must be a function or an instance of DiffEquation .')
            else:
                diff_eq = ast_analysis.DiffEquation(func=diff_eq)
        self.diff_eq = diff_eq
        self._update_code = None
        self._update_func = None

    def __call__(self, y0, t, *args):
        return self._update_func(y0, t, *args)

    def _compile(self):
        # function arguments
        func_args = ', '.join([f'_{arg}' for arg in self.diff_eq.func_args])

        # function codes
        func_code = f'def {self.py_func_name}({func_args}): \n'
        func_code += tools.indent(self._update_code + '\n' + f'return _res')
        tools.NoiseHandler.normal_pattern.sub(
            tools.NoiseHandler.vector_replace_f, func_code)

        # function scope
        code_scopes = {'numpy': np}
        for k_, v_ in self.code_scope.items():
            if backend.is_jit() and callable(v_):
                v_ = tools.numba_func(v_)
            code_scopes[k_] = v_
        code_scopes.update(ast_analysis.get_mapping_scope())
        code_scopes['_normal_like_'] = backend.normal_like

        # function compilation
        exec(compile(func_code, '', 'exec'), code_scopes)
        func = code_scopes[self.py_func_name]
        if backend.is_jit():
            func = tools.jit(func)
        self._update_func = func

    @staticmethod
    def get_integral_step(diff_eq, *args):
        raise NotImplementedError

    @property
    def py_func_name(self):
        return self.diff_eq.func_name

    @property
    def update_code(self):
        return self._update_code

    @property
    def update_func(self):
        return self._update_func

    @property
    def code_scope(self):
        scope = self.diff_eq.func_scope
        if backend.run_on_cpu():
            scope['_normal_like_'] = backend.normal_like
        return scope


class ExponentialEuler(Integrator):
    """First order, explicit exponential Euler method.

    For an ODE equation of the form

    .. math::

        y^{\\prime}=f(y), \quad y(0)=y_{0}

    its schema is given by

    .. math::

        y_{n+1}= y_{n}+h \\varphi(hA) f (y_{n})

    where :math:`A=f^{\prime}(y_{n})` and :math:`\\varphi(z)=\\frac{e^{z}-1}{z}`.

    For linear ODE system: :math:`y^{\\prime} = Ay + B`,
    the above equation is equal to

    .. math::

        y_{n+1}= y_{n}e^{hA}-B/A(1-e^{hA})

    For a SDE equation of the form

    .. math::

        d y=(Ay+ F(y))dt + g(y)dW(t) = f(y)dt + g(y)dW(t), \\quad y(0)=y_{0}

    its schema is given by [16]_

    .. math::

        y_{n+1} & =e^{\\Delta t A}(y_{n}+ g(y_n)\\Delta W_{n})+\\varphi(\\Delta t A) F(y_{n}) \\Delta t \\\\
         &= y_n + \\Delta t \\varphi(\\Delta t A) f(y) + e^{\\Delta t A}g(y_n)\\Delta W_{n}

    where :math:`\\varphi(z)=\\frac{e^{z}-1}{z}`.

    Parameters
    ----------
    diff_eq : DiffEquation
        The differential equation.

    Returns
    -------
    func : callable
        The one-step numerical integrator function.

    References
    ----------
    .. [1] Erdoğan, Utku, and Gabriel J. Lord. "A new class of exponential integrators for stochastic
           differential equations with multiplicative noise." arXiv preprint arXiv:1608.07096 (2016).
    """

    def __init__(self, diff_eq):
        super(ExponentialEuler, self).__init__(diff_eq)
        self._update_code = self.get_integral_step(diff_eq)
        self._compile()

    @staticmethod
    def get_integral_step(diff_eq, *args):
        dt = backend.get_dt()
        f_expressions = diff_eq.get_f_expressions(substitute_vars=diff_eq.var_name)

        # code lines
        code_lines = [str(expr) for expr in f_expressions[:-1]]

        # get the linear system using sympy
        f_res = f_expressions[-1]
        df_expr = ast_analysis.str2sympy(f_res.code).expr.expand()
        s_df = sympy.Symbol(f"{f_res.var_name}")
        code_lines.append(f'{s_df.name} = {ast_analysis.sympy2str(df_expr)}')
        var = sympy.Symbol(diff_eq.var_name, real=True)

        # get df part
        s_linear = sympy.Symbol(f'_{diff_eq.var_name}_linear')
        s_linear_exp = sympy.Symbol(f'_{diff_eq.var_name}_linear_exp')
        s_df_part = sympy.Symbol(f'_{diff_eq.var_name}_df_part')
        if df_expr.has(var):
            # linear
            linear = sympy.collect(df_expr, var, evaluate=False)[var]
            code_lines.append(f'{s_linear.name} = {ast_analysis.sympy2str(linear)}')
            # linear exponential
            linear_exp = sympy.exp(linear * dt)
            code_lines.append(f'{s_linear_exp.name} = {ast_analysis.sympy2str(linear_exp)}')
            # df part
            df_part = (s_linear_exp - 1) / s_linear * s_df
            code_lines.append(f'{s_df_part.name} = {ast_analysis.sympy2str(df_part)}')

        else:
            # linear exponential
            code_lines.append(f'{s_linear_exp.name} = sqrt({dt})')
            # df part
            code_lines.append(f'{s_df_part.name} = {ast_analysis.sympy2str(dt * s_df)}')

        # get dg part
        if diff_eq.is_stochastic:
            # dW
            noise = f'_normal_like_({diff_eq.var_name})'
            code_lines.append(f'_{diff_eq.var_name}_dW = {noise}')
            # expressions of the stochastic part
            g_expressions = diff_eq.get_g_expressions()
            code_lines.extend([str(expr) for expr in g_expressions[:-1]])
            g_expr = g_expressions[-1].code
            # get the dg_part
            s_dg_part = sympy.Symbol(f'_{diff_eq.var_name}_dg_part')
            code_lines.append(f'_{diff_eq.var_name}_dg_part = {g_expr} * _{diff_eq.var_name}_dW')
        else:
            s_dg_part = 0

        # update expression
        update = var + s_df_part + s_dg_part * s_linear_exp

        # The actual update step
        code_lines.append(f'{diff_eq.var_name} = {ast_analysis.sympy2str(update)}')
        return_expr = ', '.join([diff_eq.var_name] + diff_eq.return_intermediates)
        code_lines.append(f'_res = {return_expr}')

        # final
        code = '\n'.join(code_lines)
        subs_dict = {arg: f'_{arg}' for arg in diff_eq.func_args + diff_eq.expr_names}
        code = tools.word_replace(code, subs_dict)
        return code


def exponential_euler(f):
    dt = backend.get_dt()
    dt_sqrt = dt ** 0.5

    def int_f(x, t, *args):
        df, linear_part, g = f(x, t, *args)
        dW = backend.normal(0., 1., backend.shape(x))
        dg = dt_sqrt * g * dW
        exp = backend.exp(linear_part * dt)
        y1 = x + (exp - 1) / linear_part * df + exp * dg
        return y1

    return int_f
