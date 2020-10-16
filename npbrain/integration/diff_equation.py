# -*- coding: utf-8 -*-

import inspect
from collections import Counter

import sympy

from .sympy_tools import str_to_sympy
from .sympy_tools import sympy_to_str
from .. import numpy as np
from .. import profile
from .. import tools

__all__ = [
    'Expression',
    'DiffEquation',
]

_CONSTANT_NOISE = 'CONSTANT'
_FUNCTIONAL_NOISE = 'FUNCTIONAL'
_ODE_TYPE = 'ODE'
_SDE_TYPE = 'SDE'
_DIFF_EQUATION = 'diff_equation'
_SUB_EXPRESSION = 'sub_expression'


class Expression(object):
    def __init__(self, var, code):
        self.var_name = var
        self.code = code.strip()
        self._substituted_code = None

    @property
    def identifiers(self):
        return tools.get_identifiers(self.code)

    def __str__(self):
        return f'{self.var_name} = {self.code}'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if self.code != other.code:
            return False
        if self.var_name != other.var_name:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_code(self, subs=True):
        if subs:
            if self._substituted_code is None:
                return self.code
            else:
                return self._substituted_code
        else:
            return self.code


class DiffEquation(object):
    """Differential Equation.

    A differential equation is defined as the standard form:

    dx/dt = f(x) + g(x) dW

    """

    def __init__(self, f=None, g=None):
        # "f" function
        self.f = f
        if f is None:
            self.f_code = '0'
        else:
            self.f_code = tools.deindent(tools.get_main_code(f))

        # "g" function
        self.g = g
        if g is None:
            self.g_code = '0'
        else:
            self.g_code = tools.deindent(tools.get_main_code(g))

        if not callable(f):
            assert callable(g), '"f" and "g" cannot be None simultaneously.'
            # function arguments
            self.func_args = inspect.getfullargspec(g).args
            # function name
            if tools.is_lambda_function(g):
                self.func_name = f'_integral_{self.func_args[0]}_'
            else:
                self.func_name = g.__name__
            # function scope
            scope = inspect.getclosurevars(g)
            self.func_scope = dict(scope.nonlocals)
            self.func_scope.update(scope.globals)
            if isinstance(f, np.ndarray):
                self.f_code = f'return _f_{self.func_name}'
                self.func_scope[f'_f_{self.func_name}'] = f
            # noise type
            self.g_type = _FUNCTIONAL_NOISE
        else:
            # function arguments
            self.func_args = inspect.getfullargspec(f).args
            # function name
            if tools.is_lambda_function(f):
                self.func_name = f'_integral_{self.func_args[0]}_'
            else:
                self.func_name = f.__name__
            # function scope
            scope = inspect.getclosurevars(f)
            self.func_scope = dict(scope.nonlocals)
            self.func_scope.update(scope.globals)
            if callable(g):
                g_args = inspect.getfullargspec(g).args
                if self.func_args != g_args:
                    raise tools.DiffEquationError(f'The argument of "f" and "g" should be the same. But got:\n\n'
                                                  f'f({", ".join(self.func_args)})\n\n'
                                                  f'and\n\n'
                                                  f'g({", ".join(g_args)})')
                scope = inspect.getclosurevars(g)
                self.func_scope.update(scope.nonlocals)
                self.func_scope.update(scope.globals)
            elif isinstance(g, np.ndarray):
                self.g_code = f'_g_{self.func_name}'
                self.func_scope[f'_g_{self.func_name}'] = g
            # noise type
            self.g_type = _CONSTANT_NOISE
            if callable(g):
                self.g_type = _FUNCTIONAL_NOISE

        # differential variable name and time name
        self.var_name = self.func_args[0]
        self.t_name = self.func_args[1]

        # analyse f code
        if 'return' in self.f_code:
            f_variables, f_expressions, f_returns = tools.analyse_diff_eq(self.f_code)
            self.f_expressions = [Expression(v, expr) for v, expr in zip(f_variables, f_expressions)]
            self.f_returns = f_returns
            for k, num in Counter(f_variables).items():
                if num > 1:
                    raise tools.DiffEquationError(f'Found "{k}" {num} times. Please assign each expression '
                                                  f'in differential function with a unique name. ')
        else:
            self.f_expressions = [Expression('_func_res_', self.f_code)]
            self.f_returns = [self.var_name]

        # analyse g code
        if self.is_functional_noise:
            g_variables, g_expressions, g_returns = tools.analyse_diff_eq(self.g_code)
            if len(g_returns) > 1:
                raise tools.DiffEquationError(f'"g" function can only return one result, but found {len(g_returns)}.')
            for k, num in Counter(g_variables).items():
                if num > 1:
                    raise tools.DiffEquationError(f'Found "{k}" {num} times. Please assign each expression '
                                                  f'in differential function with a unique name. ')
            self.g_expressions = [Expression(v, expr) for v, expr in zip(g_variables, g_expressions)]
            self.g_returns = g_returns[0]
        else:
            self.g_expressions = []
            self.g_returns = self.g_code

    def _substitute(self, expressions):
        # Goal: Substitute dependent variables into the expresion
        # Hint: This step doesn't require the left variables are unique
        dependencies = {}
        for expr in expressions[:-1]:
            substitutions = {}
            for dep_var, dep_expr in dependencies.items():
                if dep_var in expr.identifiers:
                    code = dep_expr.get_code(subs=True)
                    substitutions[sympy.Symbol(dep_var, real=True)] = str_to_sympy(code)
            if len(substitutions):
                new_sympy_expr = str_to_sympy(expr.code).xreplace(substitutions)
                new_str_expr = sympy_to_str(new_sympy_expr)
                expr._substituted_code = new_str_expr
                dependencies[expr.var_name] = expr
            else:
                if self.var_name in expr.identifiers:
                    dependencies[expr.var_name] = expr

        # Goal: get the final differential equation
        # Hint: the step requires the expression variables must be unique
        substitutions = {}
        for dep_var, dep_expr in dependencies.items():
            code = dep_expr.get_code(subs=True)
            substitutions[sympy.Symbol(dep_var, real=True)] = str_to_sympy(code)
        if len(substitutions):
            expr = expressions[-1]
            new_sympy_expr = str_to_sympy(expr.code).xreplace(substitutions)
            new_str_expr = sympy_to_str(new_sympy_expr)
            expr._substituted_code = new_str_expr

    def get_f_expressions(self, substitute=False):
        if substitute:
            self._substitute(self.f_expressions)

        return_expressions = []
        # the derivative expression
        dif_eq_code = self.f_expressions[-1].get_code(subs=True)
        return_expressions.append(Expression(f'_df{self.var_name}_dt', dif_eq_code))
        # needed variables
        need_vars = tools.get_identifiers(dif_eq_code)
        need_vars |= tools.get_identifiers(', '.join(self.f_returns[1:]))
        # get the total return expressions
        expr_num = len(self.f_expressions)
        for expr in self.f_expressions[expr_num - 2::-1]:
            if expr.var_name in need_vars:
                if not profile.substitute_equation or expr._substituted_code is None:
                    code = expr.code
                else:
                    code = expr._substituted_code
                return_expressions.append(Expression(expr.var_name, code))
                need_vars |= tools.get_identifiers(code)
        return_expressions = return_expressions[::-1]
        return return_expressions

    def get_g_expressions(self, substitute=False):
        if self.is_functional_noise:
            if substitute:
                self._substitute(self.g_expressions)

            return_expressions = []
            # the derivative expression
            eq_code = self.g_expressions[-1].get_code(subs=True)
            return_expressions.append(Expression(f'_dg{self.var_name}_dt', eq_code))
            # needed variables
            need_vars = tools.get_identifiers(eq_code)
            # get the total return expressions
            expr_num = len(self.g_expressions)
            for expr in self.g_expressions[expr_num - 2::-1]:
                if expr.var_name in need_vars:
                    if not profile.substitute_equation or expr._substituted_code is None:
                        code = expr.code
                    else:
                        code = expr._substituted_code
                    return_expressions[expr.var_name] = Expression(expr.var_name, code)
                    need_vars |= tools.get_identifiers(code)
            return_expressions = return_expressions[::-1]
            return return_expressions
        else:
            return [Expression(f'_dg{self.var_name}_dt', self.g_code)]

    def _replace_expressions(self, expressions, name, y_sub, t_sub=None):
        return_expressions = []

        # replacement
        replacement = {self.var_name: y_sub}
        if t_sub is not None:
            replacement[self.t_name] = t_sub
        # replace variables in expressions
        for expr in expressions:
            replace = False
            identifiers = expr.identifiers
            for repl_var in replacement.keys():
                if repl_var in identifiers:
                    replace = True
                    break
            if replace:
                code = tools.word_replace(expr.code, replacement)
                new_expr = Expression(f"{expr.var_name}_{name}", code)
                return_expressions.append(new_expr)
                replacement[expr.var_name] = new_expr.var_name
        return return_expressions

    def replace_f_expressions(self, name, y_sub, t_sub=None):
        return self._replace_expressions(self.f_expressions, name=name, y_sub=y_sub, t_sub=t_sub)

    def replace_g_expressions(self, name, y_sub, t_sub=None):
        return self._replace_expressions(self.g_expressions, name=name, y_sub=y_sub, t_sub=t_sub)

    @property
    def type(self):
        return _SDE_TYPE if self.is_stochastic else _ODE_TYPE

    @property
    def is_multi_return(self):
        return len(self.f_returns) > 1

    @property
    def is_stochastic(self):
        return not (self.g is None or np.all(self.g == 0.))

    @property
    def is_functional_noise(self):
        return self.g_type == _FUNCTIONAL_NOISE

    @property
    def stochastic_type(self):
        if not self.is_stochastic:
            return None
        else:
            pass

    @property
    def f_expr_names(self):
        return [expr.var_name for expr in self.f_expressions]

    @property
    def g_expr_names(self):
        return [expr.var_name for expr in self.g_expressions]
