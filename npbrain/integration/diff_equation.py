# -*- coding: utf-8 -*-

import inspect
import re
from collections import Counter
from collections import OrderedDict

import sympy

from .sympy_tools import str_to_sympy
from .sympy_tools import sympy_to_str
from .. import _numpy as np
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


def extract_name(equation, left=False):
    """Extracts the name of a parameter/variable by looking the left term of an equation."""

    equation = equation.replace(' ', '')

    if left:
        name = equation.strip()
        # Search for increments
        operators = ['+', '-', '*', '/']
        for op in operators:
            if equation.endswith(op):
                return equation.split(op)[0]

    else:
        try:
            name = equation.split('=')[0]
        except:  # No equal sign. Eg: baseline : init=0.0
            return equation.strip()

        # Search for increments
        operators = ['+=', '-=', '*=', '/=', '>=', '<=']
        for op in operators:
            if op in equation:
                return equation.split(op)[0]

    # Check for error
    if name.strip() == "":
        raise ValueError(f'The variable name can not be extracted from "{equation}".')

    # Search for any operation in the left side
    ode = False
    operators = ['+', '-', '*', '/']
    for op in operators:
        if not name.find(op) == -1:
            ode = True
    if not ode:  # variable name is alone on the left side
        return name

    # ODE: the variable name is between d and /dt
    name = re.findall("d([\w]+)/dt", name)
    if len(name) == 1:
        return name[0].strip()
    else:
        return '_undefined'


class Expression(object):
    def __init__(self, var, code):
        self.var = var
        self._code = code.strip()
        self._substituted_code = None

    @property
    def identifiers(self):
        return tools.get_identifiers(self._code)

    def __str__(self):
        return f'{self.var} = {self._code}'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if self._code != other._code:
            return False
        if self.var != other.var:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def code(self, subs=True):
        if subs:
            if self._substituted_code is None:
                return self._code
            else:
                return self._substituted_code
        else:
            return self._code


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

        # differential variable name
        self.var = self.func_args[0]

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
            self.f_returns = ['_func_res_']

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

        # substitute expressions
        self._substitute(self.f_expressions)
        self._substitute(self.g_expressions)

    def _substitute(self, expressions):
        # Goal: Substitute dependent variables into the expresion
        # Hint: This step doesn't require the left variables are unique
        dependencies = {}
        for expr in expressions[:-1]:
            substitutions = {}
            for dep_var, dep_expr in dependencies.items():
                if dep_var in expr.identifiers:
                    code = dep_expr.code(subs=True)
                    substitutions[sympy.Symbol(dep_var, real=True)] = str_to_sympy(code)
            if len(substitutions):
                new_sympy_expr = str_to_sympy(expr._code).xreplace(substitutions)
                new_str_expr = sympy_to_str(new_sympy_expr)
                expr._substituted_code = new_str_expr
                dependencies[expr.var] = expr
            else:
                if self.var in expr.identifiers:
                    dependencies[expr.var] = expr

        # Goal: get the final differential equation
        # Hint: the step requires the expression variables must be unique
        substitutions = {}
        for dep_var, dep_expr in dependencies.items():
            code = dep_expr.code(subs=True)
            substitutions[sympy.Symbol(dep_var, real=True)] = str_to_sympy(code)
        if len(substitutions):
            expr = expressions[-1]
            new_sympy_expr = str_to_sympy(expr._code).xreplace(substitutions)
            new_str_expr = sympy_to_str(new_sympy_expr)
            expr._substituted_code = new_str_expr

    def get_f_expressions(self):
        return_expressions = OrderedDict()
        # the derivative expression
        dif_eq_code = self.f_expressions[-1].code(subs=True)
        return_expressions[f'df_{self.var}dt'] = Expression(f'df_{self.var}dt', dif_eq_code)
        # needed variables
        need_vars = tools.get_identifiers(dif_eq_code)
        need_vars |= tools.get_identifiers(', '.join(self.f_returns[1:]))
        # get the total return expressions
        expr_num = len(self.f_expressions)
        for expr in self.f_expressions[expr_num - 2::-1]:
            if expr.var in need_vars and expr.var not in return_expressions:
                if not profile.substitute_eqs or expr._substituted_code is None:
                    code = expr._code
                else:
                    code = expr._substituted_code
                return_expressions[expr.var] = Expression(expr.var, code)
                need_vars |= tools.get_identifiers(code)
        return_expressions = list(return_expressions.items())[::-1]
        return return_expressions

    def get_g_expressions(self):
        if self.is_functional_noise:
            return_expressions = OrderedDict()
            # the derivative expression
            eq_code = self.g_expressions[-1].code(subs=True)
            return_expressions[f'dg_{self.var}dt'] = Expression(f'dg_{self.var}dt', eq_code)
            # needed variables
            need_vars = tools.get_identifiers(eq_code)
            # get the total return expressions
            expr_num = len(self.g_expressions)
            for expr in self.g_expressions[expr_num - 2::-1]:
                if expr.var in need_vars and expr.var not in return_expressions:
                    if not profile.substitute_eqs or expr._substituted_code is None:
                        code = expr._code
                    else: code = expr._substituted_code
                    return_expressions[expr.var] = Expression(expr.var, code)
                    need_vars |= tools.get_identifiers(code)
            return_expressions = list(return_expressions.items())[::-1]
            return return_expressions
        else:
            return [(f'dg_{self.var}dt', Expression(f'dg_{self.var}dt', self.g_code))]

    def get_subs_expressions(self, variable):
        pass

    @property
    def type(self):
        return _SDE_TYPE if self.is_stochastic else _ODE_TYPE

    @property
    def is_multi_return(self):
        return len(self.f_returns) > 1

    @property
    def is_stochastic(self):
        return self.g is not None or self.g == 0.

    @property
    def is_functional_noise(self):
        return self.g_type == _FUNCTIONAL_NOISE

    @property
    def stochastic_type(self):
        if not self.is_stochastic:
            return None
        else:
            pass
