# -*- coding: utf-8 -*-

import inspect
from collections import Counter

import sympy

from . import constants
from . import utils
from .. import errors
from .. import profile
from .. import tools

__all__ = [
    'Expression',
    'DiffEquation',
]


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

    Parameters
    ----------
    func : callable
        The user defined differential equation.
    """

    def __init__(self, func):
        # check
        if func is None:
            raise errors.DiffEquationError('"func" cannot be None.')
        if not (callable(func) and type(func).__name__ == 'function'):
            raise errors.DiffEquationError('"func" must be a function.')

        # function
        self.func = func

        # function string
        self.code = tools.deindent(tools.get_main_code(func))
        if 'return' not in self.code:
            raise errors.DiffEquationError(f'"func" function must return something, '
                                           f'but found no return.\n{self.code}')

        # function arguments
        self.func_args = inspect.getfullargspec(func).args

        # function name
        if tools.is_lambda_function(func):
            self.func_name = f'_integral_{self.func_args[0]}_'
        else:
            self.func_name = func.__name__

        # function scope
        scope = inspect.getclosurevars(func)
        self.func_scope = dict(scope.nonlocals)
        self.func_scope.update(scope.globals)

        # differential variable name and time name
        self.var_name = self.func_args[0]
        self.t_name = self.func_args[1]

        # analyse function code
        res = tools.analyse_diff_eq(self.code)
        self.expressions = [Expression(v, expr) for v, expr in zip(res.variables, res.expressions)]
        self.returns = res.returns
        self.return_type = res.return_type
        self.f_expr = None
        self.g_expr = None
        if res.f_expr is not None:
            self.f_expr = Expression(res.f_expr[0], res.f_expr[1])
        if res.g_expr is not None:
            self.g_expr = Expression(res.g_expr[0], res.g_expr[1])
        for k, num in Counter(res.variables).items():
            if num > 1:
                raise errors.DiffEquationError(
                    f'Found "{k}" {num} times. Please assign each expression '
                    f'in differential function with a unique name. ')

        # analyse noise type
        self.g_type = constants.CONSTANT_NOISE
        self.g_value = None
        if self.g_expr is not None:
            self._substitute(self.g_expr, self.expressions)
            g_code = self.g_expr.get_code(subs=True)
            for idf in tools.get_identifiers(g_code):
                if idf not in self.func_scope:
                    self.g_type = constants.FUNCTIONAL_NOISE
                    break
            else:
                self.g_value = eval(g_code, self.func_scope)

    def _substitute(self, final_exp, expressions, substitute_vars=None):
        """Substitute expressions to get the final single expression

        Parameters
        ----------
        final_exp : Expression
            The final expression.
        expressions : list, tuple
            The list/tuple of expressions.
        """
        if substitute_vars is None:
            return
        if final_exp is None:
            return
        assert substitute_vars == 'all' or \
               substitute_vars == self.var_name or \
               isinstance(substitute_vars, (tuple, list))

        # Goal: Substitute dependent variables into the expresion
        # Hint: This step doesn't require the left variables are unique
        dependencies = {}
        for expr in expressions:
            substitutions = {}
            for dep_var, dep_expr in dependencies.items():
                if dep_var in expr.identifiers:
                    code = dep_expr.get_code(subs=True)
                    substitutions[sympy.Symbol(dep_var, real=True)] = utils.str2sympy(code).expr
            if len(substitutions):
                new_sympy_expr = utils.str2sympy(expr.code).expr.xreplace(substitutions)
                new_str_expr = utils.sympy2str(new_sympy_expr)
                expr._substituted_code = new_str_expr
                dependencies[expr.var_name] = expr
            else:
                if substitute_vars == 'all':
                    dependencies[expr.var_name] = expr
                elif substitute_vars == self.var_name:
                    if self.var_name in expr.identifiers:
                        dependencies[expr.var_name] = expr
                else:
                    ids = expr.identifiers
                    for var in substitute_vars:
                        if var in ids:
                            dependencies[expr.var_name] = expr
                            break

        # Goal: get the final differential equation
        # Hint: the step requires the expression variables must be unique
        substitutions = {}
        for dep_var, dep_expr in dependencies.items():
            code = dep_expr.get_code(subs=True)
            substitutions[sympy.Symbol(dep_var, real=True)] = utils.str2sympy(code).expr
        if len(substitutions):
            new_sympy_expr = utils.str2sympy(final_exp.code).expr.xreplace(substitutions)
            new_str_expr = utils.sympy2str(new_sympy_expr)
            final_exp._substituted_code = new_str_expr

    def get_f_expressions(self, substitute_vars=None):
        if self.f_expr is None:
            return []
        self._substitute(self.f_expr, self.expressions, substitute_vars=substitute_vars)

        return_expressions = []
        # the derivative expression
        dif_eq_code = self.f_expr.get_code(subs=True)
        return_expressions.append(Expression(f'_df{self.var_name}_dt', dif_eq_code))
        # needed variables
        need_vars = tools.get_identifiers(dif_eq_code)
        need_vars |= tools.get_identifiers(', '.join(self.returns))
        # get the total return expressions
        for expr in self.expressions[::-1]:
            if expr.var_name in need_vars:
                if not profile._substitute_equation or expr._substituted_code is None:
                    code = expr.code
                else:
                    code = expr._substituted_code
                return_expressions.append(Expression(expr.var_name, code))
                need_vars |= tools.get_identifiers(code)
        return return_expressions[::-1]

    def get_g_expressions(self):
        if self.is_functional_noise:
            return_expressions = []
            # the derivative expression
            eq_code = self.g_expr.get_code(subs=True)
            return_expressions.append(Expression(f'_dg{self.var_name}_dt', eq_code))
            # needed variables
            need_vars = tools.get_identifiers(eq_code)
            # get the total return expressions
            for expr in self.expressions[::-1]:
                if expr.var_name in need_vars:
                    if not profile._substitute_equation or expr._substituted_code is None:
                        code = expr.code
                    else:
                        code = expr._substituted_code
                    return_expressions.append(Expression(expr.var_name, code))
                    need_vars |= tools.get_identifiers(code)
            return return_expressions[::-1]
        else:
            return [Expression(f'_dg{self.var_name}_dt', self.g_expr.get_code(subs=True))]

    def _replace_expressions(self, expressions, name, y_sub, t_sub=None):
        """Replace expressions of df part.

        Parameters
        ----------
        expressions : list, tuple
            The list/tuple of expressions.
        name : str
            The name of the new expression.
        y_sub : str
            The new name of the variable "y".
        t_sub : str, optional
            The new name of the variable "t".

        Returns
        -------
        list_of_expr : list
            A list of expressions.
        """
        return_expressions = []

        # replacements
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
        """Replace expressions of df part.

        Parameters
        ----------
        name : str
            The name of the new expression.
        y_sub : str
            The new name of the variable "y".
        t_sub : str, optional
            The new name of the variable "t".

        Returns
        -------
        list_of_expr : list
            A list of expressions.
        """
        return self._replace_expressions(self.get_f_expressions(),
                                         name=name, y_sub=y_sub, t_sub=t_sub)

    def replace_g_expressions(self, name, y_sub, t_sub=None):
        if self.is_functional_noise:
            return self._replace_expressions(self.get_g_expressions(),
                                             name=name, y_sub=y_sub, t_sub=t_sub)
        else:
            return []

    @property
    def is_multi_return(self):
        return len(self.returns) > 0

    @property
    def is_stochastic(self):
        if self.g_expr is not None:
            try:
                if eval(self.g_expr.code, self.func_scope) == 0.:
                    return False
            except Exception as e:
                pass
            return True
        else:
            return False

    @property
    def is_functional_noise(self):
        return self.g_type == constants.FUNCTIONAL_NOISE

    @property
    def stochastic_type(self):
        if not self.is_stochastic:
            return None
        else:
            pass

    @property
    def expr_names(self):
        return [expr.var_name for expr in self.expressions]
