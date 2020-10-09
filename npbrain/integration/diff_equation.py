# -*- coding: utf-8 -*-

import inspect
import re
from collections import Counter
from collections import OrderedDict

import sympy

from .sympy_tools import str_to_sympy
from .sympy_tools import sympy_to_str
from .. import profile
from .. import tools

__all__ = [
    'ReturnExps',
    'Expression',
    'DiffEquation',
]


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


class ReturnExps(object):
    def __init__(self, return_expressions):
        self._returns = return_expressions

    @property
    def code_line(self):
        return ', '.join(self._returns)

    @property
    def identifiers(self):
        all_vars = []
        for expr in self._returns:
            all_vars.extend(tools.get_identifiers(expr))
        return set(all_vars)

    def __str__(self):
        return str(self._returns)

    def __repr__(self):
        return f"Returns({str(self._returns)})"


class Expression(object):
    def __init__(self, type, var, code):
        # attributes
        self.var = var
        self.type = type
        self.code = code.strip()
        self._substituted_code = None

    @property
    def identifiers(self):
        return tools.get_identifiers(self.code)

    def __str__(self):
        if self.type == _DIFF_EQUATION:
            s = 'd' + self.var + '/dt'
        else:
            s = self.var
        return f'{s} = {self.code}'

    def __repr__(self):
        return f'<{self.type} {self.var}: {self.code}'

    def __eq__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if self.code != other.code:
            return False
        if self.type != other.type:
            return False
        if self.var != other.var:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class DiffEquation(object):
    """Differential Equation.
    """

    def __init__(self, f=None, g=None, analyse=False):
        # get functions
        self.f = f
        self.g = g
        self.f_code = tools.get_main_code(f)
        self.g_code = tools.get_main_code(g)
        self.func_args = inspect.getfullargspec(f).args
        self.var = self.func_args[0]
        scope = inspect.getclosurevars(f)
        self.func_scope = dict(scope.nonlocals)
        self.func_scope.update(scope.globals)
        if callable(g):
            scope = inspect.getclosurevars(g)
            self.func_scope.update(scope.nonlocals)
            self.func_scope.update(scope.globals)
        self.analyse = analyse

        if analyse:
            # check
            expressions = []
            if 'return' not in self.f_code:
                raise ValueError(f'Bad function definition: no result returned in the function.\n\n'
                                 f'Please check function: \n{f}')

            # get code lines
            code_lines = re.sub(r'\b' + r'return' + r'\b', f'd{self.var}dt =', self.f_code)
            code_lines = code_lines.strip()
            if code_lines == '':
                raise ValueError('Empty function.')
            code_lines = code_lines.replace(';', '\n').split('\n')

            # analyse code lines
            for line in code_lines:
                # skip empty lines
                expression = line.strip()
                if expression == '':
                    continue
                # remove comments
                com = expression.split('#')
                if len(com) > 1:
                    expression = com[0]
                    if expression.strip() == '':
                        continue

                # Split the equation around operators = += -= *= /=, but not ==
                split_operators = re.findall('([\s\w\+\-\*\/\)]+)=([^=])', expression)

                # definition of a new variable
                if len(split_operators) == 1:
                    # Retrieve the name
                    eq = split_operators[0][0]
                    if eq.strip() == "":
                        raise ValueError('The equation can not be analysed, check the syntax.')
                    name = extract_name(eq, left=True)
                    if name in ['_undefined', '']:
                        raise ValueError(f'No variable name can be found in "{expression}".')
                    # Append the result
                    expressions.append({'var': name, 'type': _SUB_EXPRESSION, 'code': expression.strip()})

                # Continuation of the equation on a new line:
                # append the equation to the previous variable
                elif len(split_operators) == 0:
                    expressions[-1]['code'] += ' ' + expression.strip()
                else:
                    raise ValueError(f'Error syntax in "{expression}".\nOnly one assignment operator'
                                     f' is allowed per equation, but found {len(split_operators)}.')
            expressions[-1]['type'] = _DIFF_EQUATION

            # analyse returns
            return_expr = expressions[-1]['code'].replace(f'd{self.var}dt =', '').strip()
            if return_expr[0] == '(' and return_expr[-1] == ')':
                return_expr = return_expr[1:-1]
            return_splits = re.split(r'(?<!\(),(?![\w\s]*[\)])', return_expr)
            for sp in return_splits:
                if sp.strip() == '':
                    raise ValueError('Function return error: contains null item.\n\n'
                                     'You can code like "return a" or "return a, b", not "return a, "')
            return_expressions = [self.var]
            if len(return_splits) != 1:
                assert len(return_splits) > 1
                expressions[-1]['code'] = f'd{self.var}dt =' + return_splits[0]
                for rt in return_splits[1:]:
                    return_expressions.append(rt)

            # get the right-hand expression
            for expr in expressions:
                splits = re.split(r'([\s\+\-\*\/])=(?!=)', expr['code'])
                assert len(splits) == 3, f'Unknown expression "{expr["code"]}"'
                if splits[1].strip() == '':
                    expr['code'] = splits[2]
                else:
                    assert splits[1].strip() in ['+', '-', '*', '/']
                    expr['code'] = f"{expr['var']} {splits[1]} {splits[2]}"

            # check duplicate names
            counter = Counter([v['var'] for v in expressions])
            for k, num in counter.items():
                if num > 1:
                    raise SyntaxError(
                        f'Found "{k}" {num} times. Please assign each expression with a unique name. ')

            # return values
            self.expressions = [Expression(**expr) for expr in expressions]
            self.return_expressions = ReturnExps(return_expressions)
            self.var2expr = {expr.var: expr for expr in self.expressions}
            self.vars = [expr.var for expr in self.expressions]
            self.vars_in_returns = []
            for expr in self.expressions:
                if expr.var in self.return_expressions.identifiers:
                    self.vars_in_returns.append(expr.var)

    def substitute(self, include_subexpressions=True):
        """
        Return a list of ``(varname, expr)`` tuples, containing all
        differential equations (and optionally subexpressions) with all the
        subexpression variables substituted with the respective expressions.

        Parameters
        ----------
        include_subexpressions : bool
            Whether also to return substituted subexpressions. Default is ``True``.

        Returns
        -------
        expr_tuples : list of (str, `CodeString`)
            A list of ``(varname, expr)`` tuples, where ``expr`` is a
            `CodeString` object with all subexpression variables substituted
            with the respective expression.
        """

        # get variable dependent on "key"
        dependencies = []
        for expr in self.expressions[:-1]:
            if self.var in expr.identifiers:
                dependencies.append(expr)

        # substitute dependent variables into the expresion
        for expr in self.expressions[:-1]:
            substitutions = {}
            for dep in dependencies:
                if dep.var != expr.var and dep.var in expr.identifiers:
                    code = dep.code if dep._substituted_code is None else dep._substituted_code
                    substitutions[sympy.Symbol(dep.var, real=True)] = str_to_sympy(code)
            if len(substitutions):
                new_sympy_expr = str_to_sympy(expr.code).xreplace(substitutions)
                new_str_expr = sympy_to_str(new_sympy_expr)
                expr._substituted_code = new_str_expr
                dependencies.append(expr)

        # get the final differential equation
        substitutions = {}
        for dep in dependencies:
            code = dep.code if dep._substituted_code is None else dep._substituted_code
            substitutions[sympy.Symbol(dep.var, real=True)] = str_to_sympy(code)
        expr = self.expressions[-1]
        if len(substitutions):
            new_sympy_expr = str_to_sympy(expr.code).xreplace(substitutions)
            new_str_expr = sympy_to_str(new_sympy_expr)
            expr._substituted_code = new_str_expr

        # return
        subs_expressions = OrderedDict()
        code = self.expressions[-1].code if self.expressions[-1]._substituted_code is None \
            else self.expressions[-1]._substituted_code
        subs_expressions[f'd{self.var}dt'] = Expression(_DIFF_EQUATION, self.var, code)
        if include_subexpressions:
            code = self.expressions[-1].code if self.expressions[-1]._substituted_code is None \
                else self.expressions[-1]._substituted_code
            identifiers = tools.get_identifiers(code)
            identifiers.update(self.return_expressions.identifiers)
            for expr in self.expressions[::-1]:
                if expr.var in identifiers and expr.var not in subs_expressions:
                    code = expr.code if not profile.substitute_eqs or expr._substituted_code is None \
                        else expr._substituted_code
                    subs_expressions[expr.var] = Expression(_SUB_EXPRESSION, expr.var, code)
                    identifiers.update(tools.get_identifiers(code))
        # return list(subs_expressions.items())[::-1]
        return OrderedDict(list(subs_expressions.items())[::-1])

    @property
    def type(self):
        return _SDE_TYPE if self.is_stochastic else _ODE_TYPE

    @property
    def is_stochastic(self):
        return self.g is not None

    @property
    def is_multi_return(self):
        return len(self.vars_in_returns) > 1


