import inspect
import re
import sympy
from npbrain import helper
from npbrain.integrator.sympytools import str_to_sympy
from npbrain.integrator.sympytools import sympy_to_str
from collections import Counter
from collections import OrderedDict

DIFF_EQUATION = 'diff_equation'
SUB_EXPRESSION = 'sub_expression'



KEYWORDS = {'and', 'or', 'not', 'True', 'False'}


def get_identifiers(expr, include_numbers=False):
    """
    Return all the identifiers in a given string ``expr``, that is everything
    that matches a programming language variable like expression, which is
    here implemented as the regexp ``\\b[A-Za-z_][A-Za-z0-9_]*\\b``.

    Parameters
    ----------
    expr : str
        The string to analyze
    include_numbers : bool, optional
        Whether to include number literals in the output. Defaults to ``False``.

    Returns
    -------
    identifiers : set
        A set of all the identifiers (and, optionally, numbers) in `expr`.

    Examples
    --------
    >>> expr = '3-a*_b+c5+8+f(A - .3e-10, tau_2)*17'
    >>> ids = get_identifiers(expr)
    >>> print(sorted(list(ids)))
    ['A', '_b', 'a', 'c5', 'f', 'tau_2']
    >>> ids = get_identifiers(expr, include_numbers=True)
    >>> print(sorted(list(ids)))
    ['.3e-10', '17', '3', '8', 'A', '_b', 'a', 'c5', 'f', 'tau_2']
    """
    identifiers = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_.]*\b', expr))
    # identifiers = set(re.findall(r'\b[A-Za-z_][.?[A-Za-z0-9_]*]*\b', expr))
    if include_numbers:
        # only the number, not a + or -
        pattern = r'(?<=[^A-Za-z_])[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|^[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
        numbers = set(re.findall(pattern, expr))
    else:
        numbers = set()
    return (identifiers - KEYWORDS) | numbers


def get_source_code(func):
    if func is None:
        return None

    if helper.is_lambda_function(func):
        func_code = inspect.getsource(func)
        splits = func_code.split(':')
        if len(splits) != 2:
            raise ValueError(f'Can not parse function: \n{func_code}')
        return f'return {splits[1]}'

    else:
        func_codes = inspect.getsourcelines(func)[0]
        idx = 0
        for i, line in enumerate(func_codes):
            line = line.replace(' ', '')
            if '):' in line:
                break
            idx += 1
        return ''.join(func_codes[idx + 1:])


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


class Returns(object):
    def __init__(self, return_expressions):
        self.returns = return_expressions

    @property
    def identifiers(self):
        all_vars = []
        for expr in self.returns:
            all_vars.extend(get_identifiers(expr))
        return set(all_vars)

    def __str__(self):
        return str(self.returns)

    def __repr__(self):
        return f"Returns({str(self.returns)})"


class Expression(object):
    def __init__(self, type, var, code):
        # try to convert it to a sympy expression
        str_to_sympy(code)
        # attributes
        self.var = var
        self.type = type
        self.code = code.strip()
        self._substituted_code = None

    @property
    def identifiers(self):
        return get_identifiers(self.code)

    def __str__(self):
        if self.type == DIFF_EQUATION:
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
    def __init__(self, f, g=None, use_substituted_eqs=False):
        # get functions
        self.f = f
        self.g = g
        self.use_subs_eqs = use_substituted_eqs
        self.func_args = inspect.getfullargspec(f).args
        self.var = self.func_args[0]
        self.f_code = get_source_code(f)
        self.g_code = get_source_code(g)

        # check
        expressions = []
        if 'return' not in self.f_code:
            raise ValueError('No result returned in the function.')
        code_lines = re.sub(r'\b' + r'return' + r'\b', f'd{self.var}dt =', self.f_code)
        code_lines = code_lines.strip()
        if code_lines == '':
            raise ValueError('Empty function.')
        code_lines = code_lines.replace(';', '\n').split('\n')

        # Iterate over all lines
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
                    print(expression)
                    raise ValueError('The equation can not be analysed, check the syntax.')
                name = extract_name(eq, left=True)
                if name in ['_undefined', '']:
                    raise ValueError(f'No variable name can be found in "{expression}".')
                # Append the result
                expressions.append({'var': name, 'type': SUB_EXPRESSION, 'code': expression.strip()})

            # Continuation of the equation on a new line: append
            # the equation to the previous variable
            elif len(split_operators) == 0:
                expressions[-1]['code'] += ' ' + expression.strip()

            else:
                raise ValueError(f'Error syntax in "{expression}".\nOnly one assignment operator'
                                 f' is allowed per equation, but found {len(split_operators)}.')
        expressions[-1]['type'] = DIFF_EQUATION

        # analyse returns
        return_expr = re.split(r'(?<!\(),(?![\w\s]*[\)])', expressions[-1]['code'])
        for expr in return_expr:
            if expr.strip() == '':
                raise ValueError('Function return contains null item, please check.')
        return_expressions = [self.var]
        if len(return_expr) == 1:
            pass
        else:
            assert len(return_expr) > 1
            expressions[-1]['code'] = return_expr[0]
            for rt in return_expr[1:]:
                return_expressions.append(rt)

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
                raise SyntaxError(f'Found "{k}" {num} times. Please assign each expression with a unique name. ')

        # return values
        self.expressions = [Expression(**expr) for expr in expressions]
        self.returns = Returns(return_expressions)
        self.vars_in_returns = []
        self.vars = [expr.var for expr in self.expressions]
        self.var2exp = {expr.var: expr for expr in self.expressions}
        for expr in self.expressions:
            if expr.var in self.returns.identifiers:
                self.vars_in_returns.append(expr.var)

    def substitute(self, variables=None, include_subexpressions=False):
        """
        Return a list of ``(varname, expr)`` tuples, containing all
        differential equations (and optionally subexpressions) with all the
        subexpression variables substituted with the respective expressions.

        Parameters
        ----------
        variables : dict, optional
            A mapping of variable names to `Variable`/`Function` objects.
        include_subexpressions : bool
            Whether also to return substituted subexpressions. Default is ``False``.

        Returns
        -------
        expr_tuples : list of (str, `CodeString`)
            A list of ``(varname, expr)`` tuples, where ``expr`` is a
            `CodeString` object with all subexpression variables substituted
            with the respective expression.
        """
        _substituted_expressions = []

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
                    substitutions[sympy.Symbol(dep.var, real=True)] = str_to_sympy(code, variables)
            if len(substitutions):
                new_sympy_expr = str_to_sympy(expr.code, variables).xreplace(substitutions)
                new_str_expr = sympy_to_str(new_sympy_expr)
                expr._substituted_code = new_str_expr
                dependencies.append(expr)

        # get the final differential equation
        substitutions = {}
        for dep in dependencies:
            code = dep.code if dep._substituted_code is None else dep._substituted_code
            substitutions[sympy.Symbol(dep.var, real=True)] = str_to_sympy(code, variables)
        expr = self.expressions[-1]
        if len(substitutions):
            new_sympy_expr = str_to_sympy(expr.code, variables).xreplace(substitutions)
            new_str_expr = sympy_to_str(new_sympy_expr)
            expr._substituted_code = new_str_expr

        # return
        subs_expressions = OrderedDict()
        code = self.expressions[-1].code if self.expressions[-1]._substituted_code is None \
            else self.expressions[-1]._substituted_code
        subs_expressions[self.var] = Expression(DIFF_EQUATION, self.var, code)
        if include_subexpressions:
            code = self.expressions[-1].code if self.expressions[-1]._substituted_code is None \
                else self.expressions[-1]._substituted_code
            identifiers = get_identifiers(code)
            identifiers.update(self.returns.identifiers)
            for expr in self.expressions[::-1]:
                if expr.var in identifiers and expr.var not in subs_expressions:
                    code = expr.code if not self.use_subs_eqs or expr._substituted_code is None \
                        else expr._substituted_code
                    subs_expressions[expr.var] = Expression(SUB_EXPRESSION, expr.var, code)
                    identifiers.update(get_identifiers(code))
        return list(subs_expressions.values())[::-1]

    @property
    def is_stochastic(self):
        return self.g is not None


def try_analyse_func():
    import numpy as np

    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m, alpha, beta

    from pprint import pprint

    df = DiffEquation(int_m)
    pprint(df.expressions)
    pprint(df.returns)
    pprint(df.substitute(include_subexpressions=True))


def try_analyse_func2():
    def func(m, t):
        a = t + 2
        b = m + 6
        c = a + b
        d = m * 2 + c
        return d, c

    from pprint import pprint

    df = DiffEquation(func, use_substituted_eqs=False)
    pprint(df.expressions)
    pprint(df.returns)
    pprint(df.substitute(include_subexpressions=False))


if __name__ == '__main__':
    try_analyse_func()
    # try_analyse_func2()
