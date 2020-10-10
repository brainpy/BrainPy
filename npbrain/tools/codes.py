# -*- coding: utf-8 -*-

import ast
import inspect
import re
import types

import autopep8

from .ast2code import ast2code
from .. import _numpy as np

__all__ = [
    'is_lambda_function',
    'func_replace',
    'FuncFinder',
    'get_identifiers',
    'get_main_code',
    'get_line_indent',
    'get_code_lines',
    'indent',
    'deindent',
    'word_replace',


    'analyse_diff_eq',
    'DiffEquationAnalyser',
    'DiffEquationError',
]


def is_lambda_function(func):
    """Check whether the function is a ``lambda`` function. Comes from
    https://stackoverflow.com/questions/23852423/how-to-check-that-variable-is-a-lambda-function

    Parameters
    ----------
    func : callable function
        The function.

    Returns
    -------
    bool
        True of False.
    """
    return isinstance(func, types.LambdaType) and func.__name__ == "<lambda>"


_ID_KEYWORDS = {'and', 'or', 'not', 'True', 'False'}


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
    return (identifiers - _ID_KEYWORDS) | numbers


def analyse_diff_eq(eq_code):
    if eq_code.strip() == '':
        return [], [], ['0']
    else:
        tree = ast.parse(eq_code)
        analyser = DiffEquationAnalyser()
        analyser.visit(tree)
        return analyser.variables, analyser.expressions, analyser.returns


class DiffEquationAnalyser(ast.NodeTransformer):
    expression_ops = {
        'Add': '+', 'Sub': '-', 'Mult': '*', 'Div': '/',
        'Mod': '%', 'Pow': '**', 'BitXor': '^', 'BitAnd': '&',
    }

    # TODO : Multiple assignment like "a = b = 1" or "a, b = f()"
    def __init__(self):
        self.variables = []
        self.expressions = []
        self.returns = []

    def visit_Assign(self, node):
        targets = node.targets
        assert len(targets) == 1, 'Do not support multiple assignment.'
        self.variables.append(targets[0].id)
        self.expressions.append(ast2code(ast.fix_missing_locations(node.value)))
        return node

    def visit_AugAssign(self, node):
        targets = node.targets
        assert len(targets) == 1, 'Do not support multiple assignment.'
        var = targets[0].id
        self.variables.append(var)
        op = node.op
        expr = ast2code(ast.fix_missing_locations(node.value))
        self.expressions.append(f"{var} {op} ({expr})")
        return node

    def visit_Return(self, node):
        value = node.value
        if isinstance(value, (ast.Tuple, ast.List)):
            v0 = value.elts[0]
            if isinstance(v0, ast.Name):
                self.returns.append(v0.id)
            else:
                self.expressions.append(ast2code(ast.fix_missing_locations(v0)))
                self.variables.append("_func_res_")
                self.returns.append("_func_res_")
            for i, item in enumerate(value.elts[1:]):
                if isinstance(item, ast.Name):
                    self.returns.append(item.id)
                else:
                    self.returns.append(ast2code(ast.fix_missing_locations(item)))
        elif isinstance(value, ast.Name):
            self.returns.append(value.id)
        else:
            self.expressions.append(ast2code(ast.fix_missing_locations(value)))
            self.variables.append("_func_res_")
            self.returns.append("_func_res_")
        return node

    def visit_If(self, node):
        raise DiffEquationError('Do not support "if" statement in differential equation.')

    def visit_IfExp(self, node):
        raise DiffEquationError('Do not support "if" expression in differential equation.')

    def visit_For(self, node):
        raise DiffEquationError('Do not support "for" loop in differential equation.')

    def visit_While(self, node):
        raise DiffEquationError('Do not support "while" loop in differential equation.')

    def visit_Try(self, node):
        raise DiffEquationError('Do not support "try" handler in differential equation.')


class DiffEquationError(Exception):
    pass


def func_replace(code, func_name):
    tree = ast.parse(code.strip())
    w = FuncFinder(func_name)
    tree = w.visit(tree)
    tree = ast.fix_missing_locations(tree)
    new_code = ast2code(tree)
    return new_code, w.args, w.kwargs


class FuncFinder(ast.NodeTransformer):
    """Wraps all integers in a call to Integer()"""

    def __init__(self, func_name):
        self.name = func_name
        self.args = []
        self.kwargs = {}

    def _get_attr_value(self, node, names):
        if hasattr(node, 'value'):
            names.insert(0, node.attr)
            return self._get_attr_value(node.value, names)
        else:
            assert hasattr(node, 'id')
            names.insert(0, node.id)
            return names

    def visit_Call(self, node):
        if getattr(node, 'starargs', None) is not None:
            raise ValueError("Variable number of arguments not supported")
        if getattr(node, 'kwargs', None) is not None:
            raise ValueError("Keyword arguments not supported")

        if hasattr(node.func, 'id') and node.func.id == self.name:
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    self.args.append(arg.id)
                elif isinstance(arg, ast.Num):
                    self.args.append(arg.n)
                else:
                    s = ast2code(ast.fix_missing_locations(arg))
                    self.args.append(s.strip())
            for kv in node.keywords:
                if isinstance(kv.value, ast.Name):
                    self.kwargs[kv.arg] = kv.value.id
                elif isinstance(kv.value, ast.Num):
                    self.kwargs[kv.arg] = kv.value.n
                else:
                    s = ast2code(ast.fix_missing_locations(kv.value))
                    self.kwargs[kv.arg] = s.strip()
            return ast.Name(f'_{self.name}_res')
        else:
            args = [self.visit(arg) for arg in node.args]
            keywords = [self.visit(kv) for kv in node.keywords]
            return ast.Call(func=node.func, args=args, keywords=keywords)


def get_main_code(func):
    """Get the main function _code string.

    For lambda function, return the

    Parameters
    ----------
    func : callable, Optional, int, float

    Returns
    -------

    """
    if func is None:
        return ''
    elif callable(func):
        if is_lambda_function(func):
            func_code = inspect.getsource(func)
            splits = func_code.split(':')
            if len(splits) != 2:
                raise ValueError(f'Can not parse function: \n{func_code}')
            return f'return {splits[1]}'

        else:
            func_codes = inspect.getsourcelines(func)[0]
            idx = 0
            for i, line in enumerate(func_codes):
                idx += 1
                line = line.replace(' ', '')
                if '):' in line:
                    break
            return ''.join(func_codes[idx:])
    else:
        if isinstance(func, (int, float)):
            return str(func)
        elif isinstance(func, np.ndarray):
            return '_g'
        else:
            raise ValueError(f'Unknown function type: {type(func)}.')


def extract_name(equation, left=False):
    """Extracts the name of a parameter/variable by looking the left term of an equation."""

    equation = equation.replace(' ', '')

    name = equation.strip()
    # Search for increments
    operators = ['+', '-', '*', '/']
    for op in operators:
        if equation.endswith(op):
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


def get_line_indent(line, spaces_per_tab=4):
    line = line.replace('\t', ' ' * spaces_per_tab)
    return len(line) - len(line.lstrip())


_LINE_KEYWORDS = ('print', 'raise', 'del', 'yield', 'if ', 'elif ', 'while ', 'for ')


def get_code_lines(code_string):
    """Get _code lines from the string.

    Parameters
    ----------
    code_string

    Returns
    -------
    code_lines : list
    """
    code_lines = []

    code_string = autopep8.fix_code(deindent(code_string))
    code_splits = code_string.split('\n')

    # analyse _code lines
    for line_no, line in enumerate(code_splits):
        # skip empty lines
        if line.strip() == '':
            continue
        # remove comments
        com = line.split('#')
        if len(com) > 1:
            line = com[0]
            if line.strip() == '':
                continue

        # Split the equation around operators = += -= *= /=, but not ==
        # split_operators = re.findall(r'(?<![\(,])([\s\w\+\-\*\/\)]+)=([^=])(?![\w\s]*[\),])', line)
        split_operators = re.findall(r'(?<![\(,])([\s\[\]\'\"\w\+\-\*\/\)]+)=([^=])(?![\w\s]*[\),])', line)

        # definition of a new variable
        if len(split_operators) == 1:
            # Retrieve the name
            eq = split_operators[0][0]
            if eq.strip() == "":
                raise ValueError('The equation can not be analysed, check the syntax.')
            code_lines.append(line)
        else:
            if len(split_operators) == 0:
                line_strip = line.strip()
                if ':' in line or \
                        line_strip in ['continue', 'break', 'pass', 'print'] or \
                        line_strip.startswith(_LINE_KEYWORDS) or \
                        (line_no > 0 and get_line_indent(line) ==
                         get_line_indent(code_splits[line_no - 1])):
                    code_lines.append(line)
                else:
                    code_lines[-1] += ' ' + line
            else:
                raise ValueError(f'Error syntax in "{line}".')

    return code_lines


######################################
# String tools
######################################


def indent(text, num_tabs=1, spaces_per_tab=4, tab=None):
    if tab is None:
        tab = ' ' * spaces_per_tab
    indent_ = tab * num_tabs
    indented_string = indent_ + text.replace('\n', '\n' + indent_)
    return indented_string


def deindent(text, num_tabs=None, spaces_per_tab=4, docstring=False):
    text = text.replace('\t', ' ' * spaces_per_tab)
    lines = text.split('\n')
    # if it's a docstring, we search for the common tabulation starting from
    # line 1, otherwise we use all lines
    if docstring:
        start = 1
    else:
        start = 0
    if docstring and len(lines) < 2:  # nothing to do
        return text
    # Find the minimum indentation level
    if num_tabs is not None:
        indent_level = num_tabs * spaces_per_tab
    else:
        line_seq = [len(line) - len(line.lstrip()) for line in lines[start:] if len(line.strip())]
        if len(line_seq) == 0:
            indent_level = 0
        else:
            indent_level = min(line_seq)
    # remove the common indentation
    lines[start:] = [line[indent_level:] for line in lines[start:]]
    return '\n'.join(lines)


def word_replace(expr, substitutions):
    """Applies a dict of word substitutions.

    The dict ``substitutions`` consists of pairs ``(word, rep)`` where each
    word ``word`` appearing in ``expr`` is replaced by ``rep``. Here a 'word'
    means anything matching the regexp ``\\bword\\b``.

    Examples
    --------

    >>> expr = 'a*_b+c5+8+f(A)'
    >>> print(word_replace(expr, {'a':'banana', 'f':'func'}))
    banana*_b+c5+8+func(A)
    """
    for var, replace_var in substitutions.items():
        expr = re.sub(r'\b' + var + r'\b', str(replace_var), expr)
    return expr


def replace(s, substitutions):
    """
    Applies a dictionary of substitutions. Simpler than `word_substitute`, it
    does not attempt to only replace words
    """
    for before, after in substitutions.items():
        s = s.replace(before, after)
    return s


def strip_empty_lines(s):
    '''
    Removes all empty lines from the multi-line string `s`.

    Examples
    --------

    >>> multiline = """A string with
    ...
    ... an empty line."""
    >>> print(strip_empty_lines(multiline))
    A string with
    an empty line.
    '''
    return '\n'.join(line for line in s.split('\n') if line.strip())


def strip_empty_leading_and_trailing_lines(s):
    """
    Removes all empty leading and trailing lines in the multi-line string `s`.
    """
    lines = s.split('\n')
    while lines and not lines[0].strip():  del lines[0]
    while lines and not lines[-1].strip(): del lines[-1]
    return '\n'.join(lines)


def stripped_deindented_lines(code):
    """
    Returns a list of the lines in a multi-line string, deindented.
    """
    code = deindent(code)
    code = strip_empty_lines(code)
    lines = code.split('\n')
    return lines


def code_representation(code):
    """
    Returns a string representation for several different formats of _code

    Formats covered include:
    - A single string
    - A list of statements/strings
    - A dict of strings
    - A dict of lists of statements/strings
    """
    if not isinstance(code, (str, list, tuple, dict)):
        code = str(code)
    if isinstance(code, str):
        return strip_empty_leading_and_trailing_lines(code)
    if not isinstance(code, dict):
        code = {None: code}
    else:
        code = code.copy()
    for k, v in code.items():
        if isinstance(v, (list, tuple)):
            v = '\n'.join([str(line) for line in v])
            code[k] = v
    if len(code) == 1 and list(code.keys())[0] is None:
        return strip_empty_leading_and_trailing_lines(list(code.values())[0])
    output = []
    for k, v in code.items():
        msg = 'Key %s:\n' % k
        msg += indent(str(v))
        output.append(msg)
    return strip_empty_leading_and_trailing_lines('\n'.join(output))
