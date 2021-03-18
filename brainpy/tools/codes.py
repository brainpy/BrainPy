# -*- coding: utf-8 -*-

import ast
import re
from types import LambdaType

from brainpy import errors
from .ast2code import ast2code

__all__ = [
    # tools for code string
    'get_identifiers',
    'indent',
    'deindent',
    'word_replace',

    # other tools
    'NoiseHandler',
    'FindAtomicOp',
    'find_atomic_op',
    'is_lambda_function',
]


######################################
# String tools
######################################


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

    _ID_KEYWORDS = {'and', 'or', 'not', 'True', 'False'}
    identifiers = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_.]*\b', expr))
    # identifiers = set(re.findall(r'\b[A-Za-z_][.?[A-Za-z0-9_]*]*\b', expr))
    if include_numbers:
        # only the number, not a + or -
        pattern = r'(?<=[^A-Za-z_])[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|^[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
        numbers = set(re.findall(pattern, expr))
    else:
        numbers = set()
    return (identifiers - _ID_KEYWORDS) | numbers


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
        # expr = re.sub(r'\b' + var + r'\b', str(replace_var), expr)
        expr = re.sub(r'\b(?<!\.)' + var + r'\b(?!\.)', str(replace_var), expr)
    return expr


######################################
# Other tools
######################################


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
    return isinstance(func, LambdaType) and func.__name__ == "<lambda>"


class NoiseHandler(object):
    normal_pattern = re.compile(r'(_normal_like_)\((\w+)\)')

    @staticmethod
    def vector_replace_f(m):
        return 'numpy.random.normal(0., 1., ' + m.group(2) + '.shape)'

    @staticmethod
    def scalar_replace_f(m):
        return 'numpy.random.normal(0., 1.)'

    @staticmethod
    def cuda_replace_f(m):
        return 'xoroshiro128p_normal_float64(rng_states, _obj_i)'


class FindAtomicOp(ast.NodeTransformer):
    def __init__(self, var2idx):
        self.var2idx = var2idx
        self.left = None
        self.right = None

    def visit_Assign(self, node):
        targets = node.targets
        try:
            assert len(targets) == 1
        except AssertionError:
            raise errors.DiffEqError('Do not support multiple assignment.')
        left = ast2code(ast.fix_missing_locations(targets[0]))
        key = targets[0].slice.value.s
        value = targets[0].value.id
        if node.value.__class__.__name__ == 'BinOp':
            r_left = ast2code(ast.fix_missing_locations(node.value.left))
            r_right = ast2code(ast.fix_missing_locations(node.value.right))
            op = ast2code(ast.fix_missing_locations(node.value.op))
            if op not in ['+', '-']:
                # raise ValueError(f'Unsupported operation "{op}" for {left}.')
                return node
            self.left = f'{value}[{self.var2idx[key]}]'
            if r_left == left:
                if op == '+':
                    self.right = r_right
                if op == '-':
                    self.right = f'- {r_right}'
            elif r_left == '-' + left:
                if op == '+':
                    self.right = f"2 * {left} + {r_right}"
                if op == '-':
                    self.right = f"2 * {left} - {r_right}"
            elif r_right == left:
                if op == '+':
                    self.right = r_left
                if op == '-':
                    self.right = f"{r_left} + 2 * {left}"
            elif r_right == '-' + left:
                if op == '+':
                    self.right = f"{r_left} + 2 * {left}"
                if op == '-':
                    self.right = r_left
            else:
                return node
        return node

    def visit_AugAssign(self, node):
        op = ast2code(ast.fix_missing_locations(node.op))
        expr = ast2code(ast.fix_missing_locations(node.value))
        if op not in ['+', '-']:
            # left = ast2code(ast.fix_missing_locations(node.target))
            # raise ValueError(f'Unsupported operation "{op}" for {left}.')
            return node

        key = node.target.slice.value.s
        value = node.target.value.id

        self.left = f'{value}[{self.var2idx[key]}]'
        if op == '+':
            self.right = expr
        if op == '-':
            self.right = f'- {expr}'

        return node


def find_atomic_op(code_line, var2idx):
    tree = ast.parse(code_line.strip())
    formatter = FindAtomicOp(var2idx)
    formatter.visit(tree)
    return formatter
