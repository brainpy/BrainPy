# -*- coding: utf-8 -*-

import ast
import inspect
import re
from types import LambdaType

from .ast2code import ast2code
from .dicts import DictPlus
from ..errors import CodeError
from ..errors import DiffEquationError

__all__ = [
    'CodeLineFormatter',
    'format_code',

    'LineFormatterForTrajectory',
    'format_code_for_trajectory',

    'FindAtomicOp',
    'find_atomic_op',

    # replace function calls
    'replace_func',
    'FuncCallFinder',

    # analyse differential equations
    'analyse_diff_eq',
    'DiffEquationAnalyser',

    # string processing
    'get_identifiers',
    'get_main_code',
    'get_line_indent',

    'indent',
    'deindent',
    'word_replace',

    # others
    'is_lambda_function',

    #
    'func_call',
    'get_func_source',
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
    return isinstance(func, LambdaType) and func.__name__ == "<lambda>"


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


class DiffEquationAnalyser(ast.NodeTransformer):
    expression_ops = {
        'Add': '+', 'Sub': '-', 'Mult': '*', 'Div': '/',
        'Mod': '%', 'Pow': '**', 'BitXor': '^', 'BitAnd': '&',
    }

    def __init__(self):
        self.variables = []
        self.expressions = []
        self.f_expr = None
        self.g_expr = None
        self.returns = []
        self.return_type = None

    # TODO : Multiple assignment like "a = b = 1" or "a, b = f()"
    def visit_Assign(self, node):
        targets = node.targets
        try:
            assert len(targets) == 1
        except AssertionError:
            raise DiffEquationError('BrainPy currently does not support multiple '
                                    'assignment in differential equation.')
        self.variables.append(targets[0].id)
        self.expressions.append(ast2code(ast.fix_missing_locations(node.value)))
        return node

    def visit_AugAssign(self, node):
        var = node.target.id
        self.variables.append(var)
        op = ast2code(ast.fix_missing_locations(node.op))
        expr = ast2code(ast.fix_missing_locations(node.value))
        self.expressions.append(f"{var} {op} {expr}")
        return node

    def visit_AnnAssign(self, node):
        raise DiffEquationError('Do not support an assignment with a type annotation.')

    def visit_Return(self, node):
        value = node.value
        if isinstance(value, (ast.Tuple, ast.List)):  # a tuple/list return
            v0 = value.elts[0]
            if isinstance(v0, (ast.Tuple, ast.List)):  # item 0 is a tuple/list
                # f expression
                if isinstance(v0.elts[0], ast.Name):
                    self.f_expr = ('_f_res_', v0.elts[0].id)
                else:
                    self.f_expr = ('_f_res_', ast2code(ast.fix_missing_locations(v0.elts[0])))

                if len(v0.elts) == 1:
                    self.return_type = '(x,),'
                elif len(v0.elts) == 2:
                    self.return_type = '(x,x),'
                    # g expression
                    if isinstance(v0.elts[1], ast.Name):
                        self.g_expr = ('_g_res_', v0.elts[1].id)
                    else:
                        self.g_expr = ('_g_res_', ast2code(ast.fix_missing_locations(v0.elts[1])))
                else:
                    raise DiffEquationError(f'The dxdt should have the format of (f, g), not '
                                            f'"({ast2code(ast.fix_missing_locations(v0.elts))})"')

                # returns
                for i, item in enumerate(value.elts[1:]):
                    if isinstance(item, ast.Name):
                        self.returns.append(item.id)
                    else:
                        self.returns.append(ast2code(ast.fix_missing_locations(item)))

            else:  # item 0 is not a tuple/list
                # f expression
                if isinstance(v0, ast.Name):
                    self.f_expr = ('_f_res_', v0.id)
                else:
                    self.f_expr = ('_f_res_', ast2code(ast.fix_missing_locations(v0)))

                if len(value.elts) == 1:
                    self.return_type = 'x,'
                elif len(value.elts) == 2:
                    self.return_type = 'x,x'
                    # g expression
                    if isinstance(value.elts[1], ast.Name):
                        self.g_expr = ('_g_res_', value.elts[1].id)
                    else:
                        self.g_expr = ("_g_res_", ast2code(ast.fix_missing_locations(value.elts[1])))
                else:
                    raise DiffEquationError('Cannot parse return expression. It should have the '
                                            'format of "(f, [g]), [return values]"')
        else:
            self.return_type = 'x'
            if isinstance(value, ast.Name):  # a name return
                self.f_expr = ('_f_res_', value.id)
            else:  # an expression return
                self.f_expr = ('_f_res_', ast2code(ast.fix_missing_locations(value)))
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

    def visit_With(self, node):
        raise DiffEquationError('Do not support "with" block in differential equation.')

    def visit_Raise(self, node):
        raise DiffEquationError('Do not support "raise" statement.')

    def visit_Delete(self, node):
        raise DiffEquationError('Do not support "del" operation.')


def analyse_diff_eq(eq_code):
    assert eq_code.strip() != ''
    tree = ast.parse(eq_code)
    analyser = DiffEquationAnalyser()
    analyser.visit(tree)

    res = DictPlus(variables=analyser.variables,
                   expressions=analyser.expressions,
                   returns=analyser.returns,
                   return_type=analyser.return_type,
                   f_expr=analyser.f_expr,
                   g_expr=analyser.g_expr)
    return res


class FuncCallFinder(ast.NodeTransformer):
    """"""

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


def replace_func(code, func_name):
    tree = ast.parse(code.strip())
    w = FuncCallFinder(func_name)
    tree = w.visit(tree)
    tree = ast.fix_missing_locations(tree)
    new_code = ast2code(tree)
    return new_code, w.args, w.kwargs


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
            func_code = get_func_source(func)
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
            else:
                code = "\n".join(func_codes)
                raise ValueError(f'Can not parse function: \n{code}')
            return ''.join(func_codes[idx:])
    else:
        raise ValueError(f'Unknown function type: {type(func)}.')


def get_line_indent(line, spaces_per_tab=4):
    line = line.replace('\t', ' ' * spaces_per_tab)
    return len(line) - len(line.lstrip())


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
            raise DiffEquationError('Do not support multiple assignment.')
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


class CodeLineFormatter(ast.NodeTransformer):
    def __init__(self):
        self.lefts = []
        self.rights = []
        self.lines = []

    def visit_Assign(self, node, level=0):
        targets = node.targets
        try:
            assert len(targets) == 1
        except AssertionError:
            raise DiffEquationError('Do not support multiple assignment.')
        target = ast2code(ast.fix_missing_locations(targets[0]))
        expr = ast2code(ast.fix_missing_locations(node.value))
        prefix = '  ' * level
        self.lefts.append(target)
        self.rights.append(expr)
        self.lines.append(f'{prefix}{target} = {expr}')
        return node

    def visit_AugAssign(self, node, level=0):
        target = ast2code(ast.fix_missing_locations(node.target))
        op = ast2code(ast.fix_missing_locations(node.op))
        expr = ast2code(ast.fix_missing_locations(node.value))
        prefix = '  ' * level
        self.lefts.append(target)
        self.rights.append(f"{target} {op} {expr}")
        self.lines.append(f"{prefix}{target} {op}= {expr}")
        return node

    def visit_AnnAssign(self, node):
        raise NotImplementedError('Do not support an assignment with a type annotation.')

    def visit_node_not_assign(self, node, level=0):
        prefix = '  ' * level
        expr = ast2code(ast.fix_missing_locations(node))
        self.lines.append(f'{prefix}{expr}')

    def visit_Assert(self, node, level=0):
        self.visit_node_not_assign(node, level)

    def visit_Expr(self, node, level=0):
        self.visit_node_not_assign(node, level)

    def visit_Expression(self, node, level=0):
        self.visit_node_not_assign(node, level)

    def visit_content_in_condition_control(self, node, level):
        if isinstance(node, ast.Expr):
            self.visit_Expr(node, level)
        elif isinstance(node, ast.Assert):
            self.visit_Assert(node, level)
        elif isinstance(node, ast.Assign):
            self.visit_Assign(node, level)
        elif isinstance(node, ast.AugAssign):
            self.visit_AugAssign(node, level)
        elif isinstance(node, ast.If):
            self.visit_If(node, level)
        elif isinstance(node, ast.For):
            self.visit_For(node, level)
        elif isinstance(node, ast.While):
            self.visit_While(node, level)
        else:
            code = ast2code(ast.fix_missing_locations(node))
            raise CodeError(f'BrainPy does not support {type(node)}.\n\n{code}')

    def visit_If(self, node, level=0):
        # If condition
        prefix = '  ' * level
        compare = ast2code(ast.fix_missing_locations(node.test))
        self.lines.append(f'{prefix}if {compare}:')
        # body
        for expr in node.body:
            self.visit_content_in_condition_control(expr, level + 1)

        # elif
        while node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node = node.orelse[0]
            compare = ast2code(ast.fix_missing_locations(node.test))
            self.lines.append(f'{prefix}elif {compare}:')
            for expr in node.body:
                self.visit_content_in_condition_control(expr, level + 1)

        # else:
        if len(node.orelse) > 0:
            self.lines.append(f'{prefix}else:')
            for expr in node.orelse:
                self.visit_content_in_condition_control(expr, level + 1)

    def visit_For(self, node, level=0):
        prefix = '  ' * level
        # target
        target = ast2code(ast.fix_missing_locations(node.target))
        # iter
        iter = ast2code(ast.fix_missing_locations(node.iter))
        self.lefts.append(target)
        self.rights.append(iter)
        self.lines.append(prefix + f'for {target} in {iter}:')
        # body
        for expr in node.body:
            self.visit_content_in_condition_control(expr, level + 1)
        # else
        if len(node.orelse) > 0:
            self.lines.append(prefix + 'else:')
            for expr in node.orelse:
                self.visit_content_in_condition_control(expr, level + 1)

    def visit_While(self, node, level=0):
        prefix = '  ' * level
        # test
        test = ast2code(ast.fix_missing_locations(node.test))
        self.rights.append(test)
        self.lines.append(prefix + f'while {test}:')
        # body
        for expr in node.body:
            self.visit_content_in_condition_control(expr, level + 1)
        # else
        if len(node.orelse) > 0:
            self.lines.append(prefix + 'else:')
            for expr in node.orelse:
                self.visit_content_in_condition_control(expr, level + 1)

    def visit_Try(self, node):
        raise CodeError('Do not support "try" handler.')

    def visit_With(self, node):
        raise CodeError('Do not support "with" block.')

    def visit_Raise(self, node):
        raise CodeError('Do not support "raise" statement.')

    def visit_Delete(self, node):
        raise CodeError('Do not support "del" operation.')


def format_code(code_string):
    """Get code lines from the string.

    Parameters
    ----------
    code_string

    Returns
    -------
    code_lines : list
    """

    tree = ast.parse(code_string.strip())
    formatter = CodeLineFormatter()
    formatter.visit(tree)
    return formatter


class LineFormatterForTrajectory(CodeLineFormatter):
    def __init__(self, fixed_vars):
        super(LineFormatterForTrajectory, self).__init__()
        self.fixed_vars = fixed_vars

    def visit_Assign(self, node, level=0):
        targets = node.targets
        try:
            assert len(targets) == 1
        except AssertionError:
            raise DiffEquationError(f'Do not support multiple assignment. \n'
                                    f'Error in code line: \n\n'
                                    f'{ast2code(ast.fix_missing_locations(node))}')
        prefix = '  ' * level
        target = targets[0]
        append_lines = []

        if isinstance(target, ast.Subscript):
            if target.value.id == 'ST' and target.slice.value.s in self.fixed_vars:
                left = ast2code(ast.fix_missing_locations(target))
                self.lefts.append(left)
                self.lines.append(f'{prefix}{left} = {self.fixed_vars[target.slice.value.s]}')
                return node

        elif hasattr(target, 'elts'):
            if len(target.elts) == 1:
                elt = target.elts[0]
                if isinstance(elt, ast.Subscript):
                    if elt.value.id == 'ST' and elt.slice.value.s in self.fixed_vars:
                        left = ast2code(ast.fix_missing_locations(elt))
                        self.lefts.append(left)
                        self.lines.append(f'{prefix}{left} = {self.fixed_vars[elt.slice.value.s]}')
                        return node
                left = ast2code(ast.fix_missing_locations(elt))
                expr = ast2code(ast.fix_missing_locations(node.value))
                self.lefts.append(left)
                self.rights.append(expr)
                self.lines.append(f'{prefix}{left} = {expr}')
                return node
            else:
                for elt in target.elts:
                    if isinstance(elt, ast.Subscript):
                        if elt.value.id == 'ST' and elt.slice.value.s in self.fixed_vars:
                            left = ast2code(ast.fix_missing_locations(elt))
                            append_lines.append(f'{prefix}{left} = {self.fixed_vars[elt.slice.value.s]}')
                left = ast2code(ast.fix_missing_locations(target))
                expr = ast2code(ast.fix_missing_locations(node.value))
                self.lefts.append(target)
                self.rights.append(expr)
                self.lines.append(f'{prefix}{left} = {expr}')
                self.lines.extend(append_lines)
                return node

        left = ast2code(ast.fix_missing_locations(target))
        expr = ast2code(ast.fix_missing_locations(node.value))
        self.lefts.append(left)
        self.rights.append(expr)
        self.lines.append(f'{prefix}{left} = {expr}')
        return node

    def visit_AugAssign(self, node, level=0):
        prefix = '  ' * level
        if isinstance(node.target, ast.Subscript):
            if node.target.value.id == 'ST' and node.target.slice.value.s in self.fixed_vars:
                left = ast2code(ast.fix_missing_locations(node.target))
                self.lefts.append(left)
                self.lines.append(f'{prefix}{left} = {self.fixed_vars[node.target.slice.value.s]}')
                return node

        op = ast2code(ast.fix_missing_locations(node.op))
        left = ast2code(ast.fix_missing_locations(node.target))
        expr = ast2code(ast.fix_missing_locations(node.value))
        self.lefts.append(left)
        self.rights.append(f"{left} {op} {expr}")
        self.lines.append(f"{prefix}{left} {op}= {expr}")
        return node


def format_code_for_trajectory(code_string, fixed_vars):
    """Get _code lines from the string.

    Parameters
    ----------
    code_string

    Returns
    -------
    code_lines : list
    """

    tree = ast.parse(code_string.strip())
    formatter = LineFormatterForTrajectory(fixed_vars)
    formatter.visit(tree)
    return formatter


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


def func_call(args):
    if isinstance(args, set):
        args = sorted(list(args))
    else:
        assert isinstance(args, (tuple, list))
    func_args = []
    for i in range(0, len(args), 5):
        for arg in args[i: i + 5]:
            func_args.append(f'{arg},')
        func_args.append('\n')
    return ' '.join(func_args).strip()


def get_func_source(func):
    code = inspect.getsource(func)
    # remove @
    try:
        start = code.index('def ')
        code = code[start:]
    except ValueError:
        pass
    return code
