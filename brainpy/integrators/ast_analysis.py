# -*- coding: utf-8 -*-

import ast
import inspect
from collections import OrderedDict

from brainpy import errors
from brainpy import tools

__all__ = [
    'DiffEqReader',
    'separate_variables',
]


class DiffEqReader(ast.NodeVisitor):
    """Read the code lines which defines the logic of a differential equation system.

    Currently, DiffEqReader cannot handle the for loop, and if-else condition.
    Also, it do not assign values by a functional call. Like this:

    .. code-block:: python

        func(a, b, c)

    Instead, you should code like:

    .. code-block:: python

        c = func(a, b)

    Therefore, this class only has minimum power to analyze differential
    equations. For example, this class may help to automatically find out
    the linear part of a differential equation, thus forming the
    Exponential Euler numerical methods.
    """

    def __init__(self):
        self.code_lines = []  # list of str
        self.variables = []  # list of list
        self.returns = []  # list of str
        self.rights = []  # list of str

    @staticmethod
    def visit_container(nodes):
        variables = []
        for var in nodes:
            if isinstance(var, (ast.List, ast.Tuple)):
                variables.extend(DiffEqReader.visit_container(var.elts))
            elif isinstance(var, ast.Name):
                variables.extend(var.id)
            else:
                raise ValueError(f'Unknown target type: {var}')
        return variables

    def visit_Assign(self, node):
        variables = []
        for target in node.targets:
            if isinstance(target, (ast.List, ast.Tuple)):
                variables.extend(self.visit_container(target.elts))
            elif isinstance(target, ast.Name):
                variables.append(target.id)
            else:
                raise ValueError(f'Unknown target type: {target}')
        self.variables.append(variables)
        self.code_lines.append(tools.ast2code(ast.fix_missing_locations(node)))
        self.rights.append(tools.ast2code(ast.fix_missing_locations(node.value)))
        return node

    def visit_AugAssign(self, node):
        var = node.target.id
        self.variables.append(var)
        expr = tools.ast2code(ast.fix_missing_locations(node))
        self.code_lines.append(expr)
        self.rights.append(tools.ast2code(ast.fix_missing_locations(node.value)))
        return node

    def visit_Return(self, node):
        if isinstance(node.value, ast.Name):
            self.returns.append(node.value.id)
        elif isinstance(node.value, (ast.Tuple, ast.List)):
            for var in node.value.elts:
                if not (var, ast.Name):
                    raise errors.DiffEqError(f'Unknown return type: {node}')
                self.returns.append(var.id)
        else:
            raise errors.DiffEqError(f'Unknown return type: {node}')
        return node

    def visit_AnnAssign(self, node):
        raise errors.DiffEqError(f'Currently, {self.__class__.__name__} do not support an '
                                 f'assignment with a type annotation.')

    def visit_If(self, node):
        raise errors.DiffEqError(f'Currently, {self.__class__.__name__} do not support to '
                                 f'analyze "if-else" conditions in differential equation.')

    def visit_IfExp(self, node):
        raise errors.DiffEqError(f'Currently, {self.__class__.__name__} do not support to '
                                 f'analyze "if-else" conditions in differential equation.')

    def visit_For(self, node):
        raise errors.DiffEqError(f'Currently, {self.__class__.__name__} do not support to '
                                 f'analyze "for" loops in differential equation.')

    def visit_While(self, node):
        raise errors.DiffEqError(f'Currently, {self.__class__.__name__} do not support to '
                                 f'analyze "while" loops in differential equation.')

    def visit_Try(self, node):
        raise errors.DiffEqError(f'Currently, {self.__class__.__name__} do not support to '
                                 f'analyze "try" handler in differential equation.')

    def visit_With(self, node):
        raise errors.DiffEqError(f'Currently, {self.__class__.__name__} do not support to '
                                 f'analyze "with" block in differential equation.')

    def visit_Raise(self, node):
        raise errors.DiffEqError(f'Currently, {self.__class__.__name__} do not support to '
                                 f'analyze "raise" statement in differential equation.')

    def visit_Delete(self, node):
        raise errors.DiffEqError(f'Currently, {self.__class__.__name__} do not support to '
                                 f'analyze "del" operation in differential equation.')


def separate_variables(func_or_code):
    """Separate the expressions in a differential equation for each variable.

    For example, take the HH neuron model as an example:

    >>> eq_code = '''
    >>> def integral(m, h, t, Iext, V):
    >>>    alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    >>>    beta = 4.0 * np.exp(-(V + 65) / 18)
    >>>    dmdt = alpha * (1 - m) - beta * m
    >>>
    >>>    alpha = 0.07 * np.exp(-(V + 65) / 20.)
    >>>    beta = 1 / (1 + np.exp(-(V + 35) / 10))
    >>>    dhdt = alpha * (1 - h) - beta * h
    >>>    return dmdt, dhdt
    >>> '''
    >>> analyser = DiffEqReader()
    >>> analyser.visit(ast.parse(eq_code))
    >>> separate_variables(returns=analyser.returns,
    >>>                    variables=analyser.variables,
    >>>                    right_exprs=analyser.rights,
    >>>                    code_lines=analyser.code_lines)
    {'dhdt': ['alpha = 0.07 * np.exp(-(V + 65) / 20.0)\n',
              'beta = 1 / (1 + np.exp(-(V + 35) / 10))\n',
              'dhdt = alpha * (1 - h) - beta * h\n'],
     'dmdt': ['alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))\n',
              'beta = 4.0 * np.exp(-(V + 65) / 18)\n',
              'dmdt = alpha * (1 - m) - beta * m\n']}

    Parameters
    ----------
    func_or_code : callable, str
        The callable function or the function code.

    Returns
    -------
    anlysis : dict
        The expressions for each return variable.
    """
    if callable(func_or_code):
        func_or_code = tools.deindent(inspect.getsource(func_or_code))
    assert isinstance(func_or_code, str)
    analyser = DiffEqReader()
    analyser.visit(ast.parse(func_or_code))

    returns = analyser.returns
    variables = analyser.variables
    right_exprs = analyser.rights
    code_lines = analyser.code_lines

    return_requires = OrderedDict([(r, set(tools.get_identifiers(r))) for r in returns])
    code_lines_for_returns = OrderedDict([(r, []) for r in returns])
    variables_for_returns = OrderedDict([(r, []) for r in returns])
    expressions_for_returns = OrderedDict([(r, []) for r in returns])

    length = len(variables)
    reverse_ids = list(reversed([i - length for i in range(length)]))
    for r in code_lines_for_returns.keys():
        for rid in reverse_ids:
            dep = []
            for v in variables[rid]:
                if v in return_requires[r]:
                    dep.append(v)
            if len(dep):
                code_lines_for_returns[r].append(code_lines[rid])
                variables_for_returns[r].append(variables[rid])
                expr = right_exprs[rid]
                expressions_for_returns[r].append(expr)
                for d in dep:
                    return_requires[r].remove(d)
                return_requires[r].update(tools.get_identifiers(expr))
    for r in list(code_lines_for_returns.keys()):
        code_lines_for_returns[r] = code_lines_for_returns[r][::-1]
        variables_for_returns[r] = variables_for_returns[r][::-1]
        expressions_for_returns[r] = expressions_for_returns[r][::-1]

    analysis = tools.DictPlus(
        code_lines_for_returns=code_lines_for_returns,
        variables_for_returns=variables_for_returns,
        expressions_for_returns=expressions_for_returns,
    )
    return analysis


# def dissect_diff_eq(func_or_code):
#     if callable(func_or_code):
#         func_or_code = tools.deindent(inspect.getsource(func_or_code))
#     assert isinstance(func_or_code, str)
#     analyser = DiffEqReader()
#     analyser.visit(ast.parse(func_or_code))
#     return separate_variables(returns=analyser.returns,
#                               variables=analyser.variables,
#                               right_exprs=analyser.rights,
#                               code_lines=analyser.code_lines)
