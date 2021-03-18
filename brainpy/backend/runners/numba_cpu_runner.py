# -*- coding: utf-8 -*-

import ast
import inspect
import re

import numba

from brainpy import backend
from brainpy import errors
from brainpy import profile
from brainpy import tools
from . import utils
from .general_runner import GeneralNodeRunner

__all__ = [
    'set_numba_profile',
    'get_numba_profile',

    'StepFuncReader',
    'analyze_step_func',
    'get_func_body_code',
    'get_num_indent',

    'NumbaCPUNodeRunner',
]

NUMBA_PROFILE = {
    'nopython': True,
    'fastmath': True,
    'nogil': True,
    'parallel': False
}


def set_numba_profile(**kwargs):
    """Set the compilation options of Numba JIT function.

    Parameters
    ----------
    kwargs : Any
        The arguments, including ``cache``, ``fastmath``,
        ``parallel``, ``nopython``.
    """
    global NUMBA_PROFILE

    if 'fastmath' in kwargs:
        NUMBA_PROFILE['fastmath'] = kwargs.pop('fastmath')
    if 'nopython' in kwargs:
        NUMBA_PROFILE['nopython'] = kwargs.pop('nopython')
    if 'nogil' in kwargs:
        NUMBA_PROFILE['nogil'] = kwargs.pop('nogil')
    if 'parallel' in kwargs:
        NUMBA_PROFILE['parallel'] = kwargs.pop('parallel')


def get_numba_profile():
    """Get the compilation setting of numba JIT function.

    Returns
    -------
    numba_setting : dict
        Numba setting.
    """
    return NUMBA_PROFILE


class StepFuncReader(ast.NodeVisitor):
    def __init__(self):
        self.lefts = []
        self.rights = []
        self.lines = []

    def visit_Assign(self, node, level=0):
        targets = []
        for target in node.targets:
            targets.append(tools.ast2code(ast.fix_missing_locations(target)))
        target = ' = '.join(targets)
        self.lefts.append(target)
        expr = tools.ast2code(ast.fix_missing_locations(node.value))
        self.rights.append(expr)
        prefix = '  ' * level
        self.lines.append(f'{prefix}{target} = {expr}')
        return node

    def visit_AugAssign(self, node, level=0):
        target = tools.ast2code(ast.fix_missing_locations(node.target))
        op = tools.ast2code(ast.fix_missing_locations(node.op))
        expr = tools.ast2code(ast.fix_missing_locations(node.value))
        prefix = '  ' * level
        self.lefts.append(target)
        self.rights.append(f"{target} {op} {expr}")
        self.lines.append(f"{prefix}{target} {op}= {expr}")
        return node

    def visit_AnnAssign(self, node):
        raise NotImplementedError('Do not support an assignment with '
                                  'a type annotation in Numba backend.')

    def visit_node_not_assign(self, node, level=0):
        prefix = '  ' * level
        expr = tools.ast2code(ast.fix_missing_locations(node))
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
            code = tools.ast2code(ast.fix_missing_locations(node))
            raise errors.CodeError(f'BrainPy does not support {type(node)} '
                                   f'in Numba backend.\n\n{code}')

    def visit_If(self, node, level=0):
        # If condition
        prefix = '  ' * level
        compare = tools.ast2code(ast.fix_missing_locations(node.test))
        self.lines.append(f'{prefix}if {compare}:')
        # body
        for expr in node.body:
            self.visit_content_in_condition_control(expr, level + 1)

        # elif
        while node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node = node.orelse[0]
            compare = tools.ast2code(ast.fix_missing_locations(node.test))
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
        target = tools.ast2code(ast.fix_missing_locations(node.target))
        # iter
        iter = tools.ast2code(ast.fix_missing_locations(node.iter))
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
        test = tools.ast2code(ast.fix_missing_locations(node.test))
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
        raise errors.CodeError('Do not support "try" handler in Numba backend.')

    def visit_With(self, node):
        raise errors.CodeError('Do not support "with" block in Numba backend.')

    def visit_Raise(self, node):
        raise errors.CodeError('Do not support "raise" statement in Numba backend.')

    def visit_Delete(self, node):
        raise errors.CodeError('Do not support "del" operation in Numba backend.')


def analyze_step_func(f):
    """Analyze the step functions in a population.

    Parameters
    ----------
    f : callable
        The step function.

    Returns
    -------
    results : tuple
        The code string of the function, the code scope,
        the data need pass into the arguments,
        the data need return.
    """

    code_string = tools.deindent(inspect.getsource(f)).strip()
    tree = ast.parse(code_string)

    # arguments
    # ---
    args = tools.ast2code(ast.fix_missing_locations(tree.body[0].args)).split(',')

    # code lines
    # ---
    formatter = StepFuncReader()
    formatter.visit(tree)

    # data assigned by self.xx in line right
    # ---
    self_data_in_right = []
    if args[0] in profile.CLASS_KEYWORDS:
        code = ', \n'.join(formatter.rights)
        self_data_in_right = re.findall('\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b', code)
        self_data_in_right = list(set(self_data_in_right))

    # data assigned by self.xxx in line left
    # ---
    code = ', \n'.join(formatter.lefts)
    self_data_without_index_in_left = []
    self_data_with_index_in_left = []
    if args[0] in profile.CLASS_KEYWORDS:
        class_p1 = '\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b'
        self_data_without_index_in_left = set(re.findall(class_p1, code))
        class_p2 = '(\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*)\\[.*\\]'
        self_data_with_index_in_left = set(re.findall(class_p2, code))
        self_data_without_index_in_left -= self_data_with_index_in_left
        self_data_without_index_in_left = list(self_data_without_index_in_left)

    # code scope
    # ---
    closure_vars = inspect.getclosurevars(f)
    code_scope = dict(closure_vars.nonlocals)
    code_scope.update(closure_vars.globals)

    # final
    # ---
    self_data_in_right = sorted(self_data_in_right)
    self_data_without_index_in_left = sorted(self_data_without_index_in_left)
    self_data_with_index_in_left = sorted(self_data_with_index_in_left)
    return code_string, code_scope, self_data_in_right, \
           self_data_without_index_in_left, self_data_with_index_in_left


def get_func_body_code(code_string, lambda_func=False):
    """Get the main body code of a function.

    Parameters
    ----------
    code_string : str
        The code string of the function.
    lambda_func : bool
        Whether the code comes from a lambda function.

    Returns
    -------
    code_body : str
        The code body.
    """
    if lambda_func:
        splits = code_string.split(':')
        if len(splits) != 2:
            raise ValueError(f'Can not parse function: \n{code_string}')
        main_code = f'return {splits[1]}'
    else:
        func_codes = code_string.split('\n')
        idx = 0
        for i, line in enumerate(func_codes):
            idx += 1
            line = line.replace(' ', '')
            if '):' in line:
                break
        else:
            raise ValueError(f'Can not parse function: \n{code_string}')
        main_code = '\n'.join(func_codes[idx:])
    return main_code


def get_num_indent(code_string, spaces_per_tab=4):
    """Get the indent of a patch of source code.

    Parameters
    ----------
    code_string : str
        The code string.
    spaces_per_tab : int
        The spaces per tab.

    Returns
    -------
    num_indent : int
        The number of the indent.
    """
    lines = code_string.split('\n')
    min_indent = 1000
    for line in lines:
        line = line.replace('\t', ' ' * spaces_per_tab)
        num_indent = len(line) - len(line.lstrip())
        if num_indent < min_indent:
            min_indent = num_indent
    return min_indent


class NumbaCPUNodeRunner(GeneralNodeRunner):
    def get_steps_func(self, show_code=False):
        for step in self.steps:
            func_name = step.__name__
            class_arg, arguments = utils.get_args(step)
            host_name = self.host.name

            # arguments 1
            calls = []
            for arg in arguments:
                if hasattr(self.host, arg):
                    calls.append(f'{host_name}.{arg}')
                elif arg in backend.SYSTEM_KEYWORDS:
                    calls.append(arg)
                else:
                    raise errors.ModelDefError(f'Step function "{func_name}" of {self.host} '
                                               f'define an unknown argument "{arg}" which is not '
                                               f'an attribute of {self.host} nor the system keywords '
                                               f'{backend.SYSTEM_KEYWORDS}.')

            # analysis
            code_string, code_scope, self_data_in_right, \
            self_data_without_index_in_left, self_data_with_index_in_left = analyze_step_func(step)
            main_code = get_func_body_code(code_string)
            num_indent = get_num_indent(main_code)

            # arguments 1: data need pass
            data_need_pass = sorted(list(set(self_data_in_right + self_data_with_index_in_left)))
            replaces = {}
            new_args = arguments + []
            for data in data_need_pass:
                splits = data.split('.')
                if len(splits) == 2:
                    attr_name = splits[1]
                    attr_ = getattr(self.host, attr_name)
                    if callable(attr_):
                        replaces[data] = data.replace('.', '_')
                        code_scope[data.replace('.', '_')] = attr_
                        continue
                new_args.append(data.replace('.', '_'))
                calls.append('.'.join([host_name] + splits[1:]))
                replaces[data] = data.replace('.', '_')

            # data need return
            assigns = []
            returns = []
            for data in self_data_without_index_in_left:
                splits = data.split('.')
                assigns.append('.'.join([host_name] + splits[1:]))
                returns.append(data.replace('.', '_'))
                replaces[data] = data.replace('.', '_')

            # code scope
            code_scope[host_name] = self.host

            # codes
            main_code = f'def new_{func_name}({", ".join(new_args)}):\n' + main_code
            if len(returns):
                main_code += f'\n{" " * num_indent}return {", ".join(returns)}'
            main_code = tools.word_replace(main_code, replaces)
            if show_code:
                print(main_code)
                print(code_scope)
                print()

            # recompile
            exec(compile(main_code, '', 'exec'), code_scope)
            func = code_scope[f'new_{func_name}']
            func = numba.jit(**NUMBA_PROFILE)(func)
            self.set_data(f'new_{func_name}', func)

            # finale
            r_line = ''
            if len(assigns):
                r_line = f'{", ".join(assigns)} = '
            self.formatted_funcs[func_name] = {
                'func': func,
                'scope': {host_name: self.host, f'{host_name}_{func_name}': func},
                # 'call': [f'{r_line}{host_name}.new_{func_name}({", ".join(calls)})']
                'call': [f'{r_line}{host_name}_{func_name}({", ".join(calls)})']
            }
