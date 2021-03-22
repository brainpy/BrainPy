# -*- coding: utf-8 -*-

import ast
import inspect
import re

import numba

from brainpy import backend
from brainpy import errors
from brainpy import tools
from brainpy.simulation import delay
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
    def __init__(self, host):
        self.lefts = []
        self.rights = []
        self.lines = []
        self.visited_nodes = set()

        self.host = host
        # get delay information
        self.delay_call = {}

    def visit_Assign(self, node, level=0):
        if node not in self.visited_nodes:
            prefix = '  ' * level
            expr = tools.ast2code(ast.fix_missing_locations(node.value))
            targets = []
            for target in node.targets:
                targets.append(tools.ast2code(ast.fix_missing_locations(target)))
            _target = ' = '.join(targets)

            self.rights.append(expr)
            self.lefts.append(_target)
            self.lines.append(f'{prefix}{_target} = {expr}')

            self.visited_nodes.add(node)

        self.generic_visit(node)

    def visit_AugAssign(self, node, level=0):
        if node not in self.visited_nodes:
            prefix = '  ' * level
            op = tools.ast2code(ast.fix_missing_locations(node.op))
            expr = tools.ast2code(ast.fix_missing_locations(node.value))
            target = tools.ast2code(ast.fix_missing_locations(node.target))

            self.lefts.append(target)
            self.rights.append(f'{target} {op} {expr}')
            self.lines.append(f"{prefix}{target} = {target} {op} {expr}")

            self.visited_nodes.add(node)

        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        raise NotImplementedError('Do not support an assignment with '
                                  'a type annotation in Numba backend.')

    def visit_node_not_assign(self, node, level=0):
        if node not in self.visited_nodes:
            prefix = '  ' * level
            expr = tools.ast2code(ast.fix_missing_locations(node))
            self.lines.append(f'{prefix}{expr}')
            self.lefts.append('')
            self.rights.append(expr)
            self.visited_nodes.add(node)

        self.generic_visit(node)

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
        elif isinstance(node, ast.Call):
            self.visit_Call(node, level)
        elif isinstance(node, ast.Raise):
            self.visit_Raise(node, level)
        else:
            code = tools.ast2code(ast.fix_missing_locations(node))
            raise errors.CodeError(f'BrainPy does not support {type(node)} '
                                   f'in Numba backend.\n\n{code}')

    def visit_attr(self, node):
        if isinstance(node, ast.Attribute):
            r = self.visit_attr(node.value)
            return [node.attr] + r
        elif isinstance(node, ast.Name):
            return [node.id]
        else:
            raise ValueError

    def visit_Call(self, node, level=0):
        if node in self.delay_call:
            return
        calls = self.visit_attr(node.func)
        calls = calls[::-1]

        # delay push / delay pull
        if calls[-1] in ['push', 'pull']:
            obj = self.host
            for data in calls[1:-1]:
                obj = getattr(obj, data)
            obj_func = getattr(obj, calls[-1])
            if isinstance(obj, delay.ConstantDelay) and callable(obj_func):
                func = ".".join(calls)
                args = []
                for arg in node.args:
                    args.append(tools.ast2code(ast.fix_missing_locations(arg)))
                keywords = []
                for arg in node.keywords:
                    keywords.append(tools.ast2code(ast.fix_missing_locations(arg)))
                delay_var = '.'.join([self.host.name] + calls[1:-1])
                if calls[-1] == 'push':
                    kws_append = [f'delay_data={delay_var}_delay_data',
                                  f'delay_in_idx={delay_var}_delay_in_idx', ]
                    data_need_pass = [f'{self.host.name}.{".".join(calls[1:-1])}.delay_data',
                                      f'{self.host.name}.{".".join(calls[1:-1])}.delay_in_idx']
                else:
                    kws_append = [f'delay_data={delay_var}_delay_data',
                                  f'delay_out_idx={delay_var}_delay_out_idx', ]
                    data_need_pass = [f'{self.host.name}.{".".join(calls[1:-1])}.delay_data',
                                      f'{self.host.name}.{".".join(calls[1:-1])}.delay_out_idx']
                org_call = tools.ast2code(ast.fix_missing_locations(node))
                rep_call = f'{func}({", ".join(args + keywords + kws_append)})'
                self.delay_call[node] = dict(type=calls[-1],
                                             args=args,
                                             keywords=keywords,
                                             kws_append=kws_append,
                                             func=func,
                                             org_call=org_call,
                                             rep_call=rep_call,
                                             data_need_pass=data_need_pass)

        self.generic_visit(node)

    def visit_If(self, node, level=0):
        if node not in self.visited_nodes:
            # If condition
            prefix = '  ' * level
            compare = tools.ast2code(ast.fix_missing_locations(node.test))
            self.rights.append(f'if {compare}:')
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

            self.visited_nodes.add(node)

        self.generic_visit(node)

    def visit_For(self, node, level=0):
        if node not in self.visited_nodes:
            prefix = '  ' * level
            # target
            target = tools.ast2code(ast.fix_missing_locations(node.target))
            # iter
            iter = tools.ast2code(ast.fix_missing_locations(node.iter))
            self.rights.append(f'{target} in {iter}')
            self.lines.append(prefix + f'for {target} in {iter}:')
            # body
            for expr in node.body:
                self.visit_content_in_condition_control(expr, level + 1)
            # else
            if len(node.orelse) > 0:
                self.lines.append(prefix + 'else:')
                for expr in node.orelse:
                    self.visit_content_in_condition_control(expr, level + 1)

            self.visited_nodes.add(node)
        self.generic_visit(node)

    def visit_While(self, node, level=0):
        if node not in self.visited_nodes:
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

            self.visited_nodes.add(node)
        self.generic_visit(node)

    def visit_Raise(self, node, level=0):
        if node not in self.visited_nodes:
            prefix = '  ' * level
            line = tools.ast2code(ast.fix_missing_locations(node))
            self.lines.append(prefix + line)

            self.visited_nodes.add(node)
        self.generic_visit(node)

    def visit_Try(self, node):
        raise errors.CodeError('Do not support "try" handler in Numba backend.')

    def visit_With(self, node):
        raise errors.CodeError('Do not support "with" block in Numba backend.')

    def visit_Delete(self, node):
        raise errors.CodeError('Do not support "del" operation in Numba backend.')


def analyze_step_func(host, f):
    """Analyze the step functions in a population.

    Parameters
    ----------
    f : callable
        The step function.
    host : Population
        The data and the function host.

    Returns
    -------
    results : dict
        The code string of the function, the code scope,
        the data need pass into the arguments,
        the data need return.
    """
    code_string = tools.deindent(inspect.getsource(f)).strip()
    tree = ast.parse(code_string)

    # arguments
    # ---
    args = tools.ast2code(ast.fix_missing_locations(tree.body[0].args)).split(',')

    # code AST analysis
    # ---
    formatter = StepFuncReader(host=host)
    formatter.visit(tree)

    # data assigned by self.xx in line right
    # ---
    self_data_in_right = []
    if args[0] in backend.CLASS_KEYWORDS:
        code = ', \n'.join(formatter.rights)
        self_data_in_right = re.findall('\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b', code)
        self_data_in_right = list(set(self_data_in_right))

    # data assigned by self.xxx in line left
    # ---
    code = ', \n'.join(formatter.lefts)
    self_data_without_index_in_left = []
    self_data_with_index_in_left = []
    if args[0] in backend.CLASS_KEYWORDS:
        class_p1 = '\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b'
        self_data_without_index_in_left = set(re.findall(class_p1, code))
        class_p2 = '(\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*)\\[.*\\]'
        self_data_with_index_in_left = set(re.findall(class_p2, code)) - self_data_without_index_in_left
        self_data_with_index_in_left = list(self_data_with_index_in_left)
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

    analyzed_results = {
        'delay_call': formatter.delay_call,
        'code_string': '\n'.join(formatter.lines),
        'code_scope': code_scope,
        'self_data_in_right': self_data_in_right,
        'self_data_without_index_in_left': self_data_without_index_in_left,
        'self_data_with_index_in_left': self_data_with_index_in_left,
    }

    return analyzed_results


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
        if line.strip() == '':
            continue
        line = line.replace('\t', ' ' * spaces_per_tab)
        num_indent = len(line) - len(line.lstrip())
        if num_indent < min_indent:
            min_indent = num_indent
    return min_indent


def class2func(cls_func, host, func_name=None, show_code=False):
    """Transform the function in a class into the ordinary function which is
    compatible with the Numba JIT compilation.

    Parameters
    ----------
    cls_func : function
        The function of the instantiated class.
    func_name : str
        The function name. If not given, it will get the function by `cls_func.__name__`.
    show_code : bool
        Whether show the code.

    Returns
    -------
    new_func : function
        The transformed function.
    """
    class_arg, arguments = utils.get_args(cls_func)
    func_name = cls_func.__name__ if func_name is None else func_name
    host_name = host.name

    # arguments 1
    calls = []
    for arg in arguments:
        if hasattr(host, arg):
            calls.append(f'{host_name}.{arg}')
        elif arg in backend.SYSTEM_KEYWORDS:
            calls.append(arg)
        else:
            raise errors.ModelDefError(f'Step function "{func_name}" of {host} '
                                       f'define an unknown argument "{arg}" which is not '
                                       f'an attribute of {host} nor the system keywords '
                                       f'{backend.SYSTEM_KEYWORDS}.')

    # analysis
    analyzed_results = analyze_step_func(host=host, f=cls_func)
    delay_call = analyzed_results['delay_call']
    # code_string = analyzed_results['code_string']
    main_code = analyzed_results['code_string']
    code_scope = analyzed_results['code_scope']
    self_data_in_right = analyzed_results['self_data_in_right']
    self_data_without_index_in_left = analyzed_results['self_data_without_index_in_left']
    self_data_with_index_in_left = analyzed_results['self_data_with_index_in_left']
    # main_code = get_func_body_code(code_string)
    num_indent = get_num_indent(main_code)
    data_need_pass = sorted(list(set(self_data_in_right + self_data_with_index_in_left)))
    data_need_return = self_data_without_index_in_left

    # check delay
    replaces_early = {}
    replaces_later = {}
    if len(delay_call) > 0:
        for delay_ in delay_call.values():
            # delay_ = dict(type=calls[-1],
            #               args=args,
            #               keywords=keywords,
            #               kws_append=kws_append,
            #               func=func,
            #               org_call=org_call,
            #               rep_call=rep_call,
            #               data_need_pass=data_need_pass)
            if delay_['type'] == 'push':
                if len(delay_['args'] + delay_['keywords']) == 2:
                    func = numba.njit(delay.push_type2)
                elif len(delay_['args'] + delay_['keywords']) == 1:
                    func = numba.njit(delay.push_type1)
                else:
                    raise ValueError(f'Unknown delay push. {delay_}')
            else:
                if len(delay_['args'] + delay_['keywords']) == 1:
                    func = numba.njit(delay.pull_type1)
                elif len(delay_['args'] + delay_['keywords']) == 0:
                    func = numba.njit(delay.pull_type0)
                else:
                    raise ValueError(f'Unknown delay pull. {delay_}')
            delay_call_name = delay_['func']
            data_need_pass.remove(delay_call_name)
            data_need_pass.extend(delay_['data_need_pass'])
            replaces_early[delay_['org_call']] = delay_['rep_call']
            replaces_later[delay_call_name] = delay_call_name.replace('.', '_')
            code_scope[delay_call_name.replace('.', '_')] = func
    for target, dest in replaces_early.items():
        main_code = main_code.replace(target, dest)
    # main_code = tools.word_replace(main_code, replaces_early)

    # arguments 2: data need pass
    new_args = arguments + []
    for data in sorted(set(data_need_pass)):
        splits = data.split('.')
        replaces_later[data] = data.replace('.', '_')
        obj = host
        for attr in splits[1:]:
            obj = getattr(obj, attr)
        if callable(obj):
            code_scope[data.replace('.', '_')] = obj
            continue
        new_args.append(data.replace('.', '_'))
        calls.append('.'.join([host_name] + splits[1:]))

    # data need return
    assigns = []
    returns = []
    for data in data_need_return:
        splits = data.split('.')
        assigns.append('.'.join([host_name] + splits[1:]))
        returns.append(data.replace('.', '_'))
        replaces_later[data] = data.replace('.', '_')

    # code scope
    code_scope[host_name] = host

    # codes
    header = f'def new_{func_name}({", ".join(new_args)}):\n'
    main_code = header + tools.indent(main_code, spaces_per_tab=2)
    if len(returns):
        main_code += f'\n{" " * num_indent + "  "}return {", ".join(returns)}'
    main_code = tools.word_replace(main_code, replaces_later)
    if show_code:
        print(main_code)
        print(code_scope)
        print()

    # recompile
    exec(compile(main_code, '', 'exec'), code_scope)
    func = code_scope[f'new_{func_name}']
    func = numba.jit(**NUMBA_PROFILE)(func)
    return func, calls, assigns


class NumbaCPUNodeRunner(GeneralNodeRunner):
    def get_steps_func(self, show_code=False):
        for func_name, step in self.steps.items():
            host = step.__self__
            func, calls, assigns = class2func(cls_func=step, host=host, func_name=func_name, show_code=show_code)
            # self.set_data(f'new_{func_name}', func)
            setattr(host, f'new_{func_name}', func)

            # finale
            assignment_line = ''
            if len(assigns):
                assignment_line = f'{", ".join(assigns)} = '
            self.formatted_funcs[func_name] = {
                'func': func,
                'scope': {host.name: host},
                'call': [f'{assignment_line}{host.name}.new_{func_name}({", ".join(calls)})']
            }
