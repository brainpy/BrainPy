# -*- coding: utf-8 -*-

import ast

from brainpy import backend
from brainpy import tools
from brainpy.simulation.brainobjects import SynConn, NeuGroup
from .numba_cpu import NumbaCPUNodeRunner
from .numba_cpu import StepFuncReader


from brainpy import errors

try:
    import numba
except ModuleNotFoundError:
    raise errors.PackageMissingError(errors.backend_missing_msg.format(bk='numba'))

__all__ = [
    'NumbaCudaNodeRunner',
]


class CudaStepFuncReader(StepFuncReader):
    def __init__(self, host):
        super(CudaStepFuncReader, self).__init__(host=host)

        self.need_add_cuda = False
        # get pre assignment
        self.pre_assign = []
        # get post assignment
        self.post_assign = []

    def check_atomic_ops(self, target):
        if isinstance(self.host, SynConn) and isinstance(target, ast.Subscript):
            values = self.visit_attr(target.value)
            slice_ = tools.ast2code(ast.fix_missing_locations(target.slice))
            if len(values) >= 3 and values[-1] in backend.CLASS_KEYWORDS:
                obj = getattr(self.host, values[-2])
                if isinstance(obj, NeuGroup):
                    target_ = '.'.join(values[::-1])
                    return target_, slice_
        return None

    def visit_Assign(self, node, level=0):
        self.generic_visit(node)
        prefix = '  ' * level
        expr = tools.ast2code(ast.fix_missing_locations(node.value))
        self.rights.append(expr)

        check = None
        if len(node.targets) == 1:
            check = self.check_atomic_ops(node.targets[0])

        if check is None:
            targets = []
            for target in node.targets:
                targets.append(tools.ast2code(ast.fix_missing_locations(target)))
            _target = ' = '.join(targets)
            self.lefts.append(_target)
            self.lines.append(f'{prefix}{_target} = {expr}')
        else:
            target, slice_ = check
            self.lefts.append(target)
            self.lines.append(f'{prefix}cuda.atomic.add({target}, {slice_}, {expr})')

    def visit_AugAssign(self, node, level=0):
        self.generic_visit(node)
        prefix = '  ' * level
        op = tools.ast2code(ast.fix_missing_locations(node.op))
        expr = tools.ast2code(ast.fix_missing_locations(node.value))

        check = self.check_atomic_ops(node.target)
        if check is None:
            target = tools.ast2code(ast.fix_missing_locations(node.target))
            self.lefts.append(target)
            self.rights.append(expr)
            self.lines.append(f"{prefix}{target} {op}= {expr}")
        else:
            if op == '+':
                expr = expr
            elif op == '-':
                expr = '-' + expr
            else:
                raise ValueError
            target, slice_ = check
            self.lefts.append(target)
            self.lines.append(f'{prefix}cuda.atomic.add({target}, {slice_}, {expr})')


def analyze_step_func(f, host):
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

    # code lines
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

    analyzed_results = {
        'code_string': code_string,
        'code_scope': code_scope,
        'self_data_in_right': self_data_in_right,
        'self_data_without_index_in_left': self_data_without_index_in_left,
        'self_data_with_index_in_left': self_data_with_index_in_left,
    }

    return analyzed_results


class NumbaCudaNodeRunner(NumbaCPUNodeRunner):
    pass
