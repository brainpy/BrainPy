# -*- coding: utf-8 -*-

import ast
import math

import numpy as np
import sympy
import sympy.functions.elementary.complexes
import sympy.functions.elementary.exponential
import sympy.functions.elementary.hyperbolic
import sympy.functions.elementary.integers
import sympy.functions.elementary.miscellaneous
import sympy.functions.elementary.trigonometric
from sympy.codegen import cfunctions
from sympy.printing.precedence import precedence
from sympy.printing.str import StrPrinter

from .. import errors
from .. import profile
from .. import tools

__all__ = [
    'FUNCTION_MAPPING',
    'CONSTANT_MAPPING',
    'Parser',
    'Printer',
    'str2sympy',
    'sympy2str',
    'get_mapping_scope',
    'DiffEquationAnalyser',
    'analyse_diff_eq',
]

FUNCTION_MAPPING = {
    # 'real': sympy.functions.elementary.complexes.re,
    # 'imag': sympy.functions.elementary.complexes.im,
    # 'conjugate': sympy.functions.elementary.complexes.conjugate,

    # functions in inherit python
    # ---------------------------
    'abs': sympy.functions.elementary.complexes.Abs,

    # functions in numpy
    # ------------------
    'sign': sympy.sign,
    'sinc': sympy.functions.elementary.trigonometric.sinc,
    'arcsin': sympy.functions.elementary.trigonometric.asin,
    'arccos': sympy.functions.elementary.trigonometric.acos,
    'arctan': sympy.functions.elementary.trigonometric.atan,
    'arctan2': sympy.functions.elementary.trigonometric.atan2,
    'arcsinh': sympy.functions.elementary.hyperbolic.asinh,
    'arccosh': sympy.functions.elementary.hyperbolic.acosh,
    'arctanh': sympy.functions.elementary.hyperbolic.atanh,

    'log2': cfunctions.log2,
    'log1p': cfunctions.log1p,

    'expm1': cfunctions.expm1,
    'exp2': cfunctions.exp2,

    # 'maximum': sympy.functions.elementary.miscellaneous.Max,
    # 'minimum': sympy.functions.elementary.miscellaneous.Min,

    # functions in math
    # ------------------
    'asin': sympy.functions.elementary.trigonometric.asin,
    'acos': sympy.functions.elementary.trigonometric.acos,
    'atan': sympy.functions.elementary.trigonometric.atan,
    'atan2': sympy.functions.elementary.trigonometric.atan2,
    'asinh': sympy.functions.elementary.hyperbolic.asinh,
    'acosh': sympy.functions.elementary.hyperbolic.acosh,
    'atanh': sympy.functions.elementary.hyperbolic.atanh,

    # functions in both numpy and math
    # --------------------------------

    'cos': sympy.functions.elementary.trigonometric.cos,
    'sin': sympy.functions.elementary.trigonometric.sin,
    'tan': sympy.functions.elementary.trigonometric.tan,
    'cosh': sympy.functions.elementary.hyperbolic.cosh,
    'sinh': sympy.functions.elementary.hyperbolic.sinh,
    'tanh': sympy.functions.elementary.hyperbolic.tanh,

    'log': sympy.functions.elementary.exponential.log,
    'log10': cfunctions.log10,
    'sqrt': sympy.functions.elementary.miscellaneous.sqrt,
    'exp': sympy.functions.elementary.exponential.exp,
    'hypot': cfunctions.hypot,

    'ceil': sympy.functions.elementary.integers.ceiling,
    'floor': sympy.functions.elementary.integers.floor,
}

CONSTANT_MAPPING = {
    # constants in both numpy and math
    # --------------------------------
    'pi': sympy.pi,
    'e': sympy.E,
    'inf': sympy.S.Infinity,
}


def get_mapping_scope():
    if profile.run_on_cpu():
        return {
            'sign': np.sign, 'cos': np.cos, 'sin': np.sin, 'tan': np.tan,
            'sinc': np.sinc, 'arcsin': np.arcsin, 'arccos': np.arccos,
            'arctan': np.arctan, 'arctan2': np.arctan2, 'cosh': np.cosh,
            'sinh': np.cosh, 'tanh': np.tanh, 'arcsinh': np.arcsinh,
            'arccosh': np.arccosh, 'arctanh': np.arctanh, 'ceil': np.ceil,
            'floor': np.floor, 'log': np.log, 'log2': np.log2, 'log1p': np.log1p,
            'log10': np.log10, 'exp': np.exp, 'expm1': np.expm1, 'exp2': np.exp2,
            'hypot': np.hypot, 'sqrt': np.sqrt, 'pi': np.pi, 'e': np.e, 'inf': np.inf,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan, 'atan2': math.atan2,
            'asinh': math.asinh, 'acosh': math.acosh, 'atanh': math.atanh,
            # 'Max': np.maximum, 'Min': np.minimum
        }
    else:
        return {
            # functions in numpy
            # ------------------
            'arcsin': math.asin, 'arccos': math.acos,
            'arctan': math.atan, 'arctan2': math.atan2, 'arcsinh': math.asinh,
            'arccosh': math.acosh, 'arctanh': math.atanh,
            'sign': np.sign, 'sinc': np.sinc,
            'log2': np.log2, 'log1p': np.log1p,
            'expm1': np.expm1, 'exp2': np.exp2,
            # 'Max': max, 'Min': min,

            # functions in math
            # ------------------
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'atan2': math.atan2,
            'asinh': math.asinh,
            'acosh': math.acosh,
            'atanh': math.atanh,

            # functions in both numpy and math
            # --------------------------------
            'cos': math.cos,
            'sin': math.sin,
            'tan': math.tan,
            'cosh': math.cosh,
            'sinh': math.sinh,
            'tanh': math.tanh,
            'log': math.log,
            'log10': math.log10,
            'sqrt': math.sqrt,
            'exp': math.exp,
            'hypot': math.hypot,
            'ceil': math.ceil,
            'floor': math.floor,

            # constants in both numpy and math
            # --------------------------------
            'pi': math.pi,
            'e': math.e,
            'inf': math.inf}


class Parser(object):
    expression_ops = {
        'Add': sympy.Add,
        'Mult': sympy.Mul,
        'Pow': sympy.Pow,
        'Mod': sympy.Mod,
        # Compare
        'Lt': sympy.StrictLessThan,
        'LtE': sympy.LessThan,
        'Gt': sympy.StrictGreaterThan,
        'GtE': sympy.GreaterThan,
        'Eq': sympy.Eq,
        'NotEq': sympy.Ne,
        # Bool ops
        'And': sympy.And,
        'Or': sympy.Or,
        # BinOp
        'Sub': '-',
        'Div': '/',
        'FloorDiv': '//',
        # Compare
        'Not': 'not',
        'UAdd': '+',
        'USub': '-',
        # Augmented assign
        'AugAdd': '+=',
        'AugSub': '-=',
        'AugMult': '*=',
        'AugDiv': '/=',
        'AugPow': '**=',
        'AugMod': '%=',
    }

    def __init__(self, expr):
        self.contain_unknown_func = False
        node = ast.parse(expr.strip(), mode='eval')
        self.expr = self.render_node(node.body)

    def render_node(self, node):
        nodename = node.__class__.__name__
        methname = 'render_' + nodename
        try:
            return getattr(self, methname)(node)
        except AttributeError:
            raise SyntaxError(f"Unknown syntax: {nodename}, {node}")

    def render_Attribute(self, node):
        if node.attr in CONSTANT_MAPPING:
            return CONSTANT_MAPPING[node.attr]
        else:
            names = self._get_attr_value(node, [])
            return sympy.Symbol('.'.join(names), real=True)

    def render_Constant(self, node):
        if node.value is True or node.value is False or node.value is None:
            return self.render_NameConstant(node)
        else:
            return self.render_Num(node)

    def render_element_parentheses(self, node):
        """Render an element with parentheses around it or leave them away for
        numbers, names and function calls.
        """
        if node.__class__.__name__ in ['Name', 'NameConstant']:
            return self.render_node(node)
        elif node.__class__.__name__ in ['Num', 'Constant'] and \
                getattr(node, 'n', getattr(node, 'value', None)) >= 0:
            return self.render_node(node)
        elif node.__class__.__name__ == 'Call':
            return self.render_node(node)
        else:
            return f'({self.render_node(node)})'

    def render_BinOp_parentheses(self, left, right, op):
        """Use a simplified checking whether it is possible to omit parentheses:
        only omit parentheses for numbers, variable names or function calls.
        This means we still put needless parentheses because we ignore
        precedence rules, e.g. we write "3 + (4 * 5)" but at least we do
        not do "(3) + ((4) + (5))" """
        ops = {'BitXor': ('^', '**'),
               'BitAnd': ('&', 'and'),
               'BitOr': ('|', 'or')}
        op_class = op.__class__.__name__
        # Give a more useful error message when using bit-wise operators
        if op_class in ['BitXor', 'BitAnd', 'BitOr']:
            correction = ops.get(op_class)
            raise SyntaxError('The operator "{}" is not supported, use "{}" '
                              'instead.'.format(correction[0], correction[1]))
        return f'{self.render_element_parentheses(left)} ' \
               f'{self.expression_ops[op_class]} ' \
               f'{self.render_element_parentheses(right)}'

    def render_Assign(self, node):
        if len(node.targets) > 1:
            raise SyntaxError("Only support syntax like a=b not a=b=c")
        return f'{self.render_node(node.targets[0])} = {self.render_node(node.value)}'

    def render_AugAssign(self, node):
        target = node.target.id
        rhs = self.render_node(node.value)
        op = self.expression_ops['Aug' + node.op.__class__.__name__]
        return f'{target} {op} {rhs}'

    def _get_attr_value(self, node, names):
        if hasattr(node, 'value'):
            names.insert(0, node.attr)
            return self._get_attr_value(node.value, names)
        else:
            assert hasattr(node, 'id')
            names.insert(0, node.id)
            return names

    def render_func(self, node):
        if hasattr(node, 'id'):
            if node.id in FUNCTION_MAPPING:
                f = FUNCTION_MAPPING[node.id]
                return f
            # special workaround for the "int" function
            if node.id == 'int':
                return sympy.Function("int_")
            else:
                self.contain_unknown_func = True
                return sympy.Function(node.id)
        else:
            if node.attr in FUNCTION_MAPPING:
                return FUNCTION_MAPPING[node.attr]
            if node.attr == 'int':
                return sympy.Function("int_")
            else:
                names = self._get_attr_value(node, [])
                self.contain_unknown_func = True
                return sympy.Function('.'.join(names))

    def render_Call(self, node):
        if len(node.keywords):
            raise ValueError("Keyword arguments not supported.")
        elif getattr(node, 'starargs', None) is not None:
            raise ValueError("Variable number of arguments not supported")
        elif getattr(node, 'kwargs', None) is not None:
            raise ValueError("Keyword arguments not supported")
        elif len(node.args) == 0:
            return self.render_func(node.func)(sympy.Symbol('_placeholder_arg'))
        else:
            return self.render_func(node.func)(*(self.render_node(arg) for arg in node.args))

    def render_Compare(self, node):
        if len(node.comparators) > 1:
            raise SyntaxError("Can only handle single comparisons like a<b not a<b<c")
        op = node.ops[0]
        ops = self.expression_ops[op.__class__.__name__]
        left = self.render_node(node.left)
        right = self.render_node(node.comparators[0])
        return ops(left, right)

    def render_Name(self, node):
        if node.id in CONSTANT_MAPPING:
            return CONSTANT_MAPPING[node.id]
        else:
            return sympy.Symbol(node.id, real=True)

    def render_NameConstant(self, node):
        if node.value in [True, False]:
            return node.value
        else:
            return str(node.value)

    def render_Num(self, node):
        return sympy.Float(node.n)

    def render_BinOp(self, node):
        op_name = node.op.__class__.__name__
        # Sympy implements division and subtraction as multiplication/addition
        if op_name == 'Div':
            op = self.expression_ops['Mult']
            left = self.render_node(node.left)
            right = 1 / self.render_node(node.right)
            return op(left, right)
        elif op_name == 'FloorDiv':
            op = self.expression_ops['Mult']
            left = self.render_node(node.left)
            right = self.render_node(node.right)
            return sympy.floor(op(left, 1 / right))
        elif op_name == 'Sub':
            op = self.expression_ops['Add']
            left = self.render_node(node.left)
            right = -self.render_node(node.right)
            return op(left, right)
        else:
            op = self.expression_ops[op_name]
            left = self.render_node(node.left)
            right = self.render_node(node.right)
            return op(left, right)

    def render_BoolOp(self, node):
        op = self.expression_ops[node.op.__class__.__name__]
        return op(*(self.render_node(value) for value in node.values))

    def render_UnaryOp(self, node):
        op_name = node.op.__class__.__name__
        if op_name == 'UAdd':
            return self.render_node(node.operand)
        elif op_name == 'USub':
            return -self.render_node(node.operand)
        elif op_name == 'Not':
            return sympy.Not(self.render_node(node.operand))
        else:
            raise ValueError('Unknown unary operator: ' + op_name)


class Printer(StrPrinter):
    """
    Printer that overrides the printing of some basic sympy objects. reversal_potential.g.
    print "a and b" instead of "And(a, b)".
    """

    def _print_And(self, expr):
        return ' and '.join(['(%s)' % self.doprint(arg) for arg in expr.args])

    def _print_Or(self, expr):
        return ' or '.join(['(%s)' % self.doprint(arg) for arg in expr.args])

    def _print_Not(self, expr):
        if len(expr.args) != 1:
            raise AssertionError('"Not" with %d arguments?' % len(expr.args))
        return f'not ({self.doprint(expr.args[0])})'

    def _print_Relational(self, expr):
        return f'{self.parenthesize(expr.lhs, precedence(expr))} ' \
               f'{self._relationals.get(expr.rel_op) or expr.rel_op} ' \
               f'{self.parenthesize(expr.rhs, precedence(expr))}'

    def _print_Function(self, expr):
        # Special workaround for the int function
        if expr.func.__name__ == 'int_':
            return f'int({self.stringify(expr.args, ", ")})'
        elif expr.func.__name__ == 'Mod':
            return f'(({self.doprint(expr.args[0])})%({self.doprint(expr.args[1])}))'
        else:
            return expr.func.__name__ + f"({self.stringify(expr.args, ', ')})"


_PRINTER = Printer()


def str2sympy(str_expr):
    return Parser(str_expr)


def sympy2str(sympy_expr):
    # replace the standard functions by our names if necessary
    replacements = dict((f, sympy.Function(name)) for name, f in FUNCTION_MAPPING.items() if str(f) != name)

    # replace constants with our names as well
    replacements.update(dict((c, sympy.Symbol(name)) for name, c in CONSTANT_MAPPING.items() if str(c) != name))

    # Replace the placeholder argument by an empty symbol
    replacements[sympy.Symbol('_placeholder_arg')] = sympy.Symbol('')
    atoms = (sympy_expr.atoms() | {f.func for f in sympy_expr.atoms(sympy.Function)})
    for old, new in replacements.items():
        if old in atoms:
            sympy_expr = sympy_expr.subs(old, new)

    return _PRINTER.doprint(sympy_expr)


class DiffEquationAnalyser(ast.NodeTransformer):
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
            raise errors.DiffEquationError('BrainPy currently does not support multiple '
                                           'assignment in differential equation.')
        self.variables.append(targets[0].id)
        self.expressions.append(tools.ast2code(ast.fix_missing_locations(node.value)))
        return node

    def visit_AugAssign(self, node):
        var = node.target.id
        self.variables.append(var)
        op = tools.ast2code(ast.fix_missing_locations(node.op))
        expr = tools.ast2code(ast.fix_missing_locations(node.value))
        self.expressions.append(f"{var} {op} {expr}")
        return node

    def visit_AnnAssign(self, node):
        raise errors.DiffEquationError('Do not support an assignment with a type annotation.')

    def visit_Return(self, node):
        value = node.value
        if isinstance(value, (ast.Tuple, ast.List)):  # a tuple/list return
            v0 = value.elts[0]
            if isinstance(v0, (ast.Tuple, ast.List)):  # item 0 is a tuple/list
                # f expression
                if isinstance(v0.elts[0], ast.Name):
                    self.f_expr = ('_f_res_', v0.elts[0].id)
                else:
                    self.f_expr = ('_f_res_', tools.ast2code(ast.fix_missing_locations(v0.elts[0])))

                if len(v0.elts) == 1:
                    self.return_type = '(x,),'
                elif len(v0.elts) == 2:
                    self.return_type = '(x,x),'
                    # g expression
                    if isinstance(v0.elts[1], ast.Name):
                        self.g_expr = ('_g_res_', v0.elts[1].id)
                    else:
                        self.g_expr = ('_g_res_', tools.ast2code(ast.fix_missing_locations(v0.elts[1])))
                else:
                    raise errors.DiffEquationError(f'The dxdt should have the format of (f, g), not '
                                                   f'"({tools.ast2code(ast.fix_missing_locations(v0.elts))})"')

                # returns
                for i, item in enumerate(value.elts[1:]):
                    if isinstance(item, ast.Name):
                        self.returns.append(item.id)
                    else:
                        self.returns.append(tools.ast2code(ast.fix_missing_locations(item)))

            else:  # item 0 is not a tuple/list
                # f expression
                if isinstance(v0, ast.Name):
                    self.f_expr = ('_f_res_', v0.id)
                else:
                    self.f_expr = ('_f_res_', tools.ast2code(ast.fix_missing_locations(v0)))

                if len(value.elts) == 1:
                    self.return_type = 'x,'
                elif len(value.elts) == 2:
                    self.return_type = 'x,x'
                    # g expression
                    if isinstance(value.elts[1], ast.Name):
                        self.g_expr = ('_g_res_', value.elts[1].id)
                    else:
                        self.g_expr = ("_g_res_", tools.ast2code(ast.fix_missing_locations(value.elts[1])))
                else:
                    raise errors.DiffEquationError('Cannot parse return expression. It should have the '
                                                   'format of "(f, [g]), [*return_values]"')
        else:
            self.return_type = 'x'
            if isinstance(value, ast.Name):  # a name return
                self.f_expr = ('_f_res_', value.id)
            else:  # an expression return
                self.f_expr = ('_f_res_', tools.ast2code(ast.fix_missing_locations(value)))
        return node

    def visit_If(self, node):
        raise errors.DiffEquationError('Do not support "if" statement in differential equation.')

    def visit_IfExp(self, node):
        raise errors.DiffEquationError('Do not support "if" expression in differential equation.')

    def visit_For(self, node):
        raise errors.DiffEquationError('Do not support "for" loop in differential equation.')

    def visit_While(self, node):
        raise errors.DiffEquationError('Do not support "while" loop in differential equation.')

    def visit_Try(self, node):
        raise errors.DiffEquationError('Do not support "try" handler in differential equation.')

    def visit_With(self, node):
        raise errors.DiffEquationError('Do not support "with" block in differential equation.')

    def visit_Raise(self, node):
        raise errors.DiffEquationError('Do not support "raise" statement in differential equation.')

    def visit_Delete(self, node):
        raise errors.DiffEquationError('Do not support "del" operation in differential equation.')


def analyse_diff_eq(eq_code):
    assert eq_code.strip() != ''
    tree = ast.parse(eq_code)
    analyser = DiffEquationAnalyser()
    analyser.visit(tree)

    res = tools.DictPlus(variables=analyser.variables,
                         expressions=analyser.expressions,
                         return_intermediates=analyser.returns,
                         return_type=analyser.return_type,
                         f_expr=analyser.f_expr,
                         g_expr=analyser.g_expr)
    return res
