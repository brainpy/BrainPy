# -*- coding: utf-8 -*-

import ast

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

__all__ = [
    'FUNCTION_MAPPING',
    'CONSTANT_MAPPING',
    'SympyRender',
    'SympyPrinter',
    'str2sympy',
    'sympy2str',
    'get_mapping_scope',
]

FUNCTION_MAPPING = {
    'real': sympy.functions.elementary.complexes.re,
    'imag': sympy.functions.elementary.complexes.im,
    'conjugate': sympy.functions.elementary.complexes.conjugate,
    'sign': sympy.sign,
    'abs': sympy.functions.elementary.complexes.Abs,

    'cos': sympy.functions.elementary.trigonometric.cos,
    'sin': sympy.functions.elementary.trigonometric.sin,
    'tan': sympy.functions.elementary.trigonometric.tan,
    'sinc': sympy.functions.elementary.trigonometric.sinc,
    'arcsin': sympy.functions.elementary.trigonometric.asin,
    'arccos': sympy.functions.elementary.trigonometric.acos,
    'arctan': sympy.functions.elementary.trigonometric.atan,
    'arctan2': sympy.functions.elementary.trigonometric.atan2,

    'cosh': sympy.functions.elementary.hyperbolic.cosh,
    'sinh': sympy.functions.elementary.hyperbolic.sinh,
    'tanh': sympy.functions.elementary.hyperbolic.tanh,
    'arcsinh': sympy.functions.elementary.hyperbolic.asinh,
    'arccosh': sympy.functions.elementary.hyperbolic.acosh,
    'arctanh': sympy.functions.elementary.hyperbolic.atanh,

    'ceil': sympy.functions.elementary.integers.ceiling,
    'floor': sympy.functions.elementary.integers.floor,

    'log': sympy.functions.elementary.exponential.log,
    'log2': cfunctions.log2,
    'log1p': cfunctions.log1p,
    'log10': cfunctions.log10,
    'exp': sympy.functions.elementary.exponential.exp,
    'expm1': cfunctions.expm1,
    'exp2': cfunctions.exp2,
    'hypot': cfunctions.hypot,

    'sqrt': sympy.functions.elementary.miscellaneous.sqrt,
}

CONSTANT_MAPPING = {
    'pi': sympy.pi,
    'e': sympy.E,
    'inf': sympy.S.Infinity,
    '-inf': sympy.S.NegativeInfinity
}


def get_mapping_scope():
    return {
        'real': np.real,
        'imag': np.imag,
        'conjugate': np.conjugate,
        'sign': np.sign,
        'abs': np.abs,

        'cos': np.cos,
        'sin': np.sin,
        'tan': np.tan,
        'sinc': np.sinc,
        'arcsin': np.arcsin,
        'arccos': np.arccos,
        'arctan': np.arctan,
        'arctan2': np.arctan2,

        'cosh': np.cosh,
        'sinh': np.cosh,
        'tanh': np.tanh,
        'arcsinh': np.arcsinh,
        'arccosh': np.arccosh,
        'arctanh': np.arctanh,

        'ceil': np.ceil,
        'floor': np.floor,

        'log': np.log,
        'log2': np.log2,
        'log1p': np.log1p,
        'log10': np.log10,
        'exp': np.exp,
        'expm1': np.expm1,
        'exp2': np.exp2,
        'hypot': np.hypot,

        'sqrt': np.sqrt,

        'pi': np.pi,
        'e': np.e,
        'inf': np.inf,
        '-inf': -np.inf
    }


class SympyRender(object):
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

    def __init__(self):
        pass

    def render_expr(self, expr, strip=True):
        if strip:
            expr = expr.strip()
        node = ast.parse(expr, mode='eval')
        return self.render_node(node.body)

    def render_code(self, code):
        lines = []
        for node in ast.parse(code).body:
            lines.append(self.render_node(node))
        return '\n'.join(lines)

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
        ops = {'BitXor': ('^', '**'), 'BitAnd': ('&', 'and'), 'BitOr': ('|', 'or')}
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
                return sympy.Function(node.id)
        else:
            if node.attr in FUNCTION_MAPPING:
                return FUNCTION_MAPPING[node.attr]
            if node.attr == 'int':
                return sympy.Function("int_")
            else:
                names = self._get_attr_value(node, [])
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


class SympyPrinter(StrPrinter):
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


_RENDER = SympyRender()
_PRINTER = SympyPrinter()


def str2sympy(str_expr):
    return _RENDER.render_expr(str_expr)


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
