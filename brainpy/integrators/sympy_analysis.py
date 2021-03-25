# -*- coding: utf-8 -*-

import ast
import math
from collections import Counter

import numpy as np

from brainpy import errors
from brainpy import tools

try:
    import sympy
except ModuleNotFoundError:
    raise errors.PackageMissingError('Package "sympy" must be installed when the '
                                     'users want to utilize the sympy analysis.')
import sympy.functions.elementary.complexes
import sympy.functions.elementary.exponential
import sympy.functions.elementary.hyperbolic
import sympy.functions.elementary.integers
import sympy.functions.elementary.miscellaneous
import sympy.functions.elementary.trigonometric
from sympy.codegen import cfunctions
from sympy.printing.precedence import precedence
from sympy.printing.str import StrPrinter

CONSTANT_NOISE = 'CONSTANT'
FUNCTIONAL_NOISE = 'FUNCTIONAL'

FUNCTION_MAPPING = {
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

# Get functions in math
_functions_in_math = []
for key in dir(math):
    if not key.startswith('__'):
        _functions_in_math.append(getattr(math, key))

# Get functions in NumPy
_functions_in_numpy = []
for key in dir(np):
    if not key.startswith('__'):
        _functions_in_numpy.append(getattr(np, key))
for key in dir(np.random):
    if not key.startswith('__'):
        _functions_in_numpy.append(getattr(np.random, key))
for key in dir(np.linalg):
    if not key.startswith('__'):
        _functions_in_numpy.append(getattr(np.linalg, key))


def func_in_numpy_or_math(func):
    return func in _functions_in_math or func in _functions_in_numpy


def get_mapping_scope():
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
    }


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
    Printer that overrides the printing of some basic sympy objects.

    e.g. print "a and b" instead of "And(a, b)".
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


class Expression(object):
    def __init__(self, var, code):
        self.var_name = var
        self.code = code.strip()
        self.substituted_code = None

    @property
    def identifiers(self):
        return tools.get_identifiers(self.code)

    def __str__(self):
        return f'{self.var_name} = {self.code}'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if self.code != other.code:
            return False
        if self.var_name != other.var_name:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_code(self, subs=True):
        if subs:
            if self.substituted_code is None:
                return self.code
            else:
                return self.substituted_code
        else:
            return self.code


class SingleDiffEq(object):
    """Single Differential Equation.

    A differential equation is defined as the standard form:

    dx/dt = f(x) + g(x) dW

    Parameters
    ----------
    var_name : str
        The variable names.
    variables : list
        The code variables.
    expressions : list
        The code expressions for each line.
    derivative_expr : str
        The final derivative expression.
    scope : dict
        The code scope.
    """

    def __init__(self, var_name, variables, expressions, derivative_expr, scope,
                 func_name):
        self.func_name = func_name
        # function scope
        self.func_scope = scope

        # differential variable name and time name
        self.var_name = var_name
        self.t_name = 't'

        # analyse function code
        self.expressions = [Expression(v, expr) for v, expr in zip(variables, expressions)]
        self.f_expr = Expression('_f_res_', derivative_expr)
        for k, num in Counter(variables).items():
            if num > 1:
                raise errors.AnalyzerError(
                    f'Found "{k}" {num} times. Please assign each expression '
                    f'in differential function with a unique name. ')

    def _substitute(self, final_exp, expressions, substitute_vars=None):
        """Substitute expressions to get the final single expression

        Parameters
        ----------
        final_exp : Expression
            The final expression.
        expressions : list, tuple
            The list/tuple of expressions.
        """
        if substitute_vars is None:
            return
        if final_exp is None:
            return
        assert substitute_vars == 'all' or \
               substitute_vars == self.var_name or \
               isinstance(substitute_vars, (tuple, list))

        # Goal: Substitute dependent variables into the expresion
        # Hint: This step doesn't require the left variables are unique
        dependencies = {}
        for expr in expressions:
            substitutions = {}
            for dep_var, dep_expr in dependencies.items():
                if dep_var in expr.identifiers:
                    code = dep_expr.get_code(subs=True)
                    substitutions[sympy.Symbol(dep_var, real=True)] = str2sympy(code).expr
            if len(substitutions):
                new_sympy_expr = str2sympy(expr.code).expr.xreplace(substitutions)
                new_str_expr = sympy2str(new_sympy_expr)
                expr.substituted_code = new_str_expr
                dependencies[expr.var_name] = expr
            else:
                if substitute_vars == 'all':
                    dependencies[expr.var_name] = expr
                elif substitute_vars == self.var_name:
                    if self.var_name in expr.identifiers:
                        dependencies[expr.var_name] = expr
                else:
                    ids = expr.identifiers
                    for var in substitute_vars:
                        if var in ids:
                            dependencies[expr.var_name] = expr
                            break

        # Goal: get the final differential equation
        # Hint: the step requires the expression variables must be unique
        substitutions = {}
        for dep_var, dep_expr in dependencies.items():
            code = dep_expr.get_code(subs=True)
            substitutions[sympy.Symbol(dep_var, real=True)] = str2sympy(code).expr
        if len(substitutions):
            new_sympy_expr = str2sympy(final_exp.code).expr.xreplace(substitutions)
            new_str_expr = sympy2str(new_sympy_expr)
            final_exp.substituted_code = new_str_expr

    def get_f_expressions(self, substitute_vars=None):
        if self.f_expr is None:
            return []
        self._substitute(self.f_expr, self.expressions, substitute_vars=substitute_vars)

        return_expressions = []
        # the derivative expression
        dif_eq_code = self.f_expr.get_code(subs=True)
        return_expressions.append(Expression(f'_df{self.var_name}_dt', dif_eq_code))
        # needed variables
        need_vars = tools.get_identifiers(dif_eq_code)
        # get the total return expressions
        for expr in self.expressions[::-1]:
            if expr.var_name in need_vars:
                if expr.substituted_code is None:
                    code = expr.code
                else:
                    code = expr.substituted_code
                return_expressions.append(Expression(expr.var_name, code))
                need_vars |= tools.get_identifiers(code)
        return return_expressions[::-1]

    def _replace_expressions(self, expressions, name, y_sub, t_sub=None):
        """Replace expressions of df part.

        Parameters
        ----------
        expressions : list, tuple
            The list/tuple of expressions.
        name : str
            The name of the new expression.
        y_sub : str
            The new name of the variable "y".
        t_sub : str, optional
            The new name of the variable "t".

        Returns
        -------
        list_of_expr : list
            A list of expressions.
        """
        return_expressions = []

        # replacements
        replacement = {self.var_name: y_sub}
        if t_sub is not None:
            replacement[self.t_name] = t_sub

        # replace variables in expressions
        for expr in expressions:
            replace = False
            identifiers = expr.identifiers
            for repl_var in replacement.keys():
                if repl_var in identifiers:
                    replace = True
                    break
            if replace:
                code = tools.word_replace(expr.code, replacement)
                new_expr = Expression(f"{expr.var_name}_{name}", code)
                return_expressions.append(new_expr)
                replacement[expr.var_name] = new_expr.var_name
        return return_expressions

    def replace_f_expressions(self, name, y_sub, t_sub=None):
        """Replace expressions of df part.

        Parameters
        ----------
        name : str
            The name of the new expression.
        y_sub : str
            The new name of the variable "y".
        t_sub : str, optional
            The new name of the variable "t".

        Returns
        -------
        list_of_expr : list
            A list of expressions.
        """
        return self._replace_expressions(self.get_f_expressions(),
                                         name=name,
                                         y_sub=y_sub,
                                         t_sub=t_sub)

    @property
    def expr_names(self):
        return [expr.var_name for expr in self.expressions]
