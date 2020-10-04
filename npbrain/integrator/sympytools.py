import ast
import re
from collections import Counter

import sympy
from sympy.printing.precedence import precedence
from sympy.printing.str import StrPrinter

from npbrain.integrator.namespace import DEFAULT_CONSTANTS
from npbrain.integrator.namespace import DEFAULT_FUNCTIONS


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
        # Unary ops are handled manually
        # Bool ops
        'And': sympy.And,
        'Or': sympy.Or,
        # BinOp
        'Sub': '-',
        'Div': '/',
        'FloorDiv': '//',
        # Compare
        # Unary ops
        'Not': 'not',
        'UAdd': '+',
        'USub': '-',
        # Bool ops
        # Augmented assign
        'AugAdd': '+=',
        'AugSub': '-=',
        'AugMult': '*=',
        'AugDiv': '/=',
        'AugPow': '**=',
        'AugMod': '%=',
    }

    def __init__(self, auto_vectorise=None):
        if auto_vectorise is None:
            auto_vectorise = set()
        self.auto_vectorise = auto_vectorise

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

    def render_Constant(self, node):  # For literals in Python 3.8
        if node.value is True or node.value is False or node.value is None:
            return self.render_NameConstant(node)
        else:
            return self.render_Num(node)

    def render_element_parentheses(self, node):
        '''
        Render an element with parentheses around it or leave them away for
        numbers, names and function calls.
        '''
        if node.__class__.__name__ in ['Name', 'NameConstant']:
            return self.render_node(node)
        elif node.__class__.__name__ in ['Num', 'Constant'] and \
                getattr(node, 'n', getattr(node, 'value', None)) >= 0:
            return self.render_node(node)
        elif node.__class__.__name__ == 'Call':
            return self.render_node(node)
        else:
            return '(%s)' % self.render_node(node)

    def render_BinOp_parentheses(self, left, right, op):
        # Use a simplified checking whether it is possible to omit parentheses:
        # only omit parentheses for numbers, variable names or function calls.
        # This means we still put needless parentheses because we ignore
        # precedence rules, e.g. we write "3 + (4 * 5)" but at least we do
        # not do "(3) + ((4) + (5))"
        ops = {'BitXor': ('^', '**'), 'BitAnd': ('&', 'and'), 'BitOr': ('|', 'or')}
        op_class = op.__class__.__name__
        # Give a more useful error message when using bit-wise operators
        if op_class in ['BitXor', 'BitAnd', 'BitOr']:
            correction = ops.get(op_class)
            raise SyntaxError('The operator "{}" is not supported, use "{}" '
                              'instead.'.format(correction[0], correction[1]))
        return '%s %s %s' % (self.render_element_parentheses(left),
                             self.expression_ops[op_class],
                             self.render_element_parentheses(right))

    def render_Assign(self, node):
        if len(node.targets) > 1:
            raise SyntaxError("Only support syntax like a=b not a=b=c")
        return '%s = %s' % (self.render_node(node.targets[0]),
                            self.render_node(node.value))

    def render_AugAssign(self, node):
        target = node.target.id
        rhs = self.render_node(node.value)
        op = self.expression_ops['Aug' + node.op.__class__.__name__]
        return '%s %s %s' % (target, op, rhs)

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
            if node.id in DEFAULT_FUNCTIONS:
                f = DEFAULT_FUNCTIONS[node.id]
                return f
            # special workaround for the "int" function
            if node.id == 'int':
                return sympy.Function("int_")
            else:
                return sympy.Function(node.id)
        else:
            if node.attr in DEFAULT_FUNCTIONS:
                return DEFAULT_FUNCTIONS[node.attr]
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
        if node.id in DEFAULT_CONSTANTS:
            return DEFAULT_CONSTANTS[node.id]
        elif node.id in ['t', 'dt']:
            return sympy.Symbol(node.id, real=True, positive=True)
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
            # Nothing to do
            return self.render_node(node.operand)
        elif op_name == 'USub':
            return -self.render_node(node.operand)
        elif op_name == 'Not':
            return sympy.Not(self.render_node(node.operand))
        else:
            raise ValueError('Unknown unary operator: ' + op_name)


def check_expression_for_multiple_stateful_functions(expr, variables):
    identifiers = re.findall(r'\w+', expr)
    # Don't bother counting if we don't have any duplicates in the first place
    if len(identifiers) == len(set(identifiers)):
        return
    identifier_count = Counter(identifiers)
    for identifier, count in identifier_count.items():
        var = variables.get(identifier, None)
        if var is None:
            continue
        if count > 1 and not var.stateless:
            raise NotImplementedError(('The expression "{expr}" contains '
                                       'more than one call of {func}, this '
                                       'is currently not supported since '
                                       '{func} is a stateful function and '
                                       'its multiple calls might be '
                                       'treated incorrectly (e.g.'
                                       '"rand() - rand()" could be '
                                       ' simplified to '
                                       '"0.0").').format(expr=expr, func=identifier))



class SympyPrinter(StrPrinter):
    """
    Printer that overrides the printing of some basic sympy objects. E.g.
    print "a and b" instead of "And(a, b)".
    """

    def _print_And(self, expr):
        return ' and '.join(['(%s)' % self.doprint(arg) for arg in expr.args])

    def _print_Or(self, expr):
        return ' or '.join(['(%s)' % self.doprint(arg) for arg in expr.args])

    def _print_Not(self, expr):
        if len(expr.args) != 1:
            raise AssertionError('"Not" with %d arguments?' % len(expr.args))
        return 'not (%s)' % self.doprint(expr.args[0])

    def _print_Relational(self, expr):
        return '%s %s %s' % (self.parenthesize(expr.lhs, precedence(expr)),
                             self._relationals.get(expr.rel_op) or expr.rel_op,
                             self.parenthesize(expr.rhs, precedence(expr)))

    def _print_Function(self, expr):
        # Special workaround for the int function
        if expr.func.__name__ == 'int_':
            return "int(%s)" % self.stringify(expr.args, ", ")
        elif expr.func.__name__ == 'Mod':
            return '((%s)%%(%s))' % (self.doprint(expr.args[0]), self.doprint(expr.args[1]))
        else:
            return expr.func.__name__ + "(%s)" % self.stringify(expr.args, ", ")


PRINTER = SympyPrinter()


def str_to_sympy(expr, variables=None):
    """
    Parses a string into a sympy expression. There are two reasons for not
    using `sympify` directly: 1) sympify does a ``from sympy import *``,
    adding all functions to its namespace. This leads to issues when trying to
    use sympy function names as variable names. For example, both ``beta`` and
    ``factor`` -- quite reasonable names for variables -- are sympy functions,
    using them as variables would lead to a parsing error. 2) We want to use
    a common syntax across expressions and statements, e.g. we want to allow
    to use `and` (instead of `&`) and function names like `ceil` (instead of
    `ceiling`).

    Parameters
    ----------
    expr : str
        The string expression to parse.
    variables : dict, optional
        Dictionary mapping variable/function names in the expr to their
        respective `Variable`/`Function` objects.

    Returns
    -------
    s_expr
        A sympy expression

    Raises
    ------
    SyntaxError
        In case of any problems during parsing.
    """
    if variables is None:
        variables = {}
    check_expression_for_multiple_stateful_functions(expr, variables)

    # We do the actual transformation in a separate function that is cached
    # If we cached `str_to_sympy` itself, it would also use the contents of the
    # variables dictionary as the cache key, while it is only used for the check
    # above and does not affect the translation to sympy
    s_expr = SympyRender().render_expr(expr)
    return s_expr


def sympy_to_str(sympy_expr):
    """
    sympy_to_str(sympy_expr)

    Converts a sympy expression into a string. This could be as easy as 
    ``str(sympy_exp)`` but it is possible that the sympy expression contains
    functions like ``Abs`` (for example, if an expression such as
    ``sqrt(x**2)`` appeared somewhere). We do want to re-translate ``Abs`` into
    ``abs`` in this case.
    
    Parameters
    ----------
    sympy_expr : sympy.core.expr.Expr
        The expression that should be converted to a string.
        
    Returns
    str_expr : str
        A string representing the sympy expression.
    """
    # replace the standard functions by our names if necessary
    replacements = dict((f, sympy.Function(name)) for name, f in DEFAULT_FUNCTIONS.items() if str(f) != name)

    # replace constants with our names as well
    replacements.update(dict((c, sympy.Symbol(name)) for name, c in DEFAULT_CONSTANTS.items() if str(c) != name))

    # Replace the placeholder argument by an empty symbol
    replacements[sympy.Symbol('_placeholder_arg')] = sympy.Symbol('')
    atoms = (sympy_expr.atoms() | {f.func for f in sympy_expr.atoms(sympy.Function)})
    for old, new in replacements.items():
        if old in atoms:
            sympy_expr = sympy_expr.subs(old, new)
    expr = PRINTER.doprint(sympy_expr)

    return expr
