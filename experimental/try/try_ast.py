# -*- coding: utf-8 -*-
import ast
from fractions import Fraction


class IntegerWrapper(ast.NodeTransformer):
    """Wraps all integers in a call to Integer()"""

    def visit_Num(self, node):
        if isinstance(node.n, int):
            return ast.Call(func=ast.Name(id='Integer', ctx=ast.Load()),
                            args=[node], keywords=[])
        return node


class Integer(object):
    def __init__(self, value):
        self.value = value

    def __truediv__(self, other):
        if isinstance(other, Integer):
            return Fraction(numerator=self.value, denominator=other.value)


code = "print((1/10)+(2/10))"
print(code)
print()

print("Without AST transformation:")
exec(code)
print()

print("With AST transformation:")
tree = ast.parse(code)
tree = IntegerWrapper().visit(tree)
# Add lineno & col_offset to the nodes we created
ast.fix_missing_locations(tree)
co = compile(tree, "<ast>", "exec")
exec(co)