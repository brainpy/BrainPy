# SPDX-License-Identifier: GPL-3.0-only
import ast
import sys
import argparse
import operator
import collections

import sympy

from sympy import Basic
from sympy.printing.str import StrPrinter


class CustomStrPrinter(StrPrinter):
    def _print_Dummy(self, expr):
        return expr.name


Basic.__str__ = lambda self: CustomStrPrinter().doprint(self)


def Dummy(name):
    return sympy.Dummy(name, integer=True, nonnegative=True)


class VisitorBase(ast.NodeVisitor):
    def __init__(self, source):
        self._source_lines = source.split('\n')
        self.scope_stack = []
        self.current_scope = None
        self.unhandled = set()
        self.log_lines = collections.defaultdict(list)
        self.current_line = [None]

    def push_scope(self, s):
        self.scope_stack.append(self.current_scope)
        self.current_scope = s

    def pop_scope(self):
        self.current_scope = self.scope_stack.pop()

    def visit(self, node):
        if isinstance(node, list):
            for x in node:
                self.visit(x)
            return
        try:
            current_line = node.lineno - 1
        except AttributeError:
            current_line = None
        self.current_line.append(current_line)
        try:
            return super(VisitorBase, self).visit(node)
        except:
            self.source_backtrace(node, sys.stderr)
            raise
        finally:
            self.current_line.pop()

    def log(self, s):
        self.log_lines[self.current_line[-1]].append(s)

    def source_backtrace(self, node, file):
        try:
            lineno = node.lineno
            col_offset = node.col_offset
        except AttributeError:
            lineno = col_offset = None
        print('At node %s' % node, file=file)
        if lineno is not None and lineno > 0:
            print(self._source_lines[lineno - 1], file=file)
            print(' ' * col_offset + '^', file=file)

    def generic_visit(self, node):
        if type(node) not in self.unhandled:
            self.source_backtrace(node, sys.stderr)
            print("%s unhandled" % (type(node).__name__,), file=sys.stderr)
        self.unhandled.add(type(node).__name__)

    def visit_children(self, node):
        for child in ast.iter_child_nodes(node):
            self.visit(child)

    def visit_Module(self, node):
        self.visit_children(node)

    def print_line(self, i):
        line = self._source_lines[i]
        if i not in self.log_lines:
            print(line)
            return
        length = max(len(line), 38)
        for j, c in enumerate(self.log_lines[i]):
            if j == 0:
                l = line
            else:
                l = ''
            print('%s# %s' % (l.ljust(length), c))
        del self.log_lines[i]


class Scope(object):
    def __init__(self, parent, parameters):
        self._parent = parent
        self._locals = {
            n: Dummy(n)
            for n in parameters
        }
        self._effects = {}
        self._output = None

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, x):
        if self._output is None:
            self._output = x
        else:
            raise AttributeError("Output is already set")

    @property
    def changed_vars(self):
        return set(self[v] for v in self._effects.keys())

    def affect(self, expr):
        sub = {
            self[n]: e
            for n, e in self._effects.items()
        }
        return expr.subs(sub)

    def __getitem__(self, name):
        if isinstance(name, ast.AST):
            raise TypeError("Try to lookup a %s" % (name,))
        elif isinstance(name, sympy.Symbol):
            return name
        try:
            return self._locals[name]
        except KeyError:
            if self._parent is None:
                raise KeyError(name)
            return self._parent[name]

    def set_effect(self, name, expr):
        self._effects[name] = expr

    def add_effect(self, name, expr):
        if isinstance(name, ast.AST):
            raise TypeError("Try to add_effect on a %s" % (name,))
        if expr is None:
            raise TypeError("Try to add_effect with None")
        if isinstance(name, str):
            try:
                name = self[name]
            except KeyError:
                self._locals[name] = Dummy(name)
                name = self._locals[name]
        self._effects[name] = self.affect(expr)


def repeated(n, i, e, a, b):
    # let n_a = n; n_{a+k+1} = e(n=n_{a+k}, i=a+k+1)
    # return n_b
    if e.has(i):
        if e.has(n):
            if not (e - n).has(n):
                term = e - n
                return n + sympy.summation(term, (i, a, b))
            raise NotImplementedError("has i and n")
        else:
            return e.subs(i, b)
    else:
        if e.has(n):
            if not (e - n).has(n):
                term = e - n
                return n + term * (b - a + 1)
            c, args = e.as_coeff_add(n)
            arg, = args
            if not (arg / n).simplify().has(n):
                coeff = arg / n
                # print("Coefficient is %s, iterations is %s" %
                #       (coeff, (b-a+1)))
                return n * coeff ** (b - a + 1)
            raise NotImplementedError
        else:
            return e


def termination_function(e):
    if isinstance(e, (sympy.LessThan, sympy.GreaterThan)):
        c = 0
    elif isinstance(e, (sympy.StrictLessThan, sympy.StrictGreaterThan)):
        c = 1
    else:
        raise NotImplementedError(str(type(e)))
    return e.gts - e.lts - c


class Visitor(VisitorBase):
    def visit_Module(self, node):
        linenos = [v.lineno - 1 for v in node.body]
        linenos[0] = 0
        linenos.append(len(self._source_lines))
        for v, i, j in zip(node.body, linenos[:-1], linenos[1:]):
            self.visit(v)
            for k in range(i, j):
                self.print_line(k)

    def visit_FunctionDef(self, node):
        # print((' Function %s (line %s) ' % (node.name, node.lineno))
        #       .center(79, '='))
        self.push_scope(Scope(self.current_scope, [arg.arg for arg in node.args.args]))
        self.steps = Dummy('T')
        self.current_scope.add_effect(self.steps, sympy.S.One)
        self.visit(node.body)
        def BigO(e):
            try:
                return sympy.Order(e, (self.current_scope[node.args.args[0].arg], sympy.oo)).args[0]
            except NotImplementedError:
                return e
        self.log("Function %s: O(%s)" %
                 (node.name,
                  BigO(self.current_scope.affect(self.steps))))
        if self.current_scope.output is not None:
            print("Result: %s" % (self.current_scope.affect(self.current_scope.output),))
        # for n, e in self.current_scope._effects.items():
        #     ee = BigO(e)
        #     if ee.args:
        #         print("%s:\n%s = O(%s)" % (n, e, ee.args[0]))
        #     else:
        #         print("%s:\n%s = O(??)" % (n, e))
        self.pop_scope()
        if self.unhandled:
            print("Unhandled types: %s" %
                  ', '.join(str(c) for c in self.unhandled))
            self.unhandled = type(self.unhandled)()
        print('')

    def visit_Return(self, node):
        self.log("Result: %s" % (self.current_scope.affect(self.visit(node.value)),))

    def visit_Assign(self, node):
        target, = node.targets
        name = target.id
        expr = self.visit(node.value)
        self.current_scope.add_effect(name, expr)
        self.log("%s = %s" % (name, expr))

    def visit_BinOp(self, node):
        return self.binop(self.visit(node.left), node.op, self.visit(node.right))

    def visit_Compare(self, node):
        left = self.visit(node.left)
        rights = [self.visit(c) for c in node.comparators]
        lefts = [left] + rights[:-1]
        res = None
        for left, op, right in zip(lefts, node.ops, rights):
            r = self.binop(left, op, right)
            if res is None:
                res = r
            else:
                res = self.binop(res, ast.And, r)
        return res

    def binop(self, left, op, right):
        if isinstance(op, ast.AST):
            op = type(op)
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.And: sympy.And,
            ast.Lt: sympy.StrictLessThan,
            ast.LtE: sympy.LessThan,
            ast.Gt: sympy.StrictGreaterThan,
            ast.GtE: sympy.GreaterThan,
        }
        try:
            return operators[op](left, right)
        except KeyError:
            raise NotImplementedError("op %s" % (op,))

    def visit_AugAssign(self, node):
        target = node.target
        name = target.id
        expr = self.visit(node.value)
        aug_expr = self.binop(self.current_scope[name], node.op, expr)
        self.current_scope.add_effect(name, aug_expr)
        self.log("%s = %s" % (name, aug_expr))

    def visit_Num(self, node):
        return sympy.Rational(node.n)

    def visit_Name(self, node):
        return self.current_scope[node.id]

    def visit_For(self, node):
        if not isinstance(node.iter, ast.Call):
            raise NotImplementedError('for of non-Call')
        if node.iter.func.id != 'range':
            raise NotImplementedError('for of non-range')
        args = node.iter.args
        if len(args) == 1:
            a = sympy.S.Zero
            b = self.visit(args[0])
        elif len(args) == 2:
            a, b = self.visit(args[0]), self.visit(args[1])
        else:
            raise NotImplementedError('3-arg range')

        outer_scope = self.current_scope
        inner_scope = Scope(outer_scope, [node.target.id])
        t0 = outer_scope.affect(self.steps)
        k = inner_scope[node.target.id]
        self.push_scope(inner_scope)
        inner_scope.add_effect(self.steps, self.steps + sympy.S.One)
        self.visit(node.body)
        self.pop_scope()

        iterations = outer_scope.affect(b - a)
        effects = {}
        for n, e in self.topological_order(inner_scope._effects):
            ee = repeated(n, k, e.subs(effects), a, b - 1)
            self.log("%s = %s" % (n, outer_scope.affect(ee)))
            effects[n] = ee
        for n, e in effects.items():
            outer_scope.add_effect(n, e)
        t1 = outer_scope.affect(self.steps)
        self.log("%s iterations, %s steps" % (iterations, t1 - t0))

    def visit_While(self, node):
        test = self.visit(node.test)
        outer_scope = self.current_scope
        inner_scope = Scope(outer_scope, [])
        t0 = outer_scope.affect(self.steps)
        self.push_scope(inner_scope)
        inner_scope.add_effect(self.steps, self.steps + sympy.S.One)
        self.visit(node.body)
        self.pop_scope()

        # Compute effects of loop after `k` iterations
        k = Dummy('k')
        effects_after_k = {}
        for n, e in self.topological_order(inner_scope._effects):
            i = Dummy('i')
            ee = e.subs(effects_after_k).subs(k, i)
            effects_after_k[n] = outer_scope.affect(repeated(n, i, ee, 1, k))
            self.log("%s = %s = %s" % (n, ee, effects_after_k[n]))
        print(effects_after_k)

        o = termination_function(test).subs(effects_after_k)

        # Compute number of iterations
        try:
            iterations = sympy.solve(o, k, dict=True)[0][k]
        except:
            raise NotImplementedError("Could not solve %s for %s" % (o, k))
        effects = {
            n: e.subs(k, iterations)
            for n, e in effects_after_k.items()
        }
        for n, e in effects.items():
            self.log('%s = %s' % (n, e))
            outer_scope.set_effect(n, e)
        t1 = outer_scope.affect(self.steps)
        self.log("%s iterations, %s steps" % (iterations, t1 - t0))

    @staticmethod
    def topological_order(effects):
        dependers = {n: [] for n, e in effects.items()}
        depcount = {}
        for n, e in effects.items():
            c = 0
            for dep in e.free_symbols:
                if dep != n and dep in dependers:
                    dependers[dep].append(n)
                    c += 1
            depcount[n] = c
        while depcount:
            try:
                n = next(n for n, d in depcount.items() if d == 0)
            except StopIteration:
                raise NotImplementedError('Recursive dependency')
            del depcount[n]
            yield (n, effects[n])
            for dep in dependers[n]:
                depcount[dep] -= 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    with open(args.filename) as fp:
        source = fp.read()
    o = ast.parse(source, args.filename, 'exec')
    visitor = Visitor(source)
    visitor.visit(o)


if __name__ == "__main__":
    main()