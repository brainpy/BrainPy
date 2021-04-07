# -*- coding: utf-8 -*-

import ast
from pprint import pprint
from brainpy.tools.codes import CodeLineFormatter
from brainpy.tools.codes import find_atomic_op

code ='''
for d in range(10):
    if d > 0:
      a = 0
    elif d > -1:
      if d > -0.5:
        c = g()
      else:
        d = g()
    else:
      b['b'] = 0
      c = f()
      while \
      b > 10:
        b += 1
else:
    d = hh()
'''


# tree = ast.parse(code.strip())
# getter = CodeLineFormatter()
# getter.visit(tree)
#
# pprint(getter.lefts)
# pprint(getter.rights)
# pprint(getter.lines)


def test_code_tools():
    tree = ast.parse(code.strip())
    getter = CodeLineFormatter()
    getter.visit(tree)
    assert getter.lefts == ['d', 'a', 'c', 'd', "b['b']", 'c', 'b', 'd']


def test_find_atomic_op_by_assign():
    code_line = "post['inp'] = - post['inp'] * ST['g']"
    getter = find_atomic_op(code_line, {'inp': 0})
    assert getter.left is None and getter.right is None

    code_line = "post['inp'] = - post['inp'] + ST['g']"
    getter = find_atomic_op(code_line, {'inp': 0})
    assert getter.left == 'post[0]'
    assert getter.right == "2 * post['inp'] + ST['g']"

    code_line = "post['inp'] = post['inp'] + ST['g']"
    getter = find_atomic_op(code_line, {'inp': 0})
    assert getter.left == 'post[0]'
    assert getter.right == "ST['g']"

    code_line = "post['inp'] = post['inp'] - ST['g']"
    getter = find_atomic_op(code_line, {'inp': 0})
    assert getter.left == 'post[0]'
    assert getter.right == "- ST['g']"

    code_line = "post['inp'] = - post['inp'] - ST['g']"
    getter = find_atomic_op(code_line, {'inp': 0})
    assert getter.left == 'post[0]'
    assert getter.right == "2 * post['inp'] - ST['g']"

    code_line = "post['inp'] = ST['g'] + post['inp']"
    getter = find_atomic_op(code_line, {'inp': 0})
    assert getter.left == 'post[0]'
    assert getter.right == "ST['g']"

    code_line = "post['inp'] =  + ST['g'] - post['inp']"
    getter = find_atomic_op(code_line, {'inp': 0})
    assert getter.left == 'post[0]'
    assert getter.right == "+ST['g'] + 2 * post['inp']"

    code_line = "post['inp'] = - ST['g'] + post['inp'] "
    getter = find_atomic_op(code_line, {'inp': 0})
    assert getter.left == 'post[0]'
    assert getter.right == "-ST['g']"

    code_line = "post['inp'] = - ST['g'] - post['inp'] "
    getter = find_atomic_op(code_line, {'inp': 0})
    assert getter.left == 'post[0]'
    assert getter.right == "-ST['g'] + 2 * post['inp']"

    code_line = "post['inp'] = - ST['g'] - (-post['inp'])"
    getter = find_atomic_op(code_line, {'inp': 0})
    assert getter.left == 'post[0]'
    assert getter.right == "-ST['g']"


def test_find_atomic_op_by_augassign():
    code_line = "post['inp'] -= post['inp'] * ST['g']"
    getter = find_atomic_op(code_line, {'inp': 0})
    assert getter.left == 'post[0]'
    assert getter.right == "- post['inp'] * ST['g']"

    code_line = "post['inp'] -= ST['g']"
    getter = find_atomic_op(code_line, {'inp': 0})
    assert getter.left == 'post[0]'
    assert getter.right == "- ST['g']"

    code_line = "post['inp'] += ST['g']"
    getter = find_atomic_op(code_line, {'inp': 0})
    assert getter.left == 'post[0]'
    assert getter.right == "ST['g']"




# test_find_atomic_op_by_assign()
# test_find_atomic_op_by_augassign()

