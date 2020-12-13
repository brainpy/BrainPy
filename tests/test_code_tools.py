# -*- coding: utf-8 -*-

import ast
from pprint import pprint
from brainpy.tools.codes import CodeLineFormatter

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


tree = ast.parse(code.strip())
getter = CodeLineFormatter()
getter.visit(tree)

pprint(getter.lefts)
pprint(getter.rights)
pprint(getter.lines)
