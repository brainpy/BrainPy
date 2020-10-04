# -*- coding: utf-8 -*-


func = '''
def f(a, b, c):
    pass

'''

variables = {}
exec(compile(func, '', 'exec'), variables)

from pprint import pprint
pprint(variables)

f = variables['f']

import inspect
print(inspect.getfullargspec(f))
