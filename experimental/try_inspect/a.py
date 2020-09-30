

import inspect

def func1(a=2, b=1, c=None):
    return

print(inspect.getfullargspec(func1))
print(inspect.getcallargs(func1))
