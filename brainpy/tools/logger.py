# -*- coding: utf-8 -*-

from pprint import pprint

from ..integration.sympy_tools import FUNCTION_MAPPING
from ..integration.sympy_tools import CONSTANT_MAPPING

__all__ = [
    'show_code_scope',
    'show_code_str',
]


def show_code_str(func_code):
    print(func_code)
    print()


def show_code_scope(code_scope, ignores=()):
    scope = {}
    for k, v in code_scope.items():
        if k in ignores:
            continue
        if k in CONSTANT_MAPPING:
            continue
        if k in FUNCTION_MAPPING:
            continue
        scope[k] = v
    pprint(scope)
    print()


