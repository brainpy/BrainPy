# -*- coding: utf-8 -*-

from pprint import pprint

__all__ = [
    'show_code_scope',
    'show_code_str',
]


def show_code_str(func_code):
    print(func_code)
    print()


def show_code_scope(code_scope, ignores=()):
    scope = {k: v for k, v in code_scope.items() if k not in ignores}
    pprint(scope)
    print()


