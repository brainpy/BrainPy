# -*- coding: utf-8 -*-

from pprint import pprint

__all__ = [
    'show_code_scope'
]


def show_code_scope(code_scope, ignores=()):
    scope = {k: v for k, v in code_scope.items() if k not in ignores}
    pprint(scope)


