# -*- coding: utf-8 -*-

from pprint import pprint

from brainpy.integrators.driver import base

__all__ = [
  'NumpyDiffIntDriver',
]


class NumpyDiffIntDriver(base.DiffIntDriver):
  def build(self, *args, **kwargs):
    code = '\n'.join(self.code_lines)
    self.code = code
    if self.show_code:
      print(code)
      print()
      pprint(self.code_scope)
      print()
    exec(compile(code, '', 'exec'), self.code_scope)
    new_f = self.code_scope[self.func_name]
    return new_f
