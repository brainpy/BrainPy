# -*- coding: utf-8 -*-


import ast
from brainpy import tools

__all__ = [
  'control_transform',
]


def control_transform(f):
  return f

'''
if V >= self.V_th:
  self.V[i] = self.V_reset
  self.spike[i] = 1.
  self.t_last_spike[i] = _t
  self.refractory[i] = True
else:
  self.spike[i] = 0.
  self.V[i] = V
  self.refractory[i] = False
'''

string = '''
def update(self, _t, _i):
  for i in range(self.num):
    if _t - self.t_last_spike[i] <= self.t_refractory:
      self.refractory[i] = True
    else:
      V = self.int_V(self.V[i], _t, self.input[i])
      if V >= self.V_th:
        self.V[i] = self.V_reset
        self.spike[i] = 1.
        self.t_last_spike[i] = _t
        self.refractory[i] = True
      else:
        self.spike[i] = 0.
        self.V[i] = V
        self.refractory[i] = False
    self.input[i] = 0.
'''


class NodeTransformer(ast.NodeTransformer):
  pass



def _cond_syntax(tree):
  if len(tree.orelse) == 1 and isinstance(tree.orelse[0], ast.If):
    raise ValueError('Cannot analyze multiple conditional syntax with "if .. elif .. ". '
                     'Please code the code with "if .. else .. " syntax.')
  # code for "if" flow


  # code for "else" flow




def _for_loop_syntax(tree):
  pass


def _while_loop(tree):
  pass



def _transform(code):
  tree = ast.parse(code)

  # arguments
  args = tools.ast2code(ast.fix_missing_locations(tree.body[0].args)).split(',')
