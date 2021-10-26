# -*- coding: utf-8 -*-

from pprint import pprint

from brainpy.integrators import constants, utils

_SDE_UNKNOWN_NO = 0


def basic_info(f, g):
  vdt = 'dt'
  if f.__name__.isidentifier():
    func_name = f.__name__
  elif g.__name__.isidentifier():
    func_name = g.__name__
  else:
    global _SDE_UNKNOWN_NO
    func_name = f'unknown_sde{_SDE_UNKNOWN_NO}'
  func_new_name = constants.SDE_INT + func_name
  class_kw, variables, parameters, arguments = utils.get_args(f)
  return vdt, variables, parameters, arguments, func_new_name

