# -*- coding: utf-8 -*-

from brainpy import check


__all__ = [
  'check_shape_consistency',
  'check_shape_broadcastable',
  'check_shape_except_batch',
  'check_shape',
  'check_dict_data',
  'check_callable',
  'check_initializer',
  'check_connector',
  'check_float',
  'check_integer',
  'check_string',
  'check_sequence',
  'check_mode',

  'serialize_kwargs',
]

check_shape_consistency = check.check_shape_consistency
check_shape_broadcastable = check.check_shape_broadcastable
check_shape_except_batch = check.check_shape_except_batch
check_shape = check.check_shape
check_dict_data = check.check_dict_data
check_callable = check.check_callable
check_initializer = check.check_initializer
check_connector = check.check_connector
check_float = check.check_float
check_integer = check.check_integer
check_string = check.check_string
check_sequence = check.check_sequence
check_mode = check.check_mode
serialize_kwargs = check.serialize_kwargs
