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

check_shape_consistency = check.is_shape_consistency
check_shape_broadcastable = check.is_shape_broadcastable
check_shape_except_batch = check.check_shape_except_batch
check_shape = check.check_shape
check_dict_data = check.is_dict_data
check_callable = check.is_callable
check_initializer = check.is_initializer
check_connector = check.is_connector
check_float = check.is_float
check_integer = check.is_integer
check_string = check.is_string
check_sequence = check.is_sequence
check_mode = check.is_subclass
serialize_kwargs = check.serialize_kwargs
