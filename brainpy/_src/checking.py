# -*- coding: utf-8 -*-

from brainpy import check

from brainpy._src.deprecations import deprecation_getattr2

__deprecations = {
  'check_shape_consistency': ('brainpy.checking.check_shape_consistency',
                              'brainpy.check.is_shape_consistency',
                              check.is_shape_consistency),
  'check_shape_broadcastable': ('brainpy.checking.check_shape_broadcastable',
                                'brainpy.check.is_shape_broadcastable',
                                check.is_shape_broadcastable),
  'check_shape_except_batch': ('brainpy.checking.check_shape_except_batch',
                               'brainpy.check.check_shape_except_batch',
                               check.check_shape_except_batch),
  'check_shape': ('brainpy.checking.check_shape',
                  'brainpy.check.check_shape',
                  check.check_shape),
  'check_dict_data': ('brainpy.checking.check_dict_data',
                      'brainpy.check.is_dict_data',
                      check.is_dict_data),
  'check_callable': ('brainpy.checking.check_callable',
                     'brainpy.check.is_callable',
                     check.is_callable),
  'check_initializer': ('brainpy.checking.check_initializer',
                        'brainpy.check.is_initializer',
                        check.is_initializer),
  'check_connector': ('brainpy.checking.check_connector',
                      'brainpy.check.is_connector',
                      check.is_connector),
  'check_float': ('brainpy.checking.check_float',
                  'brainpy.check.is_float',
                  check.is_float),
  'check_integer': ('brainpy.checking.check_integer',
                    'brainpy.check.is_integer',
                    check.is_integer),
  'check_string': ('brainpy.checking.check_string',
                   'brainpy.check.is_string',
                   check.is_string),
  'check_sequence': ('brainpy.checking.check_sequence',
                     'brainpy.check.is_sequence',
                     check.is_sequence),
  'check_mode': ('brainpy.checking.check_mode',
                 'brainpy.check.is_subclass',
                 check.is_subclass),
  'serialize_kwargs': ('brainpy.checking.serialize_kwargs',
                       'brainpy.check.serialize_kwargs',
                       check.serialize_kwargs),
}
__getattr__ = deprecation_getattr2('brainpy.checking', __deprecations)
del deprecation_getattr2
