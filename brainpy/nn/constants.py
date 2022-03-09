# -*- coding: utf-8 -*-


__all__ = [
  'PASS_ONLY_ONE',
  'PASS_SEQUENCE',
  'PASS_NAME_DICT',
  'PASS_NODE_DICT',
  'DATA_PASS_TYPES',
  'DATA_PASS_FUNC',
  'register_data_pass_type',
]

PASS_ONLY_ONE = 'only_one_data'
PASS_SEQUENCE = 'seq_of_data'
PASS_NAME_DICT = 'dict_of_name_data'
PASS_TYPE_DICT = 'dict_of_type_data'
PASS_NODE_DICT = 'dict_of_node_data'

DATA_PASS_TYPES = [
  PASS_ONLY_ONE,
  PASS_SEQUENCE,
  PASS_NAME_DICT,
  PASS_TYPE_DICT,
  PASS_NODE_DICT,
]


def _pass_only_one(data):
  if len(data) > 1:
    raise ValueError(f'"PASS_ONLY_ONE" type only support one '
                     f'feedforward/feedback input. But we got {len(data)}.')
  return tuple(data.values())[0]


DATA_PASS_FUNC = {
  PASS_SEQUENCE: lambda a: (None if (a is None) else tuple(a.values())),
  PASS_NAME_DICT: lambda a: ({node.name: data for node, data in a.items()} if (a is not None) else None),
  PASS_TYPE_DICT: lambda a: ({type(node).__name__: data for node, data in a.items()} if (a is not None) else None),
  PASS_NODE_DICT: lambda a: a,
  PASS_ONLY_ONE: _pass_only_one,
}


def register_data_pass_type(name, func):
  if name in DATA_PASS_TYPES:
    raise ValueError(f'Data pass type "{name}" has been registered.')
  DATA_PASS_FUNC[name] = func
