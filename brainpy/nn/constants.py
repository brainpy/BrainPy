# -*- coding: utf-8 -*-


__all__ = [
  'PASS_SEQUENCE',
  'PASS_NAME_DICT',
  'PASS_NODE_DICT',
  'DATA_PASS_TYPES',
  'DATA_PASS_FUNC',
  'register_data_pass_type',
]

PASS_SEQUENCE = 'seq_of_data'
PASS_NAME_DICT = 'dict_of_name_data'
PASS_NODE_DICT = 'dict_of_node_data'

DATA_PASS_TYPES = [
  PASS_SEQUENCE,
  PASS_NAME_DICT,
  PASS_NODE_DICT,
]

DATA_PASS_FUNC = {
  PASS_SEQUENCE: lambda a: (a if (a is None) else tuple(a.values())),
  PASS_NAME_DICT: lambda a: ({node.name: data for node, data in a.items()}
                             if (a is not None) else None),
  PASS_NODE_DICT: lambda a: a,
}


def register_data_pass_type(name, func):
  if name in DATA_PASS_TYPES:
    raise ValueError(f'Data pass type "{name}" has been registered.')
  DATA_PASS_FUNC[name] = func
